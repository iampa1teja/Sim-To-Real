"""Static indexing regressions: scopes, safe inventory and syntax-only parsing."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "tools/code_explorer/indexer.py"
spec = importlib.util.spec_from_file_location("code_explorer_indexer", SOURCE)
indexer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(indexer)


class IndexerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True)

    def put(self, path, text):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def build(self):
        return indexer.build_index(self.root)

    def symbol(self, result, qualified, path="src/demo.py"):
        return next(s for s in result["symbols"] if s["file"] == path and s["qualifiedName"] == qualified)

    @staticmethod
    def links(result, symbol):
        return [e["target"] for e in result["edges"] if e["source"] == symbol["id"] and e["type"] == "calls"]

    def test_nested_functions_methods_lambdas_and_definition_locations(self):
        self.put("src/demo.py", '''"""Module purpose."""
def helper(value):
    """Explain the helper."""
    return value

class Worker:
    async def run(self, value: int = 2) -> int:
        def nested():
            return helper(value)
        callback = lambda x: helper(x)
        return nested() + callback(value)

raise RuntimeError("Indexer must not import this module")
''')
        r = self.build()
        helper = self.symbol(r, "helper")
        method = self.symbol(r, "Worker.run")
        nested = self.symbol(r, "Worker.run.nested")
        callback = self.symbol(r, "Worker.run.callback")
        self.assertEqual(r["stats"]["functions"], 4)
        self.assertEqual(method["kind"], "method")
        self.assertTrue(method["async"])
        self.assertEqual(helper["line"], 2)
        self.assertEqual(helper["endLine"], 4)
        self.assertEqual(helper["docstring"], "Explain the helper.")
        self.assertIn("value: int", method["signature"])
        self.assertIn("=2", method["signature"])
        self.assertEqual(nested["parentId"], method["id"])
        self.assertIn(helper["id"], self.links(r, nested))
        self.assertIn(helper["id"], self.links(r, callback))
        self.assertCountEqual(self.links(r, method), [nested["id"], callback["id"]])
        self.assertEqual(r["files"][0]["summary"], "Module purpose.")

    def test_call_resolution_does_not_guess_shadowed_or_ambiguous_names(self):
        self.put("src/demo.py", '''def target(): pass
def with_parameter(target):
    target()
def with_assignment():
    target = object()
    target()
def repeated(): pass
def repeated(): pass
def ambiguous():
    repeated()
def visible():
    target()
''')
        r = self.build()
        for name in ["with_parameter", "with_assignment", "ambiguous"]:
            self.assertEqual(self.links(r, self.symbol(r, name)), [])
        self.assertEqual(self.links(r, self.symbol(r, "visible")), [self.symbol(r, "target")["id"]])
        self.assertGreater(r["coverage"]["unresolvedCallSites"], 0)

    def test_exception_and_match_capture_bindings_shadow_outer_functions(self):
        self.put("src/demo.py", '''def target(): pass
def caught():
    try: pass
    except Exception as target: target()
def matched(value):
    match value:
        case {"x": target}: target()
def starred(value):
    match value:
        case [*target]: target()
def remainder(value):
    match value:
        case {**target}: target()
''')
        r = self.build()
        for name in ["caught", "matched", "starred", "remainder"]:
            self.assertEqual(self.links(r, self.symbol(r, name)), [])

    def test_import_aliases_relative_imports_and_module_calls(self):
        self.put("src/pkg/helper.py", "def compute(): return 4\n")
        self.put("src/pkg/__init__.py", "")
        self.put("src/pkg/runner.py", '''from .helper import compute as local_compute
from . import helper as module_helper
import pkg.helper
import pkg.helper as direct
def run():
    return local_compute() + module_helper.compute() + pkg.helper.compute() + direct.compute()
''')
        r = self.build()
        target = self.symbol(r, "compute", "src/pkg/helper.py")["id"]
        run = self.symbol(r, "run", "src/pkg/runner.py")
        self.assertEqual(len(self.links(r, run)), 1)  # same source location deduplicates the four references
        self.assertEqual(self.links(r, run), [target])
        imports = [e for e in r["edges"] if e["type"] == "imports"]
        self.assertTrue(imports)
        self.assertTrue(all(e["target"] == "src/pkg/helper.py" for e in imports))

    def test_methods_use_lexical_receiver_but_bare_names_skip_class(self):
        self.put("src/demo.py", '''def common(): pass
class Worker:
    def common(self): pass
    def run(this):
        this.common()
        common()
    def shadow(self):
        def callback(self): self.common()
        return callback
    @staticmethod
    def static(self): self.common()
    def replaced(self):
        self = object()
        self.common()
''')
        r = self.build()
        self.assertCountEqual(self.links(r, self.symbol(r, "Worker.run")),
                              [self.symbol(r, "Worker.common")["id"], self.symbol(r, "common")["id"]])
        self.assertEqual(self.links(r, self.symbol(r, "Worker.shadow.callback")), [])
        self.assertEqual(self.links(r, self.symbol(r, "Worker.static")), [])
        self.assertEqual(self.links(r, self.symbol(r, "Worker.replaced")), [])

    def test_inventory_includes_untracked_assets_and_redacts_secret_symlinks(self):
        self.put(".gitignore", "ignored.log\n")
        self.put("src/demo.py", "def visible(): pass\n")
        self.put("ignored.log", "ignored")
        self.put(".env", "API_TOKEN=must-never-appear")
        self.put(".netrc", "machine localhost login user password must-never-appear")
        self.put(".env.secret.example", "API_TOKEN=must-never-appear")
        self.put("docs/.env.example", "TOKEN=example")
        self.put("assets/robot.usdz", "binary placeholder")
        self.put("outputs/generated.py", "def hidden_output(): pass")
        self.put("web/node_modules/vendor.js", "function vendor() {}")
        (self.root / "alias.py").symlink_to(self.root / ".env")
        with tempfile.TemporaryDirectory() as outside:
            p = Path(outside) / "file.py"
            p.write_text("def outside_function(): pass")
            (self.root / "outside.py").symlink_to(p)
            r = self.build()
        files = {f["path"]: f for f in r["files"]}
        self.assertIn("assets/robot.usdz", files)
        self.assertNotIn("ignored.log", files)
        self.assertNotIn("outputs/generated.py", files)
        self.assertNotIn("web/node_modules/vendor.js", files)
        self.assertEqual(files[".env"]["parseStatus"], "redacted")
        self.assertEqual(files[".netrc"]["parseStatus"], "redacted")
        self.assertEqual(files[".env.secret.example"]["parseStatus"], "redacted")
        self.assertEqual(files["alias.py"]["parseStatus"], "redacted")
        self.assertEqual(files["outside.py"]["parseStatus"], "external-symlink")
        self.assertEqual(files["docs/.env.example"]["parseStatus"], "text")
        self.assertNotIn("must-never-appear", json.dumps(r))
        self.assertNotIn("outside_function", json.dumps(r))

    def test_syntax_errors_are_visible_and_do_not_remove_other_files(self):
        self.put("src/broken.py", "def broken(:\n")
        self.put("src/demo.py", "def good(): pass\n")
        r = self.build()
        self.assertEqual(len(r["coverage"]["parseErrors"]), 1)
        self.assertEqual(r["coverage"]["parseErrors"][0]["file"], "src/broken.py")
        self.assertTrue(self.symbol(r, "good"))

    def test_non_git_fallback_still_lists_source(self):
        import shutil
        shutil.rmtree(self.root / ".git")
        self.put("src/demo.py", "def good(): pass\n")
        r = self.build()
        self.assertIn("fallback", r["coverage"]["inventory"])
        self.assertTrue(self.symbol(r, "good"))

    def test_shell_pattern_coverage_is_explicit(self):
        self.put("run.sh", "#!/bin/bash\nfunction first() {\n  true\n}\nsecond () {\n  false\n}\n")
        r = self.build()
        self.assertEqual([s["name"] for s in r["symbols"]], ["first", "second"])
        self.assertEqual(r["files"][0]["parseStatus"], "shell-pattern")
        self.assertIn("Pattern-based", r["coverage"]["shellParser"])

    def test_python_shebang_entrypoint_and_infra_groups(self):
        self.put("s2r", "#!/usr/bin/env python3\ndef main(): pass\n")
        self.put("pyproject.toml", "[project]\nname='sample'\n")
        self.put("tests/test_code_explorer.py", "def test_one(): pass\n")
        r = self.build()
        files = {f["path"]: f for f in r["files"]}
        self.assertEqual(files["s2r"]["parseStatus"], "python-ast")
        self.assertEqual(files["pyproject.toml"]["group"], "infra")
        self.assertEqual(files["tests/test_code_explorer.py"]["group"], "tests")

    def test_babel_functions_callbacks_typescript_and_imports(self):
        self.put("web/helper.ts", "export function compute(value: number): number { return value + 1; }\n")
        self.put("web/App.tsx", '''import { compute as calculate } from './helper';
export const App = () => {
    const twice = (value: number) => calculate(value) * 2;
    return <button onClick={() => twice(2)}>Run</button>;
};
class Worker {
    run() { return this.work(); }
    work() { return calculate(4); }
}
const named = function recursive() { return recursive(); };
named();
''')
        r = self.build()
        if r["coverage"]["javascriptParser"] != "Babel":
            self.skipTest("Install explorer web dependencies to exercise Babel")
        app = self.symbol(r, "App", "web/App.tsx")
        twice = self.symbol(r, "App.twice", "web/App.tsx")
        compute = self.symbol(r, "compute", "web/helper.ts")
        self.assertIn(compute["id"], self.links(r, twice))
        callbacks = [s for s in r["symbols"] if s.get("parentId") == app["id"] and "anonymous" in s["name"]]
        self.assertEqual(len(callbacks), 1)
        self.assertIn(twice["id"], self.links(r, callbacks[0]))
        run = self.symbol(r, "Worker.run", "web/App.tsx")
        work = self.symbol(r, "Worker.work", "web/App.tsx")
        self.assertIn(work["id"], self.links(r, run))
        named = self.symbol(r, "named", "web/App.tsx")
        self.assertIn(named["id"], self.links(r, named))

    def test_js_does_not_guess_this_bindings_or_unknown_callee_receivers(self):
        self.put("web/demo.js", '''function work() {}
function run(factory) { factory().work(); }
function object_scope() {
    const obj = {target() {}};
    target();
}
class Worker {
    work() {}
    run() {
        const obj = {callback() { this.work(); }};
        const callback = function() { this.work(); };
        const arrow = () => this.work();
    }
}
''')
        r = self.build()
        if r["coverage"]["javascriptParser"] != "Babel":
            self.skipTest("Install explorer web dependencies to exercise Babel")
        for name in ["run", "object_scope", "Worker.run.callback"]:
            self.assertEqual(self.links(r, self.symbol(r, name, "web/demo.js")), [])
        callbacks = [s for s in r["symbols"] if s["name"] == "callback" and s["kind"] != "variable"]
        self.assertEqual(len(callbacks), 2)
        for callback in callbacks:
            self.assertEqual(self.links(r, callback), [])
        arrow = self.symbol(r, "Worker.run.arrow", "web/demo.js")
        self.assertEqual(self.links(r, arrow), [self.symbol(r, "Worker.work", "web/demo.js")["id"]])

    def test_python_variables_cover_scopes_destructuring_and_source_previews(self):
        self.put("src/demo.py", '''LIMIT = 8
first, *rest = values
class Worker:
    label: str = "arm"
    def run(self):
        x, y = (1, 2)
        self.total = x + y
        x += 1
        return x
''')
        r = self.build()
        variables = [s for s in r["symbols"] if s["kind"] == "variable"]
        self.assertEqual(r["stats"]["variables"], 8)
        self.assertEqual(r["stats"]["functions"], 1)
        self.assertEqual(r["stats"]["classes"], 1)
        self.assertEqual({s["name"] for s in variables}, {"LIMIT", "first", "rest", "label", "x", "y", "self.total"})
        module = next(s for s in variables if s["name"] == "LIMIT")
        self.assertEqual(module["scopeKind"], "module")
        self.assertEqual(module["bindingScope"], "src/demo.py")
        self.assertEqual(module["valuePreview"], "8")
        field = next(s for s in variables if s["name"] == "label")
        self.assertEqual(field["scopeKind"], "class")
        method = self.symbol(r, "Worker.run")
        local = next(s for s in variables if s["name"] == "self.total")
        self.assertEqual(local["parentId"], method["id"])
        self.assertEqual(local["valuePreview"], "x + y")
        self.assertIn({"source": method["id"], "target": local["id"], "type": "contains"}, r["edges"])

    def test_variable_ids_do_not_collide_with_callable_lambdas_or_add_fake_calls(self):
        self.put("src/demo.py", "callback = lambda value: value\ncallback(1)\nvalue = 2\nvalue()\n")
        r = self.build()
        ids = [s["id"] for s in r["symbols"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len([s for s in r["symbols"] if s["name"] == "callback"]), 2)
        variable_ids = {s["id"] for s in r["symbols"] if s["kind"] == "variable"}
        call_edges = [e for e in r["edges"] if e["type"] == "calls"]
        self.assertEqual(len(call_edges), 1)
        self.assertTrue(all(e["source"] not in variable_ids and e["target"] not in variable_ids for e in call_edges))

    def test_variable_previews_redact_sensitive_bindings_and_object_keys(self):
        self.put("src/demo.py", "API_TOKEN = 'must-not-index'\nconfig = {'password': 'must-not-index'}\n")
        self.put(".env", "TOKEN=must-not-index")
        self.put(".netrc", "password must-not-index")
        r = self.build()
        self.assertNotIn("must-not-index", json.dumps(r))
        self.assertTrue(all('<redacted>' in s['valuePreview'] for s in r['symbols'] if s['kind'] == 'variable'))

    def test_js_variables_destructuring_fields_and_call_graph_separation(self):
        self.put("web/demo.ts", '''const limit = 8;
const {x: renamed, ...rest} = input;
const [first, second] = values;
class Worker {
    label: string = "arm";
    run() { let local = 2; this.total = local; return local; }
}
const callback = (value: number) => value;
callback(1);
''')
        r = self.build()
        if r["coverage"]["javascriptParser"] != "Babel":
            self.skipTest("Install explorer web dependencies to exercise Babel")
        variables = [s for s in r["symbols"] if s["kind"] == "variable"]
        self.assertEqual({s['name'] for s in variables}, {'limit', 'renamed', 'rest', 'first', 'second', 'label', 'local', 'this.total', 'callback'})
        self.assertEqual(r['stats']['variables'], 9)
        self.assertEqual(next(s for s in variables if s['name'] == 'label')['scopeKind'], 'class')
        self.assertEqual(next(s for s in variables if s['name'] == 'this.total')['scopeKind'], 'local')
        variable_ids = {s['id'] for s in variables}
        self.assertTrue(all(e['source'] not in variable_ids and e['target'] not in variable_ids for e in r['edges'] if e['type'] == 'calls'))
        self.assertEqual(len(variable_ids), len(variables))

    def test_js_private_preview_and_catch_binding_are_safe(self):
        self.put('web/demo.js', '''const apiKey = "must-not-index";
const config = {password: "must-not-index"};
function target() {}
function run() { try {} catch (target) { target(); } }
''')
        r = self.build()
        if r["coverage"]["javascriptParser"] != "Babel":
            self.skipTest("Install explorer web dependencies to exercise Babel")
        self.assertNotIn('must-not-index', json.dumps(r))
        self.assertEqual(self.links(r,self.symbol(r,'run','web/demo.js')), [])

    def test_js_constructor_and_prototype_names_do_not_break_the_batch(self):
        self.put('web/worker.js', '''export class Worker {
    constructor() { this.value = 1; }
    toString() { return "worker"; }
    hasOwnProperty() { return true; }
}
export function __proto__() { return new Worker(); }
export function constructor() { return __proto__(); }
''')
        self.put('web/other.js', 'export const other = () => 2;\n')
        r = self.build()
        if r['coverage']['javascriptParser'] != 'Babel':
            self.skipTest('Install explorer web dependencies to exercise Babel')
        self.assertEqual(r['coverage']['parseErrors'], [])
        names = {s['qualifiedName'] for s in r['symbols'] if s['kind'] != 'variable'}
        self.assertTrue({'Worker.constructor', 'Worker.toString', 'Worker.hasOwnProperty', '__proto__', 'constructor', 'other'} <= names)
        self.assertEqual({f['parseStatus'] for f in r['files']}, {'babel-ast'})


if __name__ == "__main__":
    unittest.main()
