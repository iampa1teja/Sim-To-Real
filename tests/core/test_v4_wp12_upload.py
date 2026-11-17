"""WP12: dataset upload through an injected storage client (contracts §10.3, §10.1, §1.5, §1.6).

Two in-memory clients stand in for the real ones: ``FakeHubApi`` (huggingface_hub ``HfApi`` subset)
behind ``HubStorage`` and ``FakeFs`` (fsspec subset) behind ``BucketStorage``. No test touches the
network: sockets are blocked and huggingface_hub / fsspec / boto3 / lerobot / torch are made
unimportable where it matters. Decisions pinned here (docs/v4/prompts/WP12.md):

* "current audit" = ``meta/audit_report.json`` passed and its ``content_digest`` equals the digest of
  the dataset now (``test_v4_wp12_support.content_digest``); checks run before anything is listed or
  uploaded, then the hygiene checks (``GenerationError``, not ``AuditFailed``): any ``*.tmp`` file,
  an ``mp4/`` folder, or a file under ``images/`` (an open LeRobot episode buffer).
* Uploaded files: every regular file under ``data/``, ``videos/``, ``task_meta/``, ``meta/``; groups in
  that order, sorted by remote path inside a group, ``meta/audit_report.json`` last. Remote path =
  posix path relative to the root. Sidecar JSON files are left out when there are ≥ 1000 of them.
* Remote sha256 equal → skipped; ``None`` (unknown) → uploaded again; different → ``GenerationError``
  unless ``allow_overwrite``. Remote files that do not exist locally are left alone.
* ``run_upload`` uploads ``plan.items`` in order, then calls ``finalize(message)`` once.
  ``BucketStorage.finalize`` writes ``<prefix>/_COMPLETE`` (the message); ``list_files`` hides it.
"""

import io
import json
import socket
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from core.datasets.upload import (BucketStorage, HubStorage, StorageClient, UploadItem, UploadPlan, plan_upload,
                                  run_upload)
from core.generate.errors import AuditFailed, GenerationError
from core.scripts import upload_dataset
from test_v4_wp12_support import (AUDIT_REL, FakeFs, FakeHubApi, SimulatedNetworkError, build_shard, sha256_bytes,
                                  write_audit, write_info)
from v4_support import assert_frozen

REPO = "org/fx"
PREFIX = "s3://bkt/datasets/fx"
BARE = "bkt/datasets/fx"
BLOCKED_MODULES = ("huggingface_hub", "fsspec", "s3fs", "boto3", "botocore", "lerobot", "torch", "datasets")


def ready_dataset(root, episodes=3):
    """A consolidated, audited lite dataset with one video file."""
    root = Path(root)
    build_shard(root, 0, 1, episodes=episodes, audit=False, consolidate=True)
    video = root / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42 fake video")
    write_audit(root)
    return root


def expected_order(root, *, sidecars=True):
    """§10.3 upload order: data, videos, task_meta, meta (audit report last), sorted inside a group."""
    root = Path(root)
    out = []
    for top in ("data", "videos", "task_meta", "meta"):
        files = sorted(p.relative_to(root).as_posix() for p in (root / top).rglob("*") if p.is_file())
        if not sidecars:
            files = [f for f in files if not (f.startswith("task_meta/episode_") and f.endswith(".json"))]
        if top == "meta" and AUDIT_REL in files:
            files = [f for f in files if f != AUDIT_REL] + [AUDIT_REL]
        out.extend(files)
    return out


def local_listing(root, remotes):
    return {r: sha256_bytes((Path(root) / r).read_bytes()) for r in remotes}


def hub(api=None, **kwargs):
    api = FakeHubApi(REPO, **kwargs) if api is None else api
    return api, HubStorage(api, REPO)


def bucket(fs=None, **kwargs):
    fs = FakeFs(**kwargs) if fs is None else fs
    return fs, BucketStorage(fs, PREFIX)


def uploaded(client):
    """Dataset files uploaded so far through a fake, relative to the dataset root (no ``_COMPLETE``)."""
    if isinstance(client, FakeHubApi):
        return list(client.uploads)
    return [p[len(BARE) + 1:] for p in client.uploads if p != f"{BARE}/_COMPLETE"]


def no_network():
    """Context: sockets cannot connect and the real client libraries cannot be imported."""
    stack = ExitStack()
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")
    stack.enter_context(mock.patch.object(socket.socket, "connect", refuse))
    stack.enter_context(mock.patch.object(socket, "create_connection", refuse))
    stack.enter_context(mock.patch.object(socket, "getaddrinfo", refuse))
    stack.enter_context(mock.patch.dict(sys.modules, {name: None for name in BLOCKED_MODULES}))
    return stack


def run_script(argv, storage=None):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = upload_dataset.main([str(a) for a in argv], storage=storage)
    return code, out.getvalue(), err.getvalue()


class AuditGateTests(unittest.TestCase):
    """WP12 §10.3 plan_upload refuses without a passing audit for exactly the current dataset state."""

    def _refused(self, root, error=AuditFailed):
        for name, (client, storage) in {"hub": hub(), "bucket": bucket()}.items():
            with self.subTest(client=name):
                with self.assertRaises(error) as cm:
                    plan_upload(root, storage)
                if error is GenerationError:
                    self.assertNotIsInstance(cm.exception, AuditFailed)
                self.assertEqual(uploaded(client), [])
                code, out, err = run_script([root, "--repo_id", REPO], storage=storage)
                self.assertEqual(code, 1, out + err)
                self.assertIn(error.__name__, err)
                self.assertEqual(uploaded(client), [])

    def test_missing_audit(self):
        """§10.3 no meta/audit_report.json → AuditFailed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            (root / AUDIT_REL).unlink()
            self._refused(root)

    def test_failed_audit(self):
        """§10.3 an audit that did not pass → AuditFailed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            write_audit(root, passed=False)
            self._refused(root)

    def test_malformed_audit(self):
        """§10.1 an audit record that is not the AuditReport JSON → AuditFailed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            for text in ("not json", "[]", json.dumps({"passed": True})):
                with self.subTest(record=text):
                    (root / AUDIT_REL).write_text(text)
                    self._refused(root)

    def test_stale_audit(self):
        """§10.3 any change under data/, videos/, task_meta/ or meta/ after the audit → AuditFailed."""
        edits = {
            "data": lambda r: (r / "data" / "chunk-000" / "file-000000.parquet").write_bytes(b"x"),
            "video removed": lambda r: (r / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4").unlink(),
            "task_meta added": lambda r: (r / "task_meta" / "extra.json").write_text("{}"),
            "info": lambda r: write_info(r, total=3, extra={"note": "changed"}),
        }
        for name, edit in edits.items():
            with tempfile.TemporaryDirectory() as tmp, self.subTest(edit=name):
                root = ready_dataset(tmp)
                edit(root)
                self._refused(root)

    def test_audit_digest_ignores_other_paths(self):
        """§10.1 files outside the four trees and the audit report itself do not make the audit stale."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            (root / "README.md").write_text("dataset card")
            record = json.loads((root / AUDIT_REL).read_text())
            (root / AUDIT_REL).write_text(json.dumps(record, indent=4))
            plan = plan_upload(root, bucket()[1])
            self.assertNotIn("README.md", [i.remote for i in plan.items])

    def test_missing_consolidated_parquet(self):
        """§10.3 no task_meta/episodes.parquet → AuditFailed, even with a current passing audit."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            (root / "task_meta" / "episodes.parquet").unlink()
            write_audit(root)
            self._refused(root)


class HygieneTests(unittest.TestCase):
    """WP12 §10.3 leftovers of an unfinished recording → GenerationError (audit current)."""

    def _check(self, make):
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            make(root)
            write_audit(root)
            AuditGateTests._refused(self, root, error=GenerationError)

    def test_tmp_files(self):
        """§10.3 a *.tmp file anywhere under the root (inside or outside the uploaded trees)."""
        for where in ("data/chunk-000/.file-000003.parquet.tmp", "task_meta/.episodes.tmp", "scratch.tmp"):
            with self.subTest(where=where):
                self._check(lambda r, where=where: (r / where).write_text("partial"))

    def test_mp4_preview_tree(self):
        """§10.3 an mp4/ preview tree (LeRobotRecorder --save_mp4)."""
        def make(r):
            (r / "mp4" / "fx" / "front").mkdir(parents=True)
            (r / "mp4" / "fx" / "front" / "front_rgb_000.mp4").write_bytes(b"x")
        self._check(make)

    def test_open_episode_buffer(self):
        """§10.3 frames waiting under images/ (an open episode buffer)."""
        def make(r):
            (r / "images" / "observation.images.front" / "episode-000003").mkdir(parents=True)
            (r / "images" / "observation.images.front" / "episode-000003" / "frame-000000.png").write_bytes(b"x")
        self._check(make)

    def test_empty_images_folder_is_fine(self):
        """§10.3 an empty images/ folder is not an open buffer."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            (root / "images").mkdir()
            self.assertTrue(plan_upload(root, bucket()[1]).items)


class PlanTests(unittest.TestCase):
    """WP12 §10.3 plan_upload items, order, skips and overwrite rules."""

    def test_items_and_order(self):
        """§10.3 every file of the four trees once, in data/videos/task_meta/meta order, audit report last."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            plan = plan_upload(root, hub()[1])
            assert_frozen(plan)
            self.assertIsInstance(plan, UploadPlan)
            self.assertEqual([i.remote for i in plan.items], expected_order(root))
            self.assertEqual(plan.items[-1].remote, AUDIT_REL)
            self.assertEqual(plan.skipped, ())
            for item in plan.items:
                with self.subTest(remote=item.remote):
                    assert_frozen(item)
                    self.assertIsInstance(item, UploadItem)
                    self.assertEqual(Path(item.local), root / item.remote)
                    data = (root / item.remote).read_bytes()
                    self.assertEqual((item.sha256, item.size), (sha256_bytes(data), len(data)))

    def test_sidecar_threshold(self):
        """§10.3 per-episode sidecars are uploaded below 1000 episodes and left out from 1000 on."""
        for count, included in ((999, True), (1000, False)):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(episodes=count):
                root = Path(tmp)
                (root / "task_meta").mkdir()
                for i in range(count):
                    (root / "task_meta" / f"episode_{i:06d}.json").write_text("{}")
                (root / "task_meta" / "episodes.parquet").write_bytes(b"PAR1 placeholder")
                write_info(root, total=count)
                write_audit(root)
                remotes = [i.remote for i in plan_upload(root, bucket()[1]).items]
                sidecars = [r for r in remotes if r.startswith("task_meta/episode_")]
                self.assertEqual(len(sidecars), count if included else 0)
                self.assertIn("task_meta/episodes.parquet", remotes)
                self.assertEqual(remotes, expected_order(root, sidecars=included))

    def test_matching_remote_files_are_skipped(self):
        """§10.3 files whose remote sha256 matches are skipped (in upload order); the rest are planned."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            order = expected_order(root)
            done = order[1::2]
            fs, storage = bucket()
            for remote in done:
                fs.objects[f"{BARE}/{remote}"] = (root / remote).read_bytes()
            plan = plan_upload(root, storage)
            self.assertEqual(list(plan.skipped), done)
            self.assertEqual([i.remote for i in plan.items], order[0::2])

    def test_different_remote_file(self):
        """§10.3 a remote file with another sha256 → GenerationError; allow_overwrite plans it again."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            target = "meta/info.json"
            for name, (client, storage) in {"hub": hub(), "bucket": bucket()}.items():
                with self.subTest(client=name):
                    if isinstance(client, FakeHubApi):
                        client.files[target] = b"other"
                    else:
                        client.objects[f"{BARE}/{target}"] = b"other"
                    with self.assertRaises(GenerationError) as cm:
                        plan_upload(root, storage)
                    self.assertNotIsInstance(cm.exception, AuditFailed)
                    plan = plan_upload(root, storage, allow_overwrite=True)
                    self.assertEqual([i.remote for i in plan.items], expected_order(root))
                    run_upload(plan, storage, message="overwrite")
                    self.assertEqual(storage.list_files()[target], sha256_bytes((root / target).read_bytes()))

    def test_unknown_remote_hash_is_uploaded_again(self):
        """§10.3 a remote file without a known sha256 (None) is planned again without allow_overwrite."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            order = expected_order(root)
            fs, storage = bucket(with_sha=False)
            for remote in order:
                fs.objects[f"{BARE}/{remote}"] = (root / remote).read_bytes()
            self.assertEqual(set(storage.list_files().values()), {None})
            self.assertEqual([i.remote for i in plan_upload(root, storage).items], order)
            api, storage = hub(lfs=lambda path: not path.endswith(".json"))
            run_upload(plan_upload(root, storage), storage, message="first")
            listing = storage.list_files()
            self.assertTrue(all((v is None) == r.endswith(".json") for r, v in listing.items() if r != ".gitattributes"))
            again = [i.remote for i in plan_upload(root, storage).items]
            self.assertEqual(again, [r for r in order if r.endswith(".json")])

    def test_extra_remote_files_left_alone(self):
        """§10.3 remote files that do not exist locally are neither planned nor deleted."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            api, storage = hub()
            api.files["README.md"] = b"card"
            run_upload(plan_upload(root, storage), storage, message="m")
            self.assertEqual(api.files["README.md"], b"card")
            self.assertNotIn("README.md", api.uploads)


class RunTests(unittest.TestCase):
    """WP12 §10.3 run_upload: exactly once, resumable, idempotent, same result on both backends."""

    def test_each_file_once_and_same_listing(self):
        """§10.3 both clients receive every file exactly once, in plan order, and list the same files."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            order = expected_order(root)
            listings = {}
            for name, (client, storage) in {"hub": hub(), "bucket": bucket()}.items():
                with self.subTest(client=name):
                    run_upload(plan_upload(root, storage), storage, message="v4 dataset")
                    self.assertEqual(uploaded(client), order)
                    listing = storage.list_files()
                    listing.pop(".gitattributes", None)
                    listings[name] = listing
            self.assertEqual(listings["hub"], listings["bucket"])
            self.assertEqual(listings["hub"], local_listing(root, order))

    def test_finalize(self):
        """§10.3 finalize once after the last upload: bucket writes _COMPLETE (hidden from list_files); Hub no-op."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            fs, storage = bucket()
            calls = []
            original = storage.finalize
            with mock.patch.object(storage, "finalize", side_effect=lambda m: (calls.append((m, len(fs.uploads))),
                                                                                 original(m))):
                plan = plan_upload(root, storage)
                run_upload(plan, storage, message="release 1")
            self.assertEqual(calls, [("release 1", len(plan.items))])
            self.assertEqual(fs.objects[f"{BARE}/_COMPLETE"].decode(), "release 1")
            self.assertNotIn("_COMPLETE", storage.list_files())
            api, storage = hub()
            run_upload(plan_upload(root, storage), storage, message="release 1")
            before = list(api.calls)
            storage.finalize("again")
            self.assertEqual(api.calls, before)

    def test_idempotent(self):
        """§10.3 a second run after a complete upload uploads nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            for name, (client, storage) in {"hub": hub(), "bucket": bucket()}.items():
                with self.subTest(client=name):
                    run_upload(plan_upload(root, storage), storage, message="one")
                    count = len(uploaded(client))
                    plan = plan_upload(root, storage)
                    self.assertEqual(plan.items, ())
                    self.assertEqual(list(plan.skipped), expected_order(root))
                    run_upload(plan, storage, message="two")
                    self.assertEqual(len(uploaded(client)), count)

    def test_resume_after_failure(self):
        """§10.3 a failure midway: rerunning uploads only the missing files; every file lands exactly once."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            order = expected_order(root)
            for fail_at in (1, 2, len(order) // 2, len(order)):
                for name in ("hub", "bucket"):
                    with self.subTest(client=name, fail_at=fail_at):
                        client, storage = hub(fail_at=fail_at) if name == "hub" else bucket(fail_at=fail_at)
                        with self.assertRaises(SimulatedNetworkError):
                            run_upload(plan_upload(root, storage), storage, message="m")
                        self.assertEqual(uploaded(client), order[:fail_at - 1])
                        if name == "bucket":
                            self.assertNotIn(f"{BARE}/_COMPLETE", client.objects)
                        plan = plan_upload(root, storage)
                        self.assertEqual(list(plan.skipped), order[:fail_at - 1])
                        self.assertEqual([i.remote for i in plan.items], order[fail_at - 1:])
                        run_upload(plan, storage, message="m")
                        self.assertEqual(sorted(uploaded(client)), sorted(order))
                        self.assertEqual(len(uploaded(client)), len(set(uploaded(client))))

    def test_hub_storage_arguments(self):
        """§10.3 HubStorage passes repo_id, repo_type and revision with huggingface_hub keyword names."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            api = FakeHubApi("team/other", revision="v1.0")
            storage = HubStorage(api, "team/other", revision="v1.0")
            self.assertIsInstance(storage, StorageClient)
            run_upload(plan_upload(root, storage), storage, message="m")
            self.assertEqual(api.uploads, expected_order(root))

    def test_bucket_prefix_handling(self):
        """§10.3 BucketStorage: trailing slash and protocol in the prefix; neighbours of the prefix ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            fs = FakeFs()
            fs.objects.update({"bkt/datasets/fx2/meta/info.json": b"n", "bkt/other/x": b"o"})
            storage = BucketStorage(fs, PREFIX + "/")
            self.assertIsInstance(storage, StorageClient)
            self.assertEqual(storage.list_files(), {})
            run_upload(plan_upload(root, storage), storage, message="m")
            self.assertEqual(sorted(storage.list_files()), sorted(expected_order(root)))
            self.assertEqual(fs.objects["bkt/datasets/fx2/meta/info.json"], b"n")
            self.assertTrue(all(k.startswith(BARE + "/") for k in fs.uploads))

    def test_no_network(self):
        """§1.6 the whole plan/run cycle works with sockets blocked and the real client libraries absent."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            with no_network():
                for name, (client, storage) in {"hub": hub(), "bucket": bucket()}.items():
                    with self.subTest(client=name):
                        run_upload(plan_upload(root, storage), storage, message="m")
                        self.assertEqual(uploaded(client), expected_order(root))


def fake_hf_module(api):
    module = types.ModuleType("huggingface_hub")
    module.HfApi = lambda *args, **kwargs: api
    return module


def fake_fsspec_modules(fs):
    module = types.ModuleType("fsspec")
    core = types.ModuleType("fsspec.core")
    url_to_fs = lambda url, **kwargs: (fs, FakeFs.strip(url))  # noqa: E731
    module.url_to_fs = core.url_to_fs = url_to_fs
    module.filesystem = lambda protocol, **kwargs: fs
    module.core = core
    return {"fsspec": module, "fsspec.core": core}


class ScriptTests(unittest.TestCase):
    """WP12 §10.3 upload_dataset argument handling."""

    def test_injected_storage(self):
        """§10.3 main(argv, storage=...) uploads everything, finalizes with --message, exits 0."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            fs, storage = bucket()
            code, out, err = run_script([root, "--bucket", PREFIX, "--message", "release 7"], storage=storage)
            self.assertEqual(code, 0, err)
            self.assertEqual(uploaded(fs), expected_order(root))
            self.assertEqual(fs.objects[f"{BARE}/_COMPLETE"].decode(), "release 7")

    def test_dry_run(self):
        """§10.3 --dry_run plans (and prints) but uploads and finalizes nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            fs, storage = bucket()
            code, out, err = run_script([root, "--bucket", PREFIX, "--dry_run"], storage=storage)
            self.assertEqual(code, 0, err)
            self.assertEqual(fs.uploads, [])
            self.assertTrue(out.strip())

    def test_allow_overwrite_flag(self):
        """§10.3 a differing remote file fails (exit 1, GenerationError) unless --allow_overwrite."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            api, storage = hub()
            api.files["meta/info.json"] = b"other"
            code, _, err = run_script([root, "--repo_id", REPO], storage=storage)
            self.assertEqual(code, 1)
            self.assertIn("GenerationError", err)
            self.assertEqual(api.uploads, [])
            code, _, err = run_script([root, "--repo_id", REPO, "--allow_overwrite"], storage=storage)
            self.assertEqual(code, 0, err)
            self.assertEqual(api.files["meta/info.json"], (root / "meta" / "info.json").read_bytes())

    def test_target_arguments(self):
        """§10.3 exactly one of --repo_id / --bucket, and the root, are required (argparse exit 2)."""
        for argv in (["root"], ["root", "--repo_id", REPO, "--bucket", PREFIX], ["--repo_id", REPO]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                upload_dataset.main(argv, storage=bucket()[1])
            self.assertEqual(cm.exception.code, 2)

    def test_real_hub_client_is_lazy(self):
        """§1.6 without storage=, --repo_id builds HubStorage over huggingface_hub.HfApi() imported at call time."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            api = FakeHubApi(REPO, revision="v2")
            with no_network(), mock.patch.dict(sys.modules, {"huggingface_hub": fake_hf_module(api)}):
                code, _, err = run_script([root, "--repo_id", REPO, "--revision", "v2"])
            self.assertEqual(code, 0, err)
            self.assertEqual(api.uploads, expected_order(root))
            with no_network(), self.assertRaises(ImportError):
                run_script([root, "--repo_id", REPO])

    def test_real_bucket_client_is_lazy(self):
        """§1.6 without storage=, --bucket builds BucketStorage over an fsspec filesystem imported at call time."""
        with tempfile.TemporaryDirectory() as tmp:
            root = ready_dataset(tmp)
            fs = FakeFs()
            with no_network(), mock.patch.dict(sys.modules, fake_fsspec_modules(fs)):
                code, _, err = run_script([root, "--bucket", PREFIX])
            self.assertEqual(code, 0, err)
            self.assertEqual(uploaded(fs), expected_order(root))
            self.assertIn(f"{BARE}/_COMPLETE", fs.objects)


if __name__ == "__main__":
    unittest.main()
