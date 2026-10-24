#!/usr/bin/env bash
# Usage: entrypoint.sh --policy <name> [launcher args...]   |   --env <name> -- <command...>
#        entrypoint.sh --list   |   <any other command>
#   --policy  serve the policy in its own venv (policy-cache/venvs/<name>)
#   --env     run a command, usually a training script, in a separate venv (policy-cache/venvs/<name>-train)
#             with the policy's train/requirements.txt added on top; exports SO101_PYTHON and SO101_PROJECT_DIR.
#             The serving venv is never touched, so a training job cannot change a running server.
set -euo pipefail

policies_dir=${POLICIES_DIR:-/policies}
cache=${POLICY_CACHE:-/opt/cache}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

export PYTHONPATH=/opt/so101${PYTHONPATH:+:$PYTHONPATH}

# Anything that is not a serve request (bash, python, uv ...) runs untouched.
case "${1:-}" in
    --list)
        for manifest in "$policies_dir"/*/policy.toml; do
            [[ -f $manifest ]] || continue
            dir=$(dirname "$manifest")
            [[ $(basename "$dir") == _* ]] || basename "$dir"
        done
        exit 0 ;;
    --policy) mode=serve ;;
    --env) mode=env ;;
    *) exec "$@" ;;
esac
[[ $# -ge 2 ]] || fail "$1 needs a name."
name=$2
shift 2
if [[ $mode == env ]]; then
    [[ ${1:-} != -- ]] || shift
    [[ $# -ge 1 ]] || fail '--env needs a command after the policy name (entrypoint.sh --env <name> -- <command...>).'
fi
[[ $name =~ ^[a-z0-9_]+$ ]] || fail "Invalid policy name '$name' (lowercase letters, digits, underscore)."
policy_dir=$policies_dir/$name
[[ -f $policy_dir/policy.toml ]] || fail "No $policy_dir/policy.toml. Mount your policies folder at $policies_dir (docker run -v ./policies:/policies)."

# Read the manifest with the system python; print shell assignments. A command substitution used as a
# plain assignment propagates failure under set -e, so a bad manifest stops the script here.
settings=$(python3 - "$policy_dir/policy.toml" <<'PY'
import re, shlex, sys, tomllib

with open(sys.argv[1], 'rb') as handle:
    manifest = tomllib.load(handle)
if not manifest.get('policy', {}).get('enabled', True):
    sys.exit('Policy is disabled in policy.toml (policy.enabled = false).')
environment = manifest.get('environment', {})
python = str(environment.get('python', ''))
if not re.fullmatch(r'3\.\d+', python):
    sys.exit('environment.python must look like "3.12".')
git, rev = environment.get('git', ''), environment.get('rev', '')
if git and not re.fullmatch(r'[0-9a-f]{40}', rev):
    sys.exit('environment.rev must be a full 40-character commit SHA when environment.git is set.')
print(f'env_python={shlex.quote(python)}')
print(f'env_git={shlex.quote(git)}')
print(f'env_rev={shlex.quote(rev)}')
for key, value in manifest.get('runtime', {}).get('env', {}).items():
    if not re.fullmatch(r'[A-Z_][A-Z0-9_]*', key):
        sys.exit(f'Invalid runtime.env name: {key}')
    print(f'export {key}={shlex.quote(str(value))}')
PY
)
eval "$settings"

# Project that owns the lockfile: either the policy folder itself or a pinned upstream checkout.
if [[ -n $env_git ]]; then
    project=$cache/checkouts/$name-$env_rev
    if [[ ! -d $project/.git ]]; then
        mkdir -p "$cache/checkouts"
        git clone --no-checkout "$env_git" "$project"
    fi
    git -C "$project" checkout --quiet "$env_rev"
    [[ $(git -C "$project" rev-parse HEAD) == "$env_rev" ]] || fail "Checkout of $env_git is not at $env_rev."
else
    project=$policy_dir
    [[ -f $project/pyproject.toml ]] || fail "$project/pyproject.toml missing (or set environment.git in policy.toml)."
fi

# One venv per policy, kept on the cache volume. uv sync is incremental, so later starts are fast.
# Training gets its own venv so extra packages never leak into (or get synced out of) the serving one.
if [[ $mode == env ]]; then
    export UV_PROJECT_ENVIRONMENT=$cache/venvs/$name-train
else
    export UV_PROJECT_ENVIRONMENT=$cache/venvs/$name
fi
if [[ -f $project/uv.lock ]]; then
    uv sync --frozen --project "$project" --python "$env_python"
else
    printf 'WARNING: %s/uv.lock missing; resolving fresh. Commit the generated uv.lock.\n' "$project" >&2
    uv sync --project "$project" --python "$env_python"
fi

if [[ $mode == env ]]; then
    # uv sync removes packages that are not in the lockfile, so the extras are (re)installed after every sync.
    if [[ -f $policy_dir/train/requirements.txt ]]; then
        uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps -r "$policy_dir/train/requirements.txt"
    fi
    export SO101_PYTHON=$UV_PROJECT_ENVIRONMENT/bin/python SO101_PROJECT_DIR=$project
    exec "$@"
fi

# --no-sync: the venv is already correct, and a lock-driven re-sync here could change it.
exec uv run --no-sync --project "$project" python -m server_core.launcher --policy-dir "$policy_dir" "$@"
