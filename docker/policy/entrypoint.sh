
#!/usr/bin/env bash
# Usage:
#   entrypoint.sh --policy <name> [launcher args...]
#   entrypoint.sh --env <name> -- <command...>
#   entrypoint.sh --list
#
# --policy: serve a policy in its own venv (/opt/cache/venvs/<name>)
# --env:    run a command in a separate venv (/opt/cache/venvs/<name>-train)

set -euo pipefail

policies_dir=${POLICIES_DIR:-/policies}
cache=${POLICY_CACHE:-/opt/cache}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

# Shared modules copied into /opt/src by the Dockerfile.
export PYTHONPATH="/opt/src${PYTHONPATH:+:$PYTHONPATH}"

case "${1:-}" in
    --list)
        for manifest in "$policies_dir"/*/policy.toml; do
            [[ -f $manifest ]] || continue
            dir=$(dirname "$manifest")
            [[ $(basename "$dir") == _* ]] || basename "$dir"
        done
        exit 0
        ;;
    --policy)
        mode=serve
        ;;
    --env)
        mode=env
        ;;
    *)
        exec "$@"
        ;;
esac

[[ $# -ge 2 ]] || fail "${1:-Command} needs a name."

name=$2
shift 2

if [[ $mode == env ]]; then
    [[ ${1:-} != -- ]] || shift
    [[ $# -ge 1 ]] || fail '--env needs a command after the policy name.'
fi

[[ $name =~ ^[a-z0-9_]+$ ]] ||
    fail "Invalid policy name '$name' (lowercase letters, digits, underscore)."

policy_dir=$policies_dir/$name

[[ -f $policy_dir/policy.toml ]] ||
    fail "No $policy_dir/policy.toml. Mount your policies folder at $policies_dir."

# Read policy configuration using system Python.
settings=$(python3 - "$policy_dir/policy.toml" <<'PY'
import re
import shlex
import sys
import tomllib

with open(sys.argv[1], "rb") as handle:
    manifest = tomllib.load(handle)

if not manifest.get("policy", {}).get("enabled", True):
    sys.exit("Policy is disabled in policy.toml (policy.enabled = false).")

environment = manifest.get("environment", {})
python = str(environment.get("python", ""))

if not re.fullmatch(r"3\.\d+", python):
    sys.exit('environment.python must look like "3.12".')

git = environment.get("git", "")
rev = environment.get("rev", "")

if git and not re.fullmatch(r"[0-9a-f]{40}", rev):
    sys.exit("environment.rev must be a full 40-character commit SHA when environment.git is set.")

print(f"env_python={shlex.quote(python)}")
print(f"env_git={shlex.quote(git)}")
print(f"env_rev={shlex.quote(rev)}")

for key, value in manifest.get("runtime", {}).get("env", {}).items():
    if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
        sys.exit(f"Invalid runtime.env name: {key}")

    print(f"export {key}={shlex.quote(str(value))}")
PY
)

eval "$settings"

# Project that owns the lockfile:
# either the policy directory or a pinned upstream checkout.
if [[ -n $env_git ]]; then
    project=$cache/checkouts/$name-$env_rev

    if [[ ! -d $project/.git ]]; then
        mkdir -p "$cache/checkouts"
        git clone --no-checkout "$env_git" "$project"
    fi

    git -C "$project" checkout --quiet "$env_rev"

    [[ $(git -C "$project" rev-parse HEAD) == "$env_rev" ]] ||
        fail "Checkout of $env_git is not at $env_rev."
else
    project=$policy_dir

    [[ -f $project/pyproject.toml ]] ||
        fail "$project/pyproject.toml missing (or set environment.git in policy.toml)."
fi

# Separate serving and training environments.
sync_flags=()

if [[ $mode == env ]]; then
    export UV_PROJECT_ENVIRONMENT=$cache/venvs/$name-train
    sync_flags=(--inexact)
else
    export UV_PROJECT_ENVIRONMENT=$cache/venvs/$name
fi

if [[ -f $project/uv.lock ]]; then
    uv sync --frozen "${sync_flags[@]}" --project "$project" --python "$env_python"
else
    printf 'WARNING: %s/uv.lock missing; resolving fresh. Commit the generated uv.lock.\n' "$project" >&2
    uv sync "${sync_flags[@]}" --project "$project" --python "$env_python"
fi

if [[ $mode == env ]]; then
    # Install training-only requirements.
    if [[ -f $policy_dir/train/requirements.txt ]]; then
        uv pip install \
            --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
            --no-deps \
            -r "$policy_dir/train/requirements.txt"
    fi

    export S2R_PYTHON=$UV_PROJECT_ENVIRONMENT/bin/python
    export S2R_PROJECT_DIR=$project

    exec "$@"
fi

# Run the policy server launcher.
exec uv run --no-sync --project "$project" \
    python -m policy_server.launcher \
    --policy-dir "$policy_dir" "$@"
