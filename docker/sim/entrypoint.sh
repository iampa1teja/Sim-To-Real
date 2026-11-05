#!/usr/bin/env bash 

set -e 

export ISAAC_PATH="${ISAACSIM_ROOT_PATH:-/isaac-sim}"
export CARB_APP_PATH="$ISAAC_PATH/kit"
export EXP_PATH="$ISAAC_PATH/apps"
source "$ISAAC_PATH/setup_python_env.sh"

export PYTHONPATH="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])'):/workspace/s2r/src:/workspace/s2r${PYTHONPATH:+:$PYTHONPATH}"

export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/xdg-runtime-$(id -u)}
install -d -m 0700 "$XDG_RUNTIME_DIR"

exec "$@"