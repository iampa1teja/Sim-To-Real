#!/bin/bash
set -e

ISAAC_SIM=/workspace/isaaclab/_isaac_sim

export CARB_APP_PATH=$ISAAC_SIM/kit
export ISAAC_PATH=$ISAAC_SIM
export EXP_PATH=$ISAAC_SIM/apps
source ${ISAAC_SIM}/setup_python_env.sh
source /root/env 2>/dev/null

export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/xdg-runtime-$(id -u)}
install -d -m 0700 "$XDG_RUNTIME_DIR"

cat > /usr/local/bin/python << 'WRAPPER'
#!/bin/bash
exec /workspace/isaaclab/_isaac_sim/python.sh "$@"
WRAPPER
chmod +x /usr/local/bin/python

python -m pip install --quiet --disable-pip-version-check --root-user-action=ignore -e /workspace/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/

exec "$@"
