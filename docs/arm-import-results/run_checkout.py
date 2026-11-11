"""Select a checkout after s2r-entrypoint has configured the SDK environment."""
import runpy
import sys
from pathlib import Path
checkout=Path(sys.argv.pop(1)).resolve()
module=sys.argv.pop(1)
sys.path.insert(0,str(checkout/'src'))
sys.argv[0]=module
print(f'[checkout] {checkout}: {module}',flush=True)
runpy.run_module(module,run_name='__main__')
