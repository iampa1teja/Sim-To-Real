"""Add acknowledged per-trial RNG seeding to the pinned official GR00T server."""
import os
from pathlib import Path
import random
import runpy
import sys
import numpy as np
import torch


def seed_policy(seed):
    if type(seed) is not int or not 0 <= seed <= 2**32-1:
        raise ValueError('seed must be a uint32 integer')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {'seed': seed}


def main():
    sys.path.insert(0, '/Isaac-GR00T')
    from gr00t.policy.server_client import PolicyServer
    original = PolicyServer.__init__
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.register_endpoint('seed_policy', seed_policy)
    PolicyServer.__init__ = initialize
    n17 = Path('/opt/so101/launch_gr00t_server_n17.py')
    if n17.is_file():
        sys.path.insert(0, str(n17.parent))
        sys.argv = [str(n17), '--groot', '/Isaac-GR00T', *sys.argv[1:]]
        runpy.run_path(str(n17), run_name='__main__')
    else:
        os.chdir('/Isaac-GR00T')
        runpy.run_path('/Isaac-GR00T/gr00t/eval/run_gr00t_server.py', run_name='__main__')


if __name__ == '__main__':
    main()
