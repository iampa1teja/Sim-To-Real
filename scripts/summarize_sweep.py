#!/usr/bin/env python3
"""Repository entry point for summarize_eval_sweep."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source'))
from sim_to_real_so101.scripts.summarize_sweep import main
if __name__ == '__main__': main()
