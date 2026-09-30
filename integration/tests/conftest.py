from pathlib import Path
import sys


ALIGNMENT_ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ALIGNMENT_ROOT.parent / "algorithm/llada"
sys.path[:0] = [str(ALIGNMENT_ROOT),
               str(ALGORITHM_ROOT)]
