import sys
from pathlib import Path

# Add the root directory to sys.path
upper_levels = 2
root_dir = Path(__file__).resolve().parents[upper_levels]
sys.path.append(str(root_dir))