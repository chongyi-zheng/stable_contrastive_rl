import sys
import warnings

idx = 0
while idx < len(sys.path):
    if 'sawyer_control' in sys.path[idx]:
        warnings.warn("Undoing ros generated __init__ for parallel until a better fix is found")
        del sys.path[idx]
    else:
        idx += 1


