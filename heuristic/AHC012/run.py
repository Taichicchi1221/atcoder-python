import os
import subprocess
from time import perf_counter

for i in range(100):
    idx = str(i).zfill(4)
    filepath = f"/workspaces/atcoder-python/heuristic/AHC012/in/{idx}.txt"

    START = perf_counter()
    subprocess.run(
        f"python /workspaces/atcoder-python/heuristic/AHC012/ahc.py < {filepath} 1> /dev/null",
        shell=True
    )
    END = perf_counter()
    elapsed = END - START
    print(f"### input: {idx},  {elapsed:.4f} sec")
    print()
