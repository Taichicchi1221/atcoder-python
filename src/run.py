import os
import subprocess
from time import perf_counter
os.chdir("/workspaces/atcoder-python/src")

for i in range(100):
    idx = str(i).zfill(4)
    filepath = f"../test/in/{idx}.txt"

    START = perf_counter()
    subprocess.run(f"python ahc.py < {filepath}", shell=True)
    END = perf_counter()
    elapsed = END - START
    print(f"{elapsed:.4f} sec")