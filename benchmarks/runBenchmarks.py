import subprocess
import os
from datetime import datetime

# Auxiliary functions
def readLines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

# Data
benchmarkSet = [
    # "AFG.txt",
    # "Dumas.txt",
    # "GendreauDumasExtended.txt",
    "Langevin.txt"
    # "OhlmannThomas.txt",
    # "Solnon25_feasible.txt",
    # "Solnon25_infeasible.txt",
    # "SolomonPesant.txt",
    # "SolomonPotvinBengio.txt"
]
timeout = 20 * 60 
exe = "../cmake-build-release-ilyin/tsptw_gpu_1_cabs"
resultsDir  = "./results"

# Run
exeName = os.path.basename(exe)
useGpu = True
for benchmark in benchmarkSet:
    logfile = resultsDir + "/" + exeName + "_"  + ("g" if useGpu else "c") + "_" + str(timeout) + "_" + benchmark.removesuffix(".txt") + "_" + datetime.now().strftime("%Y%m%d%H%M") + ".txt"
    instances = readLines(benchmark)
    with open(logfile, "w") as f:
        for instance in instances:
            cmd = [exe,
                   "-t", str(timeout),
                   "-g" if useGpu else "",
                   "-i", instance
                  ]
            cmd = [c for c in cmd if c]
            cmdStr = " ".join(str(c) for c in cmd)
            print(cmdStr)
            f.write(cmdStr + "\n")
            subprocess.run(cmd, stdout=f, stderr=f)
            f.write("\n")
            f.flush()