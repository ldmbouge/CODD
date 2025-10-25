import subprocess
import os
from datetime import datetime

# Auxiliary functions
def readLines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

# Data
benchmarkSet = [
     #"AFG.txt",
     #"Dumas.txt",
     #"GendreauDumasExtended.txt",
     #"Langevin.txt",
     #"OhlmannThomas.txt",
     "Solnon25_feasible.txt",
     "Solnon25_infeasible.txt",
     #"SolomonPesant.txt",
     #"SolomonPotvinBengio.txt"
]
timeout = 30 * 60
exe = "../cmake-build-release-ding/tsptw_gpu_1_cbs"
resultsDir  = "./results"

# Run
exeName = os.path.basename(exe)
useGpu = True
for benchmark in benchmarkSet:
    logfile = (f"{resultsDir}/{exeName}_" +
               f"{'g' if useGpu else 'c'}_" +
               f"{timeout}_" +
               f"{benchmark.removesuffix('.txt')}_" +
               f"{datetime.now().strftime('%Y%m%d%H%M')}" +
               f".txt")
    instances = readLines(benchmark)
    with open(logfile, "w") as f:
        for instance in instances:
            cmd = ["/usr/bin/time",
                   "-v",
                   exe,
                   "-t", str(timeout),
                   "-g" if useGpu else "",
                   "-i", instance
                  ]
            cmd = [c for c in cmd if c]
            cmdStr = " ".join(str(c) for c in cmd)
            print(cmdStr)
            f.write(f"COMMAND: {cmdStr}\n")
            f.flush()
            subprocess.run(cmd, stdout=f, stderr=f)
            f.write("\n")
            f.flush()