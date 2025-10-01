import subprocess
from datetime import datetime

# Auxiliary functions
def readLines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

# Data
benchmarkSet = [
    "AFG.txt",
    "Dumas.txt",
    "GendreauDumasExtended.txt",
    "Langevin.txt",
    "OhlmannThomas.txt",
    "Solnon25_feasible.txt",
    "Solnon25_infeasible.txt",
    "SolomonPesant.txt",
    "SolomonPotvinBengio.txt"
]
timeout = 20 * 60 
exe = "../cmake-build-release-ding/tsptw3gpu"
resultsDir  = "./results"

# Run
useGpu = True
for benchmark in benchmarkSet:
    logfile = resultsDir + "/" + benchmark.removesuffix(".txt") + "_" + ("g" if useGpu else "c") + "_" + str(timeout) + "_" + datetime.now().strftime("%Y%m%d%H%M") + ".txt"
    instances = readLines(benchmark)
    with open(logfile, "w") as f:
        for instance in instances:
            cmd = [exe, 
                   "-w", str(2),
                   "-s", "RONQ",
                   "-t", str(timeout),
                   "-g", str(0 if useGpu else 32 * 1024 * 1024),
                   "-i", instance
                  ]
            cmdStr = " ".join(str(c) for c in cmd);
            print(cmdStr)
            f.write(cmdStr + "\n")
            subprocess.run(cmd, stdout=f, stderr=f)
            f.write("\n")
            f.flush()