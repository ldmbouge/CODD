import subprocess
import os
from datetime import datetime

# Auxiliary functions
def readLines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]
def readFirstInt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return int(f.readline().strip())

def roundUpToMultiple(x,b):
    return ((x + b - 1) // b) * b
# Data
benchmarkSet = [
     #"Dumas.txt",
     #"Langevin.txt",
     "Solnon25_feasible.txt",
     "Solnon25_infeasible.txt",
     "SolomonPesant.txt",
     "SolomonPotvinBengio.txt",
     "AFG.txt",
     "GendreauDumasExtended.txt",
     "OhlmannThomas.txt"
]
timeout = 10 * 60
exe = "/home/fabio/RouteOpt/packages/application/cvrp/bin/benchmarks_tw_single"
resultsDir  = "./results"

# Run
exeName = os.path.basename(exe)
for benchmark in benchmarkSet:
    logfile = (f"{resultsDir}/{exeName}_" +
               f"t{timeout}_" +
               f"{benchmark.removesuffix('.txt')}_" +
               f"{datetime.now().strftime('%Y%m%d%H%M')}" +
               f".txt")
    instances = readLines(benchmark)
    with open(logfile, "w") as f:
        for instance in instances:
            nCities = readFirstInt(instance)
            cmd = ["/usr/bin/time",
                   "-v",
                   exe,
                   instance,
                  ]
            cmd = [c for c in cmd if c]
            cmdStr = " ".join(str(c) for c in cmd)
            print(cmdStr)
            f.write(f"COMMAND: {cmdStr}\n")
            f.flush()
            subprocess.run(cmd, stdout=f, stderr=f)
            f.write("\n")
            f.flush()