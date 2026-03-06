import subprocess
import os
import sys
import argparse
from datetime import datetime

# ── Benchmarks ────────────────────────────────────────────────────────────────

knapsackBenchmarks = [
    "../data/knapsack/breaksRF2.txt",
    "../data/knapsack/breaksRF.txt",
    "../data/knapsack/easy2.txt",
    "../data/knapsack/easy3.txt",
    "../data/knapsack/easy4.txt",
    "../data/knapsack/easy.txt",
    "../data/knapsack/f10_l-d_kp_20_879.txt",
    "../data/knapsack/f1_l-d_kp_10_269.txt",
    "../data/knapsack/f2_l-d_kp_20_878.txt",
    "../data/knapsack/f3_l-d_kp_4_20.txt",
    "../data/knapsack/f4_l-d_kp_4_11.txt",
    "../data/knapsack/f6_l-d_kp_10_60.txt",
    "../data/knapsack/f7_l-d_kp_7_50.txt",
    "../data/knapsack/f8_l-d_kp_23_10000.txt",
    "../data/knapsack/f9_l-d_kp_5_80.txt",
    "../data/knapsack/knapPI_1_10000_1000_1.txt",
    "../data/knapsack/knapPI_1_1000_1000_1.txt",
    "../data/knapsack/knapPI_1_100_1000_1.txt",
    "../data/knapsack/knapPI_1_2000_1000_1.txt",
    "../data/knapsack/knapPI_1_200_1000_1.txt",
    "../data/knapsack/knapPI_1_5000_1000_1.txt",
    "../data/knapsack/knapPI_1_500_1000_1.txt",
    "../data/knapsack/knapPI_2_10000_1000_1.txt",
    "../data/knapsack/knapPI_2_1000_1000_1.txt",
    "../data/knapsack/knapPI_2_100_1000_1.txt",
    "../data/knapsack/knapPI_2_2000_1000_1.txt",
    "../data/knapsack/knapPI_2_200_1000_1.txt",
    "../data/knapsack/knapPI_2_5000_1000_1.txt",
    "../data/knapsack/knapPI_2_500_1000_1.txt",
    "../data/knapsack/knapPI_3_10000_1000_1.txt",
    "../data/knapsack/knapPI_3_1000_1000_1.txt",
    "../data/knapsack/knapPI_3_100_1000_1.txt",
    "../data/knapsack/knapPI_3_2000_1000_1.txt",
    "../data/knapsack/knapPI_3_200_1000_1.txt",
    "../data/knapsack/knapPI_3_5000_1000_1.txt",
    "../data/knapsack/knapPI_3_500_1000_1.txt",
    None  # guard
]

mispBenchmarks = [
    "../data/misp/brock200_1.clq",
    "../data/misp/brock200_2.clq",
    "../data/misp/brock200_3.clq",
    "../data/misp/brock200_4.clq",
    "../data/misp/brock400_1.clq",
    "../data/misp/brock400_2.clq",
    "../data/misp/brock400_3.clq",
    "../data/misp/brock400_4.clq",
    "../data/misp/brock800_1.clq",
    "../data/misp/brock800_2.clq",
    "../data/misp/brock800_3.clq",
    "../data/misp/brock800_4.clq",
    "../data/misp/c-fat200-1.clq",
    "../data/misp/c-fat200-2.clq",
    "../data/misp/c-fat200-5.clq",
    "../data/misp/c-fat500-10.clq",
    "../data/misp/c-fat500-1.clq",
    "../data/misp/c-fat500-2.clq",
    "../data/misp/c-fat500-5.clq",
    "../data/misp/hamming10-2.clq",
    "../data/misp/hamming10-4.clq",
    "../data/misp/hamming6-2.clq",
    "../data/misp/hamming6-4.clq",
    "../data/misp/hamming8-2.clq",
    "../data/misp/hamming8-4.clq",
    "../data/misp/johnson16-2-4.clq",
    "../data/misp/johnson32-2-4.clq",
    "../data/misp/johnson8-2-4.clq",
    "../data/misp/johnson8-4-4.clq",
    "../data/misp/keller4.clq",
    "../data/misp/keller5.clq",
    "../data/misp/MANN_a27.clq",
    "../data/misp/MANN_a45.clq",
    "../data/misp/MANN_a9.clq",
    "../data/misp/p_hat1000-1.clq",
    "../data/misp/p_hat1000-2.clq",
    "../data/misp/p_hat1000-3.clq",
    "../data/misp/p_hat1500-1.clq",
    "../data/misp/p_hat1500-2.clq",
    "../data/misp/p_hat1500-3.clq",
    "../data/misp/p_hat300-1.clq",
    "../data/misp/p_hat300-2.clq",
    "../data/misp/p_hat300-3.clq",
    "../data/misp/p_hat500-1.clq",
    "../data/misp/p_hat500-2.clq",
    "../data/misp/p_hat500-3.clq",
    "../data/misp/p_hat700-1.clq",
    "../data/misp/p_hat700-2.clq",
    "../data/misp/p_hat700-3.clq",
    "../data/misp/san1000.clq",
    "../data/misp/san200_0.7_1.clq",
    "../data/misp/san200_0.7_2.clq",
    "../data/misp/san200_0.9_1.clq",
    "../data/misp/san200_0.9_2.clq",
    "../data/misp/san200_0.9_3.clq",
    "../data/misp/san400_0.5_1.clq",
    "../data/misp/san400_0.7_1.clq",
    "../data/misp/san400_0.7_2.clq",
    "../data/misp/san400_0.7_3.clq",
    "../data/misp/san400_0.9_1.clq",
    "../data/misp/sanr200_0.7.clq",
    "../data/misp/sanr200_0.9.clq",
    "../data/misp/sanr400_0.5.clq",
    "../data/misp/sanr400_0.7.clq",
    None  # guard
]

golomBenchmarks = [
    # "../data/gruler/0.txt",
    # "../data/gruler/1.txt",
    # "../data/gruler/2.txt",
    # "../data/gruler/3.txt",
    # "../data/gruler/4.txt",
    # "../data/gruler/5.txt",
    # "../data/gruler/6.txt",
    # "../data/gruler/7.txt",
    # "../data/gruler/8.txt",
    # "../data/gruler/9.txt",
    "../data/gruler/10.txt",
    "../data/gruler/11.txt",
    "../data/gruler/12.txt",
    "../data/gruler/13.txt",
    "../data/gruler/14.txt",
    "../data/gruler/15.txt",
    # "../data/gruler/16.txt",
    # "../data/gruler/17.txt",
    # "../data/gruler/18.txt",
    # "../data/gruler/19.txt",
    # "../data/gruler/20.txt",
    None  # guard
]

benchmarks = {
    "knapsack" : knapsackBenchmarks,
    "misp"     : mispBenchmarks,
    "golom"    : golomBenchmarks,
}

# ── Instance resolution ───────────────────────────────────────────────────────

def resolveInstance(benchmark):
    """Convert a benchmark string into a list of command-line arguments.
    If it is a file path, return its absolute path as a single-element list.
    Otherwise split by whitespace, dropping empty tokens."""
    if os.path.exists(benchmark):
        return [os.path.abspath(benchmark)]
    return [t for t in benchmark.split() if t]

# ── Arguments ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()

# Runner arguments
parser.add_argument("-b", "--benchmark", required=True,
                    help=f"Benchmark name: {list(benchmarks.keys())}")
parser.add_argument("-x", "--exe",       required=True,
                    help="Path to the solver executable")

# Solver arguments (forwarded verbatim to the executable)
parser.add_argument("-g", "--gpu-width",   type=int,   required=True,
                    help="DD relaxed width (GPU)")
parser.add_argument("-c", "--cpu-width",   type=int,   default=4096,
                    help="DD restricted width (CPU) [default: 4096]")
parser.add_argument("-v", "--validators",  type=int,   default=16,
                    help="Number of restricted DD (CPU) [default: 16]")
parser.add_argument("-m", "--memory",      type=float, default=42,
                    help="Working memory in GB [default: 42]")
parser.add_argument("-p", "--pop",         type=int,   default=100,
                    help="Max nodes to pop in parallel [default: 100]")
parser.add_argument("-l", "--lambda",      type=float, default=1.0,
                    dest="lambda_",
                    help="Favour exact nodes [default: 1.0]")
parser.add_argument("-t", "--timeout",     type=int,   required=True,
                    help="Timeout in seconds")

args = parser.parse_args()

if args.benchmark not in benchmarks:
    print(f"Unknown benchmark '{args.benchmark}'. Available: {list(benchmarks.keys())}")
    sys.exit(1)

exe     = os.path.abspath(args.exe)
timeout = args.timeout
problem = args.benchmark

# Build the solver flags that will be forwarded on every invocation.
# Only include flags that were explicitly provided.
solverFlags = []
if args.gpu_width  is not None: solverFlags += ["--gpu-width",  str(args.gpu_width)]
if args.cpu_width  is not None: solverFlags += ["--cpu-width",  str(args.cpu_width)]
if args.validators is not None: solverFlags += ["--validators", str(args.validators)]
if args.memory     is not None: solverFlags += ["--memory",     str(args.memory)]
if args.pop        is not None: solverFlags += ["--pop",        str(args.pop)]
if args.lambda_    is not None: solverFlags += ["--lambda",     str(args.lambda_)]
solverFlags += ["--timeout", str(timeout)]

resultsDir = "./results"
os.makedirs(resultsDir, exist_ok=True)

# ── Run ───────────────────────────────────────────────────────────────────────

exeName = os.path.basename(exe)

# Build a compact tag from the non-None solver options for use in log filenames
def flagTag(args):
    parts = []
    if args.gpu_width  is not None: parts.append(f"g{args.gpu_width}")
    if args.cpu_width  is not None: parts.append(f"c{args.cpu_width}")
    if args.validators is not None: parts.append(f"v{args.validators}")
    if args.memory     is not None: parts.append(f"m{args.memory}")
    if args.pop        is not None: parts.append(f"p{args.pop}")
    if args.lambda_    is not None: parts.append(f"l{args.lambda_}")
    return "_".join(parts) if parts else "default"

tag = flagTag(args)

for benchmark in benchmarks[problem]:
    if benchmark is None:
        continue

    instance = resolveInstance(benchmark)
    instanceName = "_".join(
        os.path.splitext(os.path.basename(t))[0] if os.path.exists(t) else t
        for t in instance
    )

    logfile = (f"{resultsDir}/"
               f"{problem}_"
               f"{instanceName}_"
               f"{tag}_"
               f"{datetime.now().strftime('%Y%m%d%H%M')}"
               f".txt")

    with open(logfile, "w") as f:
        cmd = ["/usr/bin/time", "-v",
               "/usr/bin/timeout", str(timeout + 15),
               exe, *solverFlags,
               "-i", *instance]
        cmdStr = " ".join(str(c) for c in cmd)
        print(cmdStr)
        f.write(f"COMMAND: {cmdStr}\n")
        f.flush()
        subprocess.run(cmd, stdout=f, stderr=f)
        f.write("\n")
        f.flush()