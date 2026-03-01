import subprocess
import os
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
    # ...
    None  # guard
]

golomBenchmarks = [
    "5 25",
    "6 36",
    "7 49",
    "8 64",
    "9 81",
    "10 100",
    "11 121",
    "12 144",
    "13 169",
    "14 196",
    "15 225",
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
parser.add_argument("-b", "--benchmark", required=True,  help=f"Benchmark name: {list(benchmarks.keys())}")
parser.add_argument("-x", "--exe",       required=True,  help="Path to the solver executable")
parser.add_argument("-w", "--width",     required=True,  type=int, nargs='+', help="Width parameter(s)")
parser.add_argument("-t", "--timeout",   required=True,  type=int, help="Timeout in seconds")
args = parser.parse_args()

if args.benchmark not in benchmarks:
    print(f"Unknown benchmark '{args.benchmark}'. Available: {list(benchmarks.keys())}")
    sys.exit(1)

exe     = os.path.abspath(args.exe)
timeout = args.timeout
width   = args.width
problem = args.benchmark

resultsDir = "./results"
os.makedirs(resultsDir, exist_ok=True)

# ── Run ───────────────────────────────────────────────────────────────────────

exeName = os.path.basename(exe)
for width in args.width:
    for benchmark in benchmarks[problem]:
        if benchmark is None:
            continue

        instance = resolveInstance(benchmark)
        instanceName = "_".join(os.path.splitext(os.path.basename(t))[0] if os.path.exists(t) else t for t in instance)

        logfile = (f"{resultsDir}/"
                   f"{problem}_"
                   f"{instanceName}_"
                   f"w{width}_"
                   f"{datetime.now().strftime('%Y%m%d%H%M')}"
                   f".txt")

        with open(logfile, "w") as f:
            cmd = ["/usr/bin/time", "-v",
                   "/usr/bin/timeout", str(timeout + 15),
                   exe,
                   *instance,
                   str(width)]
            cmdStr = " ".join(str(c) for c in cmd)
            print(cmdStr)
            f.write(f"COMMAND: {cmdStr}\n")
            f.flush()
            subprocess.run(cmd, stdout=f, stderr=f)
            f.write("\n")
            f.flush()
