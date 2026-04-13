# CODD
CODD is a C++/CUDA solver for combinatorial optimization problems represented as state-based models.
It implements *Complete Anytime Decision Diagram Search (CADDS)* with *GPU acceleration*, achieving speedups up to two orders of magnitude over sequential methods.


## Quickinfo
**Model** The user needs to specify just four components:

- *State:* the information characterizing the problem.
- *Labeling function:* returns the possible decisions from a given state.
- *Transition function:* returns the successor state resulting from a decision.
- *Cost function:* returns the cost of a decision.

Two optional components can be included to accelerate the solving:

- *Dominance rule:* a condition to discard a state dominated by another.
- *Heuristic function:* returns an approximate cost of a state.

**CADDS** It views the search space as a Multi-valued Decision Diagram (MDD) and
incrementally explores it to collect solutions until optimality is proven.

**GPU acceleration** It leverages the layered structure of MDDs. Given a set of
states from the same layer, the GPU parallelizes both the successor generation
and the filtering of duplicate, dominated, and suboptimal states.

## Structure

```
├── gfl/            # GPU-Friendly Library: types, bitset, allocators, views, ...
├── src/            # Solver core: expansion engines, nodes, queue, solution tracking, ...
├── examples/       # Entry points: one file per problem/backend combination
└── data/           # Benchmark instances: AFG, Dumas, Solnon, Solomon, ...
```

## Requirements

- CMake >= 3.28
- CUDA Toolkit >= 12.5
- GCC >= 13.2

## Build

```
mkdir build
cd build
cmake .. -DENABLE_GPU=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc 
make
```

## Run

```
./TsptwCaddsGpu -m 8 -i ../data/tsptw/SolnonFeasible/n31g60b40.001.txt
Search: CADD
Engine: GPU
Instance: ../data/tsptw/Solnon25_feasible/n31g60b40.001.txt
Fragment: Auto
Memory: 8 GB
Timeout: None
---
Time [s]         Primal       Expanded          Queue        Nodes/s
    5.00              -       12919280        1466281        2583553
   10.00              -       30761085        4445764        3567975
   11.49         607.00       37828021        4445764        4738322
   12.37         603.00       42192609        3358122        4974031
   13.48         602.00       47787834        2286592        5035407
   19.00         602.00       73660689          87411        4686440
---
Status           = Completed
Extracted        = 80948258
Queue            = 0
Search Time      = 20.60
Solution Time    = 13.48
Solution Cost    = 602.00
Solution         = 17,25,23,27,7,28,6,24,19,22,4,13,14,20,16,11,10,9,2,18,8,12,3,21,15,5,1,26,30,29,0
```

| Flag              | Description                          |
|-------------------|--------------------------------------|
| `-i, --instance`  | Path to the instance file (required) |
| `-m, --memory`    | Working memory in GB (required)      |
| `-f, --fragment`  | Maximum states per expansion step    |        |
| `-t, --timeout`   | Timeout in seconds                   |

## Citation

```bibtex
@inproceedings{tardivo2025cadds,
    title     = {Complete Anytime Decision Diagram Search with GPU-Accelerated State Expansion},
    author    = {Tardivo, Fabio and Michel, Laurent and van Hoeve, Willem-Jan},
    booktitle = {Proceedings of CPAIOR},
    year      = {2026}
}
```