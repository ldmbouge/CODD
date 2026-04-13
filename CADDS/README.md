# CODD

CODD is a C++/CUDA solver for combinatorial optimization problems represented as state-based models. It implements **Complete Anytime Decision Diagram Search (CADDS)** with optional GPU acceleration, achieving speedups of up to two orders of magnitude over sequential methods.

## Overview

CODD requires the user to specify four core components:

| Component | Description |
|-----------|-------------|
| **State** | The information characterizing the problem |
| **Labeling function** | Returns the possible decisions from a given state |
| **Transition function** | Returns the successor state resulting from a decision |
| **Cost function** | Returns the cost of a decision |

Two optional components can further accelerate solving:

| Component | Description |
|-----------|-------------|
| **Dominance rule** | A condition to discard a state dominated by another |
| **Heuristic function** | Returns an approximate cost-to-go from a state |

**CADDS** views the search space as a Multi-valued Decision Diagram (MDD) and incrementally explores it, collecting improving solutions until optimality is proven.

**GPU acceleration** exploits the layered structure of MDDs: given a set of states from the same layer, the GPU parallelizes both successor generation and the filtering of duplicate, dominated, and suboptimal states.

## Repository Structure

```
├── examples/       # Entry points: one file per problem/backend combination
├── gfl/            # GPU-Friendly Library: types, bitsets, allocators, views, ...
└── src/            # Solver core: expansion engines, nodes, queue, ...
```

## Requirements

| Tool | Minimum Version | Tested Version |
|------|----------------|----------------|
| CMake | 3.28 | 3.28, 4.3      |
| CUDA Toolkit | 12.5 | 12.5, 13.2     |
| GCC | 13.3 | 13.3, 15.2     |
> Tested on Ubuntu 24.04 LTS and Arch Linux.

## Build

### GPU build

Requires an NVIDIA GPU and a compatible CUDA Toolkit installation.

```bash
mkdir build 
cd build
cmake .. -DENABLE_GPU=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
make
```

> Adapt the path to `nvcc` to match your local CUDA installation.

### CPU-only build

```bash
mkdir build 
cd build
cmake ..
make
```

### Build type

CMake defaults to `Release`. To set it explicitly:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

> Valid options are `Release`, `RelWithDebInfo`, and `Debug`.

## Run

The `data/tsptwAsInt` folder contains TSPTW instances with distances scaled to integer values.

### GPU

```
./TsptwCaddsGpu -m 8 -i ../../data/tsptwAsInt/SolnonFeasible/n31g60b40.001.txt
Search: CADD
Engine: GPU
Instance: ../../data/tsptwAsInt/SolnonFeasible/n31g60b40.001.txt
Fragment: Auto
Memory: 8 GB
Timeout: None
---
Time [s]         Primal       Expanded          Queue        Nodes/s
    2.64         607.00       37828021        4445764       14315694
    2.73         603.00       42192609        3358122       51863444
    2.83         602.00       47787834        2286592       53299754
---
Status           = Completed
Extracted        = 80948258
Queue            = 0
Search Time      = 3.64
Solution Time    = 2.83
Solution Cost    = 602.00
Solution         = 17,25,23,27,7,28,6,24,19,22,4,13,14,20,16,11,10,9...
```

### CPU

```
./TsptwCaddsSeq -m 8 -i ../../data/tsptwAsInt/SolnonFeasible/n31g60b40.001.txt
Search: CADD
Engine: Sequential
Instance: ../../data/tsptwAsInt/SolnonFeasible/n31g60b40.001.txt
Fragment: Auto
Memory: 8 GB
Timeout: None
---
Time [s]         Primal       Expanded          Queue        Nodes/s
   10.00              -        6597341         340238         801761
   15.00              -        8797631         516626         440012
   20.00              -       11131852         863761         466797
   ...
   67.90         611.00       42298169        7506426         370579
   73.01         611.00       50852076        3379949        1675323
   74.72         606.00       53463209        3379949        1521605
   ...
   88.27         602.00       71499839         863761        1276913
   94.01         602.00       78540049         340238        1225870
   99.01         602.00       83671218              0        1026132
---
Status           = Completed
Extracted        = 87101286
Queue            = 0
Search Time      = 101.47
Solution Time    = 88.27
Solution Cost    = 602.00
Solution         = 17,25,23,27,7,28,6,24,19,22,4,13,14,20,16,11,10,9,...
```

## Citation

```bibtex
@inproceedings{tardivo2025cadds,
    title     = {Complete Anytime Decision Diagram Search with GPU-Accelerated State Expansion},
    author    = {Tardivo, Fabio and Michel, Laurent and van Hoeve, Willem-Jan},
    booktitle = {Proceedings of CPAIOR},
    year      = {2026}
}
```