
# Table of Contents

1.  [CODD](#org2153b40)
    1.  [Dependencies](#orgfd8c3a3)
    2.  [C++ Standard](#orgc3f268f)
    3.  [Build system](#orgfca022f)
    4.  [Examples](#orgfa2ca46)
    5.  [Unit test](#org8ec9b5d)
    6.  [Library](#org14e7933)
    7.  [Brief documentation](#orgf214349)


<a id="org2153b40"></a>

# CODD

CODD is a C++ library that implements the DDOpt framework.
It has:

-   restricted / exact / relax diagrams
-   state definition, initial, terminal and state merging functions separated
-   equivalence predicate for sink.
-   restricted construction is truly bounded now (not truncation based).


<a id="orgfd8c3a3"></a>

## Dependencies

You need `graphviz` (The `dot` binary) to create graph images. It happens
automatically when the `display` method is called. Temporary files are created
in `/tmp` and then macOS `open` command is used (via `fork/execlp`)  to open the generated
PDF. The same functionality needs to be added on Linux (different API to pop up a viewer).


<a id="orgc3f268f"></a>

## C++ Standard

You need a C++-20 capable compiler. gcc and clang should both work. I work on macOS, and
I use the mainline clang coming with Xcode.

The implementation uses templates and concepts to factor the code.


<a id="orgfca022f"></a>

## Build system

This is `cmake`. Simply do the following

    mkdir build
    cd build
    cmake ..
    make -j4

And it will compile the whole thing. To compile in optimized mode, simply change
the variable `CMAKE_BUILD_TYPE` from `Debug` to `Release` as shown below:

    cmake .. -DCMAKE_BUILD_TYPE=Release


<a id="orgfa2ca46"></a>

## Examples

To be found in the `examples` folder

-   `coloringtoy` tiny coloring bench (same as in python)
-   `foo` maximum independent set (toy size)
-   `tsp` TSP solver (same as in python)
-   `gruler` golomb ruler (usage <size> <ubOnLabels>)
-   `knapsack` Binary knapsack
-   `MISP` for the Maximum independent set

Each benchmark may have several variants that are incresingly faster (because they use local bounds, dominance, better state representation, etc&#x2026;). For instance, `tsp_test4` is one of the variant for TSP. Most executable do take an input for the filename containing the instance as well as the width. For instance running `tsp_test4` can be done as follows:

    ./tsp_test4 ../data/atsp/br17.atsp 32

from the `build` folder to run on the br17 instance with a width of 32. The output would look like this:

    ities:{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}
    sz(LightHashtable):200191
    B&B searching...
    sz(LightHashtable):200191
    sz(LightHashtable):200191
    B&B Nodes          Dual	Primal	Gap(%)
    ----------------------------------------------
    TIME:80
    P TIGHTEN: <P:42, INC:(17)[12 10 1 2 13 9 16 7 8 14 5 6 3 4 15 11 0 ]>
    B&B(    1)	      6	     42	 85.71%	time:  0.08s
    TIME:2203
    P TIGHTEN: <P:41, INC:(17)[11 2 13 12 9 10 1 8 16 7 14 5 6 3 4 15 0 ]>
    B&B(   48)	     11	     41	 73.17%	time: 2.203s
    TIME:2242
    P TIGHTEN: <P:40, INC:(17)[11 9 12 1 2 13 10 8 16 7 3 4 6 15 5 14 0 ]>
    B&B(   49)	     11	     40	  72.5%	time: 2.242s
    TIME:3063
    P TIGHTEN: <P:39, INC:(17)[11 13 2 1 10 12 9 6 14 5 15 3 4 8 16 7 0 ]>
    B&B(   69)	     11	     39	 71.79%	time: 3.063s
    B&B(  294)	     13	     39	 66.67%	time: 8.067s
    B&B(  758)	     13	     39	 66.67%	time: 13.08s
    B&B( 1348)	     17	     39	 56.41%	time: 18.09s
    B&B( 3418)	     19	     39	 51.28%	time:  23.1s
    B&B( 6569)	     24	     39	 38.46%	time:  28.1s
    Done(32):39	#nodes:13912/13912	P/D:0/0	Time:3.063/31.995s	LIM?:0

Every 5 seconds, the branch and bound reports the number of nodes, current node value, incumbent value, duality gap and the time since the start. The last line (with the `Done`) says that the optimal was 39, that it took 13912 nodes to close the optimality proof, that no nodes where discarded because of a dominance and that the optimum was found after 3.063s and the proof took 31.995s (if there was a limit, it was <span class="underline">not</span> reached). 


<a id="org8ec9b5d"></a>

## Unit test

In the `test` folder


<a id="org14e7933"></a>

## Library

All of it in the `src` folder


<a id="orgf214349"></a>

## Brief documentation

A [small site](./doc/CODD.html) with some documentation in HTML is available too.

