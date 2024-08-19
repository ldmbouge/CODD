
# Table of Contents

1.  [CODD](#org21a8a30)
    1.  [Dependencies](#org41dad86)
    2.  [C++ Standard](#org08ed221)
    3.  [Build system](#org4b62dc2)
    4.  [Examples](#org9bbb0bd)
    5.  [Unit test](#org236ff64)
    6.  [Library](#org9874101)
2.  [Tasks](#org021c718):noexports:
    1.  [Decent set of integer implementation](#org8b68b17)
    2.  [Check and fix leaks (the cache should be deallocated, not \_an)](#orge8cf3d9)
    3.  [Implement instance reader for tsp to do bigger instances](#org28e88db)
    4.  [Implement instance reader for MISP](#org1c9091e)
    5.  [Rename MISP (`foo`) to misptoy ;-)](#org52f9ad1)
    6.  [Profile and pick up the low hanging fruits](#orgc45eabc)
    7.  [Fix calls to find in order to remove from \_an](#org1c4306a)
    8.  [Fix calls to find before updateKey in heaps](#orgc4edd23)
    9.  [Change the makeNode / duplicate so that hash is computed only once (not twice).](#org4bf1b60)
    10. [Experiment with permanent state cache](#org12a7c58)
        1.  [segregate Edge allocator](#org969fe64)
        2.  [keep the node cache (at least for relaxed) so that they get reused](#orgc0bd272)
        3.  [clear the edge allocator since those must be rebuild](#org46443df)
        4.  [runs the risk of runaway node cache. Maybe clear periodically? (Every 10K B&B node)](#org66ac2ba)
    11. [Implement a label generator](#org739d1d1)
    12. [Cleanup the edge transfer (no more allocating, just moving)](#org1d95658)
    13. [Change relax to merge as we go](#org935f8f6)


<a id="org21a8a30"></a>

# CODD

CODD is a C++ library that implements the DDOpt framework.
It has:

-   restricted / exact / relax diagrams
-   state definition, initial, terminal and state merging functions separated
-   equivalence predicate for sink.
-   restricted construction is truly bounded now (not truncation based).


<a id="org41dad86"></a>

## Dependencies

You need `graphviz` (The `dot` binary) to create graph images. It happens
automatically when the `display` method is called. Temporary files are created
in `/tmp` and then macOS `open` command is used (via `fork/execlp`)  to open the generated
PDF. The same functionality needs to be added on Linux (different API to pop up a viewer).


<a id="org08ed221"></a>

## C++ Standard

You need a C++-20 capable compiler. gcc and clang should both work. I work on macOS, and
I use the mainline clang coming with Xcode.

The implementation uses templates and concepts to factor the code.


<a id="org4b62dc2"></a>

## Build system

This is `cmake`. Simply do the following

    mkdir build
    cd build
    cmake ..
    make -j4

And it will compile the whole thing. To compile in optimized mode, simply change
the variable `CMAKE_BUILD_TYPE` from `Debug` to `Release` as shown below:

    cmake .. -DCMAKE_BUILD_TYPE=Release


<a id="org9bbb0bd"></a>

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


<a id="org236ff64"></a>

## Unit test

In the `test` folder


<a id="org9874101"></a>

## Library

All of it in the `src` folder


<a id="org021c718"></a>

# Tasks     :noexports:


<a id="org8b68b17"></a>

## DONE Decent set of integer implementation

On my own heap.
With template overload that is size dependent (up to label 64, all ops should be O(1))
After that, it should be O(label/64). Unless we start using the 128 bit registers ;-)


<a id="orge8cf3d9"></a>

## DONE Check and fix leaks (the cache should be deallocated, not \_an)


<a id="org28e88db"></a>

## TODO Implement instance reader for tsp to do bigger instances

Done for coloring.


<a id="org1c9091e"></a>

## TODO Implement instance reader for MISP


<a id="org52f9ad1"></a>

## TODO Rename MISP (`foo`) to misptoy ;-)


<a id="orgc45eabc"></a>

## DONE Profile and pick up the low hanging fruits


<a id="org1c4306a"></a>

## DONE Fix calls to find in order to remove from \_an

-   Those should be O(1) via locators.
-   Implement the trick to O(1) removal (affects mergeLayer / truncate)
-   I now directly link the ANode with each other. It avoids the needs for location. Removal can still be O(1).


<a id="orgc4edd23"></a>

## DONE Fix calls to find before updateKey in heaps

-   Heap is already location aware
-   We need to track the location (by node id, we have those)
-   Then use the location to have an O(1) operation (affects computeBest & computeBestBackward)


<a id="org4bf1b60"></a>

## DONE Change the makeNode / duplicate so that hash is computed only once (not twice).

-   Use opaque ADT in Hashtable to support that (HTAt is the opaque type)


<a id="org12a7c58"></a>

## Experiment with permanent state cache


<a id="org969fe64"></a>

### segregate Edge allocator


<a id="orgc0bd272"></a>

### DONE keep the node cache (at least for relaxed) so that they get reused

This did not work. It creates and keeps far too many nodes. Collision lists were getting too long. It's far easier to clear and rebuild as many DDs are quite small. 
CLOSED: <span class="timestamp-wrapper"><span class="timestamp">[2024-02-15 Thu 16:46]</span></span>


<a id="org46443df"></a>

### clear the edge allocator since those must be rebuild


<a id="org66ac2ba"></a>

### runs the risk of runaway node cache. Maybe clear periodically? (Every 10K B&B node)


<a id="org739d1d1"></a>

## DONE Implement a label generator


<a id="org1d95658"></a>

## DONE Cleanup the edge transfer (no more allocating, just moving)


<a id="org935f8f6"></a>

## TODO Change relax to merge as we go

