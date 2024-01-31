
# Table of Contents

1.  [C++DDOpt](#orgd83ee0d)
    1.  [Dependencies](#org116bc10)
    2.  [C++ Standard](#orgdd64fdb)
    3.  [Build system](#orga319374)
    4.  [Examples](#orgd8f8440)
    5.  [Unit test](#org28b73e6)
    6.  [Library](#org2ce19b9)
2.  [Tasks](#org8a5c4c8)
    1.  [Decent set of integer implementation](#orgbcc1555)
    2.  [Check and fix leaks (the cache should be deallocated, not \_an)](#org3c4baf9)
    3.  [Implement instance reader for tsp to do bigger instances](#org69cb0e3)
    4.  [Implement instance reader for MISP](#orga16bfec)
    5.  [Rename MISP (`foo`) to misptoy ;-)](#org0efee95)
    6.  [Implement an O(1) removal from \_an](#org48792c3)
    7.  [Profile and pick up the low hanging fruits](#org700f692)
    8.  [Fix calls to find in order to remove from \_an](#orga9fe759)
    9.  [Fix calls to find before updateKey in heaps](#org5081633)
    10. [Change the makeNode / duplicate so that hash is computed only once (not twice).](#org6584e9b)


<a id="orgd83ee0d"></a>

# C++DDOpt

C++ library implement the DDOpt framework.
It has:

-   restricted / exact / relax diagrams
-   state definition, initial, terminal and state merging functions separated


<a id="org116bc10"></a>

## Dependencies

You need `graphviz` (The `dot` binary) to create graph images. It happens
automatically when the `display` method is called. Temporary files are created
in `/tmp` and the macOS `open` command is used (via `fork/execlp`)  to open the generated
PDF.


<a id="orgdd64fdb"></a>

## C++ Standard

You need a C++-20 capable compiler. gcc and clang should both work. I work on macOS, os
I use the mainline clang coming with Xcode.

The implementation uses templates and concepts to factor the code.


<a id="orga319374"></a>

## Build system

This is `cmake`. Simply do the following

    mkdir build
    cd build
    cmake ..
    make -j4

And it will compile the whole thing


<a id="orgd8f8440"></a>

## Examples

To be found in the `examples` folder

-   `coloringtoy` tiny coloring bench (same as in python)
-   `foo` maximum independent set (toy size)
-   `tstpoy` tiny TSP instance (same as in python)
-   `gruler` golomb ruler (usage <size> <ubOnLabels>)


<a id="org28b73e6"></a>

## Unit test

In the `test` folder


<a id="org2ce19b9"></a>

## Library

All of it in the `src` folder


<a id="org8a5c4c8"></a>

# Tasks


<a id="orgbcc1555"></a>

## DONE Decent set of integer implementation

On my own heap.
With template overload that is size dependent (up to label 64, all ops should be O(1))
After that, it should be O(label/64). Unless we start using the 128 bit registers ;-)


<a id="org3c4baf9"></a>

## DONE Check and fix leaks (the cache should be deallocated, not \_an)


<a id="org69cb0e3"></a>

## TODO Implement instance reader for tsp to do bigger instances


<a id="orga16bfec"></a>

## TODO Implement instance reader for MISP


<a id="org0efee95"></a>

## TODO Rename MISP (`foo`) to misptoy ;-)


<a id="org48792c3"></a>

## TODO Implement an O(1) removal from \_an


<a id="org700f692"></a>

## DONE Profile and pick up the low hanging fruits


<a id="orga9fe759"></a>

## TODO Fix calls to find in order to remove from \_an

-   Those should be O(1) via locators.
-   Adopt Vec<T> rather than the std::vector<T> for \_an
-   Create a Vec<T> version that is location aware
-   Implement the trick to O(1) removal (affects mergeLayer / truncate)


<a id="org5081633"></a>

## TODO Fix calls to find before updateKey in heaps

-   Heap is already location aware
-   We need to track the location (by node id, we have those)
-   Then use the location to have an O(1) operation (affects computeBest & computeBestBackward)


<a id="org6584e9b"></a>

## TODO Change the makeNode / duplicate so that hash is computed only once (not twice).

-   Use opaque ADT in Hashtable to support that

