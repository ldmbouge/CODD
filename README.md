
# Table of Contents

1.  [C++DDOpt](#org4f34ee7)
    1.  [Dependencies](#orga57ae86)
    2.  [C++ Standard](#orgca33d9d)
    3.  [Build system](#org2bad4be)
    4.  [Examples](#orga88e4a4)
    5.  [Unit test](#orgbf2f109)
    6.  [Library](#orgced5e48)
2.  [Tasks](#org2642a52)
    1.  [Decent set of integer implementation](#orgf14d530)
    2.  [Check and fix leaks (the cache should be deallocated, not \_an)](#org0fec76c)
    3.  [Implement instance reader for tsp to do bigger instances](#org321b4da)
    4.  [Implement instance reader for MISP](#org69ac622)
    5.  [Rename MISP (`foo`) to misptoy ;-)](#orgec68b25)
    6.  [Implement an O(1) removal from \_an](#orgc3bbad0)
    7.  [Profile and pick up the low hanging fruits](#org70238b8)


<a id="org4f34ee7"></a>

# C++DDOpt

C++ library implement the DDOpt framework.
It has:

-   restricted / exact / relax diagrams
-   state definition, initial, terminal and state merging functions separated


<a id="orga57ae86"></a>

## Dependencies

You need `graphviz` (The `dot` binary) to create graph images. It happens
automatically when the `display` method is called. Temporary files are created
in `/tmp` and the macOS `open` command is used (via `fork/execlp`)  to open the generated
PDF.


<a id="orgca33d9d"></a>

## C++ Standard

You need a C++-20 capable compiler. gcc and clang should both work. I work on macOS, os
I use the mainline clang coming with Xcode.

The implementation uses templates and concepts to factor the code.


<a id="org2bad4be"></a>

## Build system

This is `cmake`. Simply do the following

    mkdir build
    cd build
    cmake ..
    make -j4

And it will compile the whole thing


<a id="orga88e4a4"></a>

## Examples

To be found in the `examples` folder

-   `coloringtoy` tiny coloring bench (same as in python)
-   `foo` maximum independent set (toy size)
-   `tstpoy` tiny TSP instance (same as in python)
-   `gruler` golomb ruler (usage <size> <ubOnLabels>)


<a id="orgbf2f109"></a>

## Unit test

In the `test` folder


<a id="orgced5e48"></a>

## Library

All of it in the `src` folder


<a id="org2642a52"></a>

# Tasks


<a id="orgf14d530"></a>

## DONE Decent set of integer implementation

On my own heap.
With template overload that is size dependent (up to label 64, all ops should be O(1))
After that, it should be O(label/64). Unless we start using the 128 bit registers ;-)


<a id="org0fec76c"></a>

## DONE Check and fix leaks (the cache should be deallocated, not \_an)


<a id="org321b4da"></a>

## TODO Implement instance reader for tsp to do bigger instances


<a id="org69ac622"></a>

## TODO Implement instance reader for MISP


<a id="orgec68b25"></a>

## TODO Rename MISP (`foo`) to misptoy ;-)


<a id="orgc3bbad0"></a>

## TODO Implement an O(1) removal from \_an


<a id="org70238b8"></a>

## TODO Profile and pick up the low hanging fruits

