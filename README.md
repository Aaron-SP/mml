# mml
Repository for the Minimal Math Library

Welcome to the Minimal Math Library! This is the prerelease alpha version 0.1.

This is a 'compile time', 'N variable', non-linear optimization tool.

Algorithms:
- newton multivariate zero
- strictly convex backtracking minimization
- newton hessian minimization

The library builds on Linux and Win32 platforms. A GNU makefile is available for compilation with GCC/MINGW for Win32 platforms or GCC/X11 for Linux platforms. The makefile should work for both environments without modification.

The GNU makefile contains various build targets.

- 'make' - builds all tests
- 'make tests' - builds only tests
- 'make clean' - cleans up all generated output files

These build targets have been tested for compilation on Arch Linux x64 and Windows 7 x64 platforms.

To perform code formatting, run the formatting script command in the project root (clang is required):

- ./format