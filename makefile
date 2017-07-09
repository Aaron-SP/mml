# MML Makefile - CYGWIN | LINUX

# Include directories
LIB_SOURCES = -Isource/math
TEST_SOURCES = -Itest/math
TEST = test/test.cpp

# Compile parameters
PARAMS = -std=c++14 -Wall -O3 -march=native -fomit-frame-pointer -freciprocal-math -ffast-math --param max-inline-insns-auto=100 --param early-inlining-insns=200

# Linker parameters
ifeq ($(OS),Windows_NT)
	MML_PATH = C:/cygwin/usr/i686-w64-mingw32/sys-root/mingw/include/mml
else
	MML_PATH = /usr/include/mml
endif

# Override if MGL_DESTDIR specified
ifdef MML_DESTDIR
	MML_PATH = $(MML_DESTDIR)/mml
endif

# Default run target
default: tests

install:
	mkdir -p $(MML_PATH)
	cp -r source/* $(MML_PATH)
uninstall:
	rm -rI $(MML_PATH)
tests:
	g++ $(LIB_SOURCES) $(TEST_SOURCES) -Itest $(PARAMS) $(TEST) -o bin/test 2> "gcc.txt"

# clean targets
clean: clean_junk clean_tests
clean_junk:
	rm -f gcc.txt
clean_tests:
	rm -f bin/test