#!/bin/bash

set -eux

g++ -std=c++11 -shared -O3 -fPIC system.cpp beam2016implementation.cpp -lhdf5 -lhdf5_cpp -lboost_filesystem -o libbeam.so
g++ -std=c++11 -O3 -fPIC test.cpp -L . -lbeam -o test
LD_LIBRARY_PATH+=":." ./test
