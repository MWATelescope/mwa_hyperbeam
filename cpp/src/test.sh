#!/bin/bash

set -eux

if [[ beam2016implementation.cpp -nt libbeam.so ]]; then
    g++ -std=c++11 -shared -O3 -fPIC system.cpp beam2016implementation.cpp -lhdf5 -lhdf5_cpp -lboost_filesystem -o libbeam.so
fi
if [[ test.cpp -nt test ]]; then
    g++ -std=c++11 -O3 -fPIC test.cpp -L . -lbeam -o test
fi
LD_LIBRARY_PATH+=":." ./test
