CXX = g++

all: everybeam_example

everybeam_example: everybeam_example.cpp
	$(CXX) -O3 -march=native \
    -leverybeam -lcasa_ms -lcasa_measures -lcasa_casa \
    everybeam_example.cpp -o everybeam_example
