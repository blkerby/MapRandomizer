#!/bin/bash
c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) cpp/connectivity.cpp -o connectivity$(python3-config --extension-suffix)
