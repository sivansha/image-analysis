#!/bin/bash
cd build/
cmake ..
make -j 8
cd ..
./build/aia2 ./img/orig.jpg ./img/blatt_art1.jpg ./img/blatt_art2.jpg

