#!/usr/bin/env bash

cd approxmatch
chmod +x tf*.sh
sh tf*.sh
cd ..

cd grouping
chmod +x tf*.sh
sh tf*.sh
cd ..

cd interpolation
chmod +x tf*.sh
sh tf*.sh
cd ..

cd nn_distance
chmod +x tf*.sh
sh tf*.sh 
cd ..

cd sampling
chmod +x tf*.sh
sh tf*.sh
cd ..


cd renderball
chmod +x tf*.sh
sh tf*.sh
cd ..
