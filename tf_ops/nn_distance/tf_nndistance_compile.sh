



#!/usr/bin/env bash
nvcc=/usr/local/cuda-10.0/bin/nvcc
cudalib=/usr/local/cuda-10.0/lib64/
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

$nvcc  tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11  -I $TF_INC -DGOOGLE_CUDA=1\
 -x cu -Xcompiler -fPIC -O2

g++ tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so   -std=c++11 -shared -fPIC -I $TF_INC \
-I$TF_INC/external/nsync/public -L$TF_LIB  -l:libtensorflow_framework.so -lcudart -L $cudalib -O2



