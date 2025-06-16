nvcc -O3 -arch=sm_80 -use_fast_math src/cuda_value.cu src/testmain.cu  -o ./testmain
./testmain
rm -rf ./testmain