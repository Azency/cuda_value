nvcc -O3 -arch=sm_80 -use_fast_math --relocatable-device-code=true  src/cuda_value.cu src/testmain.cu  -o ./testmain -lcudadevrt
CUDA_VISIBLE_DEVICES=4 ./testmain
rm -rf ./testmain