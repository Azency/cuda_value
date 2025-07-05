# nvcc -O3 -arch=sm_80 -use_fast_math --relocatable-device-code=true  src/cuda_value.cu src/testmain.cu  -o ./testmain -lcudadevrt

# nv
# CUDA_VISIBLE_DEVICES=4 ./testmain
# rm -rf ./testmain


# 测试性能
rm -rf ./testmain
nvcc -lineinfo -g -O2 -Xcompiler -Wall -dc src/cuda_value.cu -o cuda_value.o
nvcc -lineinfo -g -O2 -Xcompiler -Wall -dc src/testmain.cu -o testmain.o
nvcc cuda_value.o testmain.o -o ./testmain -lcudadevrt
rm -rf cuda_value.o testmain.o
CUDA_VISIBLE_DEVICES=4 ./testmain
rm -rf ./testmain

