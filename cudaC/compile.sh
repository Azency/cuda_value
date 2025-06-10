nvcc -O3 -arch=sm_80 -use_fast_math cuda_value.cu testmain.cu -o testmain
./testmain
rm -rf ./testmain