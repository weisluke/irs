nvcc -std=c++17 --use_fast_math -O3 -D IPM_map -D is_float -o ipm_gpu ./irs/main.cu
