nvcc -o $1 $1.cu -arch=sm_90 -lcuda -lcudart
echo "Compilation complete! running..."
./$1