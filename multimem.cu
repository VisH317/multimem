#include "multicast_alloc.cuh"
#include <vector>
#include <iostream>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void multiMemKernel(CUdeviceptr inputMultimemAddr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024*1024*1024) {
        float result = 0;
        // Multimem load-reduce (add) operation from input
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.add.f32 %0, [%1];" 
            : "=f"(result) 
            : "l"(inputMultimemAddr + idx * sizeof(float))
        );
        
        // Multimem store operation to output
        // asm volatile(
        //     "multimem.st.relaxed.sys.f32 [%0], %1;" 
        //     : 
        //     : "l"(outputMultimemAddr + idx * sizeof(float)), "f"(result)
        // );

        printf("inside kernel result: %f\n", result);
    }
}

int main() {
    try {
        // std::cout<<"testing"<<std::endl;
        MC::initializeCuda();

        std::cout<<"cuda initialized"<<std::endl;

        // initialize multicast vars
        std::vector<int> gpus = { 0, 1 };
        size_t size = 1024 * 1024 * 16; // 1 GB allocation

        MC::MCResource mc = MC::createMulticastObject(size, gpus.size());

        std::cout<<"mc object initialized"<<std::endl;

        for(int gpu : gpus) {
            std::cout << "adding memory to multicast for device " << gpu << std::endl;
            CUmemAccessDesc accessDesc;
            accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDesc.location.id = gpu;  // We'll set this for each GPU
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            MC::bindDeviceMemToMulticast(&mc, gpu, accessDesc);
        }

        // // memset the allocated addresses
        // float fvalue = 1.0;
        // int value = *(int*)&fvalue;
        // MC::populateMemory(&mc, value);

        // // create the multimem address, now stored in mc->mcBuff
        // MC::allocateMultimemAddress(&mc, accessDesc, 0);

        // std::cout << "Running multimem kernel" << std::endl;
        // multiMemKernel<<<1, 1>>>(mc.mcBuff);
        // gpuErrchk(cudaPeekAtLastError());
        // gpuErrchk(cudaDeviceSynchronize());
        // std::cout << "Multimem kernel complete" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}