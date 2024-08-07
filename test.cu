#include "va_wrapper.cuh"

#include <iostream>
#include <cuda_runtime.h>

void printGranularities() {

    // Calculate minimum and recommended granularity for multicast object
    CUmulticastObjectProp mcProp = {};
    mcProp.size = 1024 * 1024 * 1024;  // 1 GB, same as the size we'll use later
    mcProp.numDevices = 3;  // Matching the number of GPUs we'll use
    mcProp.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
    mcProp.flags = 0;

    size_t minGranularity, recGranularity;
    
    CUresult result = cuMulticastGetGranularity(&minGranularity, &mcProp, CU_MULTICAST_GRANULARITY_MINIMUM);
    if (result != CUDA_SUCCESS) {
        const char* errorString;
        cuGetErrorString(result, &errorString);
        throw std::runtime_error("Failed to get minimum granularity: " + std::string(errorString));
    }

    result = cuMulticastGetGranularity(&recGranularity, &mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED);
    if (result != CUDA_SUCCESS) {
        const char* errorString;
        cuGetErrorString(result, &errorString);
        throw std::runtime_error("Failed to get recommended granularity: " + std::string(errorString));
    }

    std::cout << "Minimum granularity: " << minGranularity << " bytes" << std::endl;
    std::cout << "Recommended granularity: " << recGranularity << " bytes" << std::endl;
}


__global__ void multiMemKernel(CUdeviceptr inputMultimemAddr, CUdeviceptr outputMultimemAddr) {
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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main() {
    try {
        CudaVirtualMemory::initializeCuda();

        printGranularities();

        std::vector<int> gpuIds = {0,1};  // Example: using GPUs 0, 1, and 2
        size_t size = 1024 * 1024 * 16;  // 1 GB

        auto handles = CudaVirtualMemory::reserveMemoryAcrossGPUs(gpuIds, size);
        std::cout << "Successfully allocated memory on " << handles.size() << " GPUs" << std::endl;

        auto mcHandle = CudaVirtualMemory::createMulticastDescriptor(gpuIds, handles, size);
        std::cout << "Successfully created multicast descriptor: " << mcHandle << std::endl;

        auto mappedPtrs = CudaVirtualMemory::mapAllocationsToVA(handles, gpuIds, size);
        std::cout << "Successfully mapped allocations to VA space" << std::endl;

        // Here you can populate the memory with values
        // For example, to populate the first allocation with a value:
        float fvalue = 42.0;
        int value = *(int*)&fvalue;
        CudaVirtualMemory::populateMemory(gpuIds, mappedPtrs, size, value);
        std::cout << "Populated allocations with value: " << value << std::endl;

        std::cout << "Running multimem kernel" << std::endl;
        multiMemKernel<<<1, 1>>>((CUdeviceptr)&mcHandle, (CUdeviceptr)&mcHandle);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        std::cout << "Multimem kernel complete" << std::endl;

        // Copy 1 KB of memory back from each of the mapped pointers on each GPU to host float arrays
        const size_t copySize = 1024; // 1 KB
        const size_t numFloats = copySize / sizeof(float);
        
        std::vector<std::vector<float>> hostArrays(gpuIds.size());
        
        for (size_t i = 0; i < gpuIds.size(); ++i) {
            hostArrays[i].resize(numFloats);
            
            CUcontext context;
            CHECK_CUDA(cuCtxCreate(&context, 0, gpuIds[i]));
            
            CHECK_CUDA(cuMemcpyDtoH(hostArrays[i].data(), mappedPtrs[i], copySize));
            
            CHECK_CUDA(cuCtxPopCurrent(&context));
            CHECK_CUDA(cuCtxDestroy(context));
            
            std::cout << "Copied " << copySize << " bytes from GPU " << gpuIds[i] << std::endl;
            
            // Print the first few values as a sanity check
            std::cout << "First few values from GPU " << gpuIds[i] << ": ";
            for (size_t j = 0; j < std::min(numFloats, size_t(5)); ++j) {
                std::cout << hostArrays[i][j] << " ";
            }
            std::cout << std::endl;
        }

        // Clean up
        CudaVirtualMemory::unmapAndFreeVA(mappedPtrs, size);
        CudaVirtualMemory::releaseMulticastDescriptor(mcHandle);
        CudaVirtualMemory::releaseMemoryAcrossGPUs(handles);

        std::cout << "Successfully released all resources" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}