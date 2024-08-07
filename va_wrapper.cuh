#include <cuda.h>
#include <vector>
#include <stdexcept>
#include <iostream>

static void checkCudaErrors(CUresult result, const char* func, const char* file, int line) {
    if (result != CUDA_SUCCESS) {
        const char* errorString;
        cuGetErrorString(result, &errorString);
        throw std::runtime_error(std::string(func) + " failed with error " + 
                                    errorString + " at " + file + ":" + std::to_string(line));
    }
}

class CudaVirtualMemory {
public:
    #define CHECK_CUDA(func) checkCudaErrors(func, #func, __FILE__, __LINE__)

    static void initializeCuda() {
        CHECK_CUDA(cuInit(0));
    }

    static std::vector<CUmemGenericAllocationHandle> reserveMemoryAcrossGPUs(
        const std::vector<int>& gpuIds,
        size_t size
    ) {
        std::vector<CUmemGenericAllocationHandle> handles;
        handles.reserve(gpuIds.size());

        for (int gpuId : gpuIds) {

            CUmemGenericAllocationHandle handle;
            CUmemAllocationProp prop = {};

            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = gpuId;

            CUresult result = cuMemCreate(&handle, size, &prop, 0);
            if (result != CUDA_SUCCESS) {
                const char* errorString;
                cuGetErrorString(result, &errorString);
                std::cerr << "cuMemCreate failed with error " << errorString 
                          << " on GPU " << gpuId << std::endl;
                continue;
            }

            handles.push_back(handle);
        }

        return handles;
    }

    static void releaseMemoryAcrossGPUs(
        const std::vector<CUmemGenericAllocationHandle>& handles
    ) {
        for (const auto& handle : handles) {
            CHECK_CUDA(cuMemRelease(handle));
        }
    }


    static CUmemGenericAllocationHandle createMulticastDescriptor(
        const std::vector<int>& gpuIds,
        const std::vector<CUmemGenericAllocationHandle>& handles,
        size_t size
    ) {
        if (gpuIds.size() != handles.size()) {
            throw std::runtime_error("Number of GPU IDs must match number of handles");
        }

        CUmulticastObjectProp mcProp = {};
        mcProp.size = 1024*1024*1024; // for now. note this seems to need to be a multiple of the minimum granularity.
        mcProp.numDevices = static_cast<unsigned int>(gpuIds.size());
        mcProp.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
        mcProp.flags = 0;

        CUmemGenericAllocationHandle mcHandle;
        CHECK_CUDA(cuMulticastCreate(&mcHandle, &mcProp));

        for (size_t i = 0; i < gpuIds.size(); ++i) {
            CUdevice device;
            CHECK_CUDA(cuDeviceGet(&device, gpuIds[i]));
            CHECK_CUDA(cuMulticastAddDevice(mcHandle, device));
        }
        
        for (size_t i = 0; i < handles.size(); ++i) {
            CHECK_CUDA(cuMulticastBindMem(mcHandle, 0, handles[i], 0, size, 0));
        }

        return mcHandle;
    }

    static void releaseMulticastDescriptor(CUmemGenericAllocationHandle mcHandle) {
        CHECK_CUDA(cuMemRelease(mcHandle));
    }

    static std::vector<CUdeviceptr> mapAllocationsToVA(
        const std::vector<CUmemGenericAllocationHandle>& handles,
        const std::vector<int>& gpuIds,
        size_t size
    ) {
        std::vector<CUdeviceptr> mappedPtrs;
        mappedPtrs.reserve(handles.size());

        for (size_t i = 0; i < handles.size(); ++i) {
            CUdeviceptr ptr;
            
            // Reserve VA space
            CHECK_CUDA(cuMemAddressReserve(&ptr, size, 0, 0, 0));

            // Map the allocation to the reserved VA space
            CHECK_CUDA(cuMemMap(ptr, size, 0, handles[i], 0));

            // Set access permissions
            CUmemAccessDesc accessDesc = {};
            accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDesc.location.id = gpuIds[i];  // We'll set this for each GPU
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            CHECK_CUDA(cuMemSetAccess(ptr, size, &accessDesc, 1));

            mappedPtrs.push_back(ptr);
        }

        return mappedPtrs;
    }

    static void populateMemory(const std::vector<int>& gpuIds, const std::vector<CUdeviceptr>& ptrs, size_t size, int value) {
        for (size_t i = 0; i < ptrs.size(); ++i) {
            CUcontext context;
            CHECK_CUDA(cuCtxCreate(&context, 0, gpuIds[i]));
            std::cout << "Writing memory to GPU " << gpuIds[i] << " at address " << ptrs[i] << std::endl;
            CHECK_CUDA(cuMemsetD32(ptrs[i], value, size / sizeof(int)));
            CHECK_CUDA(cuCtxPopCurrent(&context));
            CHECK_CUDA(cuCtxDestroy(context));
        }
    }

    static void unmapAndFreeVA(const std::vector<CUdeviceptr>& ptrs, size_t size) {
        for (const auto& ptr : ptrs) {
            CHECK_CUDA(cuMemUnmap(ptr, size));
            CHECK_CUDA(cuMemAddressFree(ptr, size));
        }
    }
};