#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <iterator>
#include <memory>
#include <vector>

static void checkCudaErrors(CUresult result, const char* func, const char* file, int line) {
    if (result != CUDA_SUCCESS) {
        const char* errorString;
        cuGetErrorString(result, &errorString);
        throw std::runtime_error(std::string(func) + " failed with error " + 
                                    errorString + " at " + file + ":" + std::to_string(line));
    }
}

#define CHECK_CUDA(func) checkCudaErrors(func, #func, __FILE__, __LINE__)

namespace MC {
    struct MCResource {
        CUmemGenericAllocationHandle mcHandle;
        size_t size = 0;
        CUdeviceptr mcBuff = 0; // only one because I'm not sure if you can have multiple multimem addresses referencing the same handle
        std::vector<CUDeviceptr> deviceMem;
        std::vector<CUmemGenericAllocationHandle> deviceVHandle;
        std::vector<int> gpuIds;
        bool deinit = false;
    };

    void initializeCuda() {
        CHECK_CUDA(cuInit(0));
    }

    MCResource createMulticastObject(size_t size, unsigned int ngpus) {
        CUmemGenericAllocationHandle handle;
        CUmulticastObjectProp prop = { .size = size, .numDevices = ngpus };
        CHECK_CUDA(cuMulticastCreate(&handle, &prop));

        MCResource mc = { .size = size, .mcHandle = handle };
    }

    void bindDeviceMemToMulticast(MCResource* mc, int gpuId, CUMemAccessDesc accessDesc) {
        if(mc->deinit) {
            fprintf(stderr, "Cannot manipulate MCResource with released multicast descriptor");
            return;
        }
        CUDeviceptr ptr = 0;
        CUmemAllocationProp prop;
        size_t gran;
        CUmemGenericAllocationHandle handle;

        memset(&prop, 0, sizeof(prop));
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; // allocating pinned memory
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; // allocation on device
        prop.location.id = gpuId;

        CHECK_CUDA(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)); // flag: recommended or minimum granularity

        // 1. reserve virtual memory on specified device, bind virtual memory ptr
        CHECK_CUDA(cuMemAddressReserve(&ptr, mc->size, gran, 0U, 0)); // 0U is starting address range
        CHECK_CUDA(cuMemCreate(&handle, mc->size, &prop, 0)); // 2. create physical memory on specified device, bind to local handle object
        CHECK_CUDA(cuMemMap(ptr, mc->size, 0, handle, 0)); // 3. map the virtual memory pointer to the physical memory handle
        CHECK_CUDA(cuMemSetAccess(ptr, mc->size, &accessDesc, 1)); // 4. set access descriptor for virtual memory after binding
        // NOTE: now that the device ptr is bound to the physical memory handle, the pointer can be used like regular device memory
        // CHECK_CUDA(cudaMemset((void*)ptr, value, mc->size));

        // add to struct
        mc->deviceMem.push_back(ptr);
        mc->deviceVHandle.push_back(handle);
        mc->gpuIds.push_back(gpuId);


        // 5. bind virtual mem from device onto multicast object
        CHECK_CUDA(cuMulticastBindMem(mc->mcHandle, mc->size, 0, handle, 0, mc->size, 0));
    }

    // creates a multicast address that is accessible (cannot use the handle directly, this will return a usable CUDeviceptr)
    void allocateMultimemAddress(MCResource* mc, CUMemAccessDesc accessDesc) {
        if(mc->deinit) {
            fprintf(stderr, "Cannot manipulate MCResource with released multicast descriptor");
            return;
        }

        if(mc->mcBuff) {
            fprintf(stderr, "multimem address is already initialized, cannot create more than one");
            return;
        }

        CUDeviceptr ptr = 0;
        size_t gran;
        CUmemGenericAllocationHandle handle;

        memset(&prop, 0, sizeof(prop));
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; // allocating pinned memory
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; // allocation on device
        prop.location.id = gpuId;

        CHECK_CUDA(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)); // flag: recommended or minimum granularity

        CHECK_CUDA(cuMemAddressReserve(ptr, mc->size, gran, 0U, 0)); // 1. create a new virtual memory address
        CHECK_CUDA(cuMemMap(ptr, mc->size, 0, mc->mcHandle, 0)); // 2. map the multicast object to the created deviceptr virtual memory address

        CHECK_CUDA(cuMemSetAccess(ptr, mc->size, &accessDesc, 1));

        mc->mcBuff = ptr;
    }

    void populateMemory(MCResource* mc, int value) {
        if(mc->deinit) {
            fprintf(stderr, "Cannot manipulate MCResource with released multicast descriptor");
            return;
        }
        
        for (size_t i = 0; i < mc->deviceMem.size(); ++i) {
            CUcontext context;
            CHECK_CUDA(cuCtxCreate(&context, 0, mc->gpuIds[i]));
            std::cout << "Writing memory to GPU " << mc->gpuIds[i] << " at address " << mc->deviceMem[i] << std::endl;
            CHECK_CUDA(cuMemsetD32(mc->deviceMem[i], value, size / sizeof(int)));
            CHECK_CUDA(cuCtxPopCurrent(&context));
            CHECK_CUDA(cuCtxDestroy(context));
        }
    }

    void releaseDeviceMem(MCResource* mc, int id) {
        if(mc->deinit) {
            fprintf(stderr, "Cannot manipulate MCResource with released multicast descriptor");
            return;
        }
        
        CHECK_CUDA(cuMemUnmap(mc->deviceMem[id], mc->size)); // unmap virtual memory address to physical memory handle
        CHECK_CUDA(cuMemRelease(mc->deviceVHandle[id])); // release the physical memory handle
        CHECK_CUDA(cuMemAddressFree(mc->deviceMem[id], mc->size)); // release the virtual memory

        // remove from array
        mc->deviceMem.erase(std::next(mc->deviceMem.begin(), id));
        mc->deviceVHandle.erase(std::next(mc->deviceVHandle.begin(), id));
    }

    void releaseAllDeviceMem(MCResource* mc) {
        if(mc->deinit) {
            fprintf(stderr, "Cannot manipulate MCResource with released multicast descriptor");
            return;
        }

        for(int i = 0; i < mc->deviceMem.size(); i++) {
            CHECK_CUDA(cuMemUnmap(mc->deviceMem[i], mc->size));
            CHECK_CUDA(cuMemRelease(mc->deviceVHandle[i]));
            CHECK_CUDA(cuMemAddressFree(mc->deviceMem[i], mc->size));
        }

        // remove all leftover objects
        mc->deviceMem.clear();
        mc->deviceVHandle.clear();
    }

    void releaseMultimemBuffer(MCResource *mc) {
        if(mc->deinit) {
            fprintf(stderr, "Cannot manipulate MCResource with released multicast descriptor");
            return;
        }
        
        CHECK_CUDA(cuMemUnmap(mc->mcBuff, mc->size)); // unmap from multicast object
        CHECK_CUDA(cuMemAddressFree(mc->mcBuff, mc->size)); // free virtual memory addresses

        mc->mcBuff = 0; // reset main multimem pointer for future allocations
    }

    // must run before ending program, deallocates multicast object
    void releaseMulticastDescriptor(MCResource mc) {
        CHECK_CUDA(cuMemRelease(mcHandle));
        mc->deinit = true;
    }
}
