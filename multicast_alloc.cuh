#include <cuda.h>
#include <vector>
#include <stdexcept>
#include <iostream>


static void checkCudaErrors(CUresult result, const char* func, const char* file, int line) {
    if (result != CUDA_SUCCESS) {
        const char* errorString;
        cuGetErrorString(result, &errorString);
        throw std::runtime_error(std::string(func) + " failed with error \"" + 
                                    errorString + "\" at " + file + ":" + std::to_string(line));
    }
}

#define CHECK_CUDA(func) checkCudaErrors(func, #func, __FILE__, __LINE__)

namespace MC {
    struct MCResource {
        CUmemGenericAllocationHandle mcHandle;
        size_t size = 0;
        CUdeviceptr mcBuff = 0; // only one because I'm not sure if you can have multiple multimem addresses referencing the same handle
        std::vector<CUdeviceptr> deviceMem;
        std::vector<CUmemGenericAllocationHandle> deviceVHandle;
        std::vector<int> gpuIds;
        bool deinit = false;

        MCResource(size_t size, CUmemGenericAllocationHandle handle) : size(size), mcHandle(handle) {};
    };

    void initializeCuda() {
        CHECK_CUDA(cuInit(0));
    }

    MCResource createMulticastObject(size_t size, unsigned int ngpus) {
        CUmulticastObjectProp mcProp = {};
        mcProp.size = 1024*1024*1024; // for now. note this seems to need to be a multiple of the minimum granularity.
        mcProp.numDevices = ngpus;
        mcProp.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
        mcProp.flags = 0;

        CUmemGenericAllocationHandle mcHandle;
        CHECK_CUDA(cuMulticastCreate(&mcHandle, &mcProp));

        // /////////
        // int gpuId = 0;

        // CUmemAccessDesc accessDesc;
        // accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        // accessDesc.location.id = gpuId;  // We'll set this for each GPU
        // accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        // CUdevice device;
        // CHECK_CUDA(cuDeviceGet(&device, gpuId));
        // CHECK_CUDA(cuMulticastAddDevice(mcHandle, device));
        // std::cout<<"device "<<gpuId<<" added to multicast list of devices"<<std::endl;

        // CUdeviceptr ptr = 0;
        // CUmemAllocationProp prop;
        // size_t gran;
        // CUmemGenericAllocationHandle handle;

        // memset(&prop, 0, sizeof(prop));
        // prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; // allocating pinned memory
        // prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; // allocation on device
        // prop.location.id = gpuId;

        // CHECK_CUDA(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)); // flag: recommended or minimum granularity

        // // 1. reserve virtual memory on specified device, bind virtual memory ptr
        // CHECK_CUDA(cuMemAddressReserve(&ptr, size, gran, 0U, 0)); // 0U is starting address range
        // CHECK_CUDA(cuMemCreate(&handle, size, &prop, 0)); // 2. create physical memory on specified device, bind to local handle object
        // CHECK_CUDA(cuMemMap(ptr, size, 0, handle, 0)); // 3. map the virtual memory pointer to the physical memory handle
        // CHECK_CUDA(cuMemSetAccess(ptr, size, &accessDesc, 1)); // 4. set access descriptor for virtual memory after binding
        // // NOTE: now that the device ptr is bound to the physical memory handle, the pointer can be used like regular device memory
        // // CHECK_CUDA(cudaMemset((void*)ptr, value, mc->size));

        // std::cout << "mapped memory to virtual address on device "<<std::to_string(gpuId)<<" at address "<<ptr<<std::endl;

        // // add to struct
        // // mc->deviceMem.push_back(ptr);
        // // mc->deviceVHandle.push_back(handle);
        // // mc->gpuIds.push_back(gpuId);

        // std::cout << "binding memory to multicast address..."<<std::endl;
        // // 5. bind virtual mem from device onto multicast object
        // CHECK_CUDA(cuMulticastBindMem(mcHandle, 0, handle, 0, size, 0));

        // std::cout << "bound to multicast!"<<std::endl;
        // /////////

        MCResource mc(size, mcHandle);

        return mc;
    }

    void bindDeviceMemToMulticast(MCResource* mc, int gpuId, CUmemAccessDesc accessDesc) {
        if(mc->deinit) {
            fprintf(stderr, "Cannot manipulate MCResource with released multicast descriptor");
            return;
        }

        CUdevice device;
        CHECK_CUDA(cuDeviceGet(&device, gpuId));
        CHECK_CUDA(cuMulticastAddDevice(mc->mcHandle, device));
        std::cout<<"device "<<gpuId<<" added to multicast list of devices"<<std::endl;

        CUdeviceptr ptr = 0;
        CUmemAllocationProp prop = {};
        size_t gran;
        CUmemGenericAllocationHandle handle;

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

        std::cout << "mapped memory to virtual address on device "<<std::to_string(gpuId)<<" at address "<<ptr<<std::endl;

        // add to struct
        mc->deviceMem.push_back(ptr);
        mc->deviceVHandle.push_back(handle);
        mc->gpuIds.push_back(gpuId);

        std::cout << "binding memory to multicast address..."<<std::endl;
        // 5. bind virtual mem from device onto multicast object
        CHECK_CUDA(cuMulticastBindMem(mc->mcHandle, 0, handle, 0, mc->size, 0));

        std::cout << "bound to multicast!"<<std::endl;
    }

    // creates a multicast address that is accessible (cannot use the handle directly, this will return a usable CUdeviceptr)
    void allocateMultimemAddress(MCResource* mc, CUmemAccessDesc accessDesc, int gpuId) {
        if(mc->deinit) {
            fprintf(stderr, "Cannot manipulate MCResource with released multicast descriptor");
            return;
        }

        if(mc->mcBuff) {
            fprintf(stderr, "multimem address is already initialized, cannot create more than one");
            return;
        }

        CUdeviceptr ptr = 0;
        size_t gran;
        CUmemAllocationProp prop;

        memset(&prop, 0, sizeof(prop));
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; // allocating pinned memory
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; // allocation on device
        prop.location.id = gpuId;

        CHECK_CUDA(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)); // flag: recommended or minimum granularity

        CHECK_CUDA(cuMemAddressReserve(&ptr, mc->size, gran, 0U, 0)); // 1. create a new virtual memory address
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
            CHECK_CUDA(cuMemsetD32(mc->deviceMem[i], value, mc->size / sizeof(int)));
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
        CHECK_CUDA(cuMemRelease(mc.mcHandle));
        mc.deinit = true;
    }
}
