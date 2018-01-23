#pragma once
#include <cub/util_allocator.cuh>
#include <moderngpu/context.hxx>

// This is a copy&paste clone of mgpu::standard_context.
// The only difference is that it's using a cub::CachingDeviceAllocator
// in alloc() and free() instead of straight cudaMalloc and cudaFree.

class caching_context_t : public mgpu::standard_context_t {
protected:
  static cub::CachingDeviceAllocator allocator;

public:
  caching_context_t(cudaStream_t stream_ = 0)
      : mgpu::standard_context_t(false, stream_) {}

  virtual void *alloc(size_t size, mgpu::memory_space_t space) override {
    void *p = nullptr;
    if (size) {
      cudaError_t result;
      if (mgpu::memory_space_device == space) {
        result = allocator.DeviceAllocate(&p, size, stream());
      } else {
        result = cudaMallocHost(&p, size);
      }
      if (cudaSuccess != result)
        throw mgpu::cuda_exception_t(result);
    }
    return p;
  }

  virtual void free(void *p, mgpu::memory_space_t space) override {
    if (p) {
      cudaError_t result;
      if (mgpu::memory_space_device == space) {
        result = allocator.DeviceFree(p);
      } else {
        result = cudaFreeHost(p);
      }
      if (cudaSuccess != result)
        throw mgpu::cuda_exception_t(result);
    }
  }
};
