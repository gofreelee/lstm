#pragma once

#include "bindings.h"

namespace mica::dataplane {

class WorkQueue {
  public:
    virtual ~WorkQueue() = default;
    virtual void Sync() = 0;
};

class GPUShim {
  public:
    virtual ~GPUShim() = default;
    virtual void GetGPUInfo(GPUInfo *out) = 0;
    virtual bool Initialize();
    virtual uintptr_t AllocateBlob(BlobType type, size_t size) = 0;
    virtual void FreeBlob(uintptr_t blob) = 0;
    virtual uintptr_t AllocateBuffer(BlobType type, size_t size) = 0;
    virtual void FreeBuffer(uintptr_t blob) = 0;
    virtual WorkQueue *AllocateQueue(unsigned priority) = 0;
    virtual void FreeQueue(WorkQueue *queue) = 0;
    void SyncQueue(WorkQueue *queue) { queue->Sync(); }
    virtual void LaunchKernel(WorkQueue *queue, uintptr_t kernel_blob,
                              unsigned kernel_id, unsigned grid_size,
                              unsigned block_size, const char *args,
                              size_t args_size) = 0;
    virtual void CopyToDevice(WorkQueue *queue, uintptr_t dst, const char *src,
                              size_t size);
    virtual void CopyFromDevice(WorkQueue *queue, char *dst, uintptr_t src,
                                size_t size);
};

unsigned EnumerateGPUDevices(GPUShim *out, size_t size);
} // namespace mica::dataplane