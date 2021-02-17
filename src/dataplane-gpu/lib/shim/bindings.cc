#include "bindings.h"
#include "gpu_shim.h"

using namespace mica::dataplane;

unsigned mica_dataplane_enumerate_gpu_device(GPUShim *out, size_t size) {
    return EnumerateGPUDevices(out, size);
}

void mica_dataplane_gpu_shim_get_gpu_info(void *shim, GPUInfo *out) {
    GPUShim *self = static_cast<GPUShim *>(shim);
    return self->GetGPUInfo(out);
}

int mica_dataplane_gpu_shim_initialize(void *shim) {
    GPUShim *self = static_cast<GPUShim *>(shim);
    return self->Initialize();
}

uintptr_t mica_dataplane_gpu_shim_allocate_blob(void *shim, BlobType type,
                                                size_t size) {
    GPUShim *self = static_cast<GPUShim *>(shim);
    return self->AllocateBlob(type, size);
}

void mica_dataplane_gpu_shim_free_blob(void *shim, uintptr_t blob) {
    GPUShim *self = static_cast<GPUShim *>(shim);
    return self->FreeBlob(blob);
}

uintptr_t mica_dataplane_gpu_shim_allocate_buffer(void *shim, BlobType type,
                                                  size_t size) {
    GPUShim *self = static_cast<GPUShim *>(shim);
    return self->AllocateBuffer(type, size);
}

void mica_dataplane_gpu_shim_free_buffer(void *shim, uintptr_t blob) {
    GPUShim *self = static_cast<GPUShim *>(shim);
    return self->FreeBuffer(blob);
}

uintptr_t mica_dataplane_gpu_shim_allocate_queue(void *shim,
                                                 unsigned priority) {
    GPUShim *self = static_cast<GPUShim *>(shim);
    return reinterpret_cast<uintptr_t>(self->AllocateQueue(priority));
}

void mica_dataplane_gpu_shim_free_queue(void *shim, uintptr_t queue) {
    GPUShim *self = static_cast<GPUShim *>(shim);
    return self->FreeQueue(reinterpret_cast<WorkQueue *>(queue));
}

void mica_dataplane_gpu_shim_sync_queue(void *shim, uintptr_t queue) {
    GPUShim *self = static_cast<GPUShim *>(shim);
    return self->SyncQueue(reinterpret_cast<WorkQueue *>(queue));
}

void mica_dataplane_gpu_shim_launch_kernel(void *shim, uintptr_t queue,
                                           uintptr_t kernel_blob,
                                           unsigned kernel_id,
                                           unsigned grid_size,
                                           unsigned block_size,
                                           const char *args, size_t args_size) {
    GPUShim *self = static_cast<GPUShim *>(shim);
    WorkQueue *q = reinterpret_cast<WorkQueue *>(queue);
    return self->LaunchKernel(q, kernel_blob, kernel_id, grid_size, block_size,
                              args, args_size);
}

void mica_dataplane_gpu_shim_copy_to_device(void *shim, uintptr_t queue,
                                            uintptr_t dst, const char *src,
                                            size_t size) {
    GPUShim *self = static_cast<GPUShim *>(shim);
    WorkQueue *q = reinterpret_cast<WorkQueue *>(queue);
    return self->CopyToDevice(q, dst, src, size);
}

void mica_dataplane_gpu_shim_copy_from_device(void *shim, uintptr_t queue,
                                              char *dst, uintptr_t src,
                                              size_t size) {
    GPUShim *self = static_cast<GPUShim *>(shim);
    WorkQueue *q = reinterpret_cast<WorkQueue *>(queue);
    return self->CopyFromDevice(q, dst, src, size);
}