#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum BlobType {
    kKernel,
    kBuffer,
};

struct GPUInfo {
    enum GPUType {
        kAMD,
        kNVIDIA,
    };
    enum {
        kMaxNameSize = 256,
    };
    GPUType type;
    unsigned pci_device_id;
    uint64_t total_memory;
    uint64_t free_memory;
    char name[kMaxNameSize];
};

unsigned mica_dataplane_enumerate_gpu_device(void **out, size_t size);
void mica_dataplane_gpu_shim_get_gpu_info(void *shim, GPUInfo *out);
int mica_dataplane_gpu_shim_initialize(void *shim);
uintptr_t mica_dataplane_gpu_shim_allocate_blob(void *shim, BlobType type,
                                                size_t size);
void mica_dataplane_gpu_shim_free_blob(void *shim, uintptr_t blob);
uintptr_t mica_dataplane_gpu_shim_allocate_buffer(void *shim, BlobType type,
                                                  size_t size);
void mica_dataplane_gpu_shim_free_buffer(void *shim, uintptr_t blob);
uintptr_t mica_dataplane_gpu_shim_allocate_queue(void *shim, unsigned priority);
void mica_dataplane_gpu_shim_free_queue(void *shim, uintptr_t queue);
void mica_dataplane_gpu_shim_sync_queue(void *shim, uintptr_t queue);
void mica_dataplane_gpu_shim_launch_kernel(void *shim, uintptr_t queue,
                                           uintptr_t kernel_blob,
                                           unsigned kernel_id,
                                           unsigned grid_size,
                                           unsigned block_size,
                                           const char *args, size_t args_size);
void mica_dataplane_gpu_shim_copy_to_device(void *shim, uintptr_t queue,
                                            uintptr_t dst, const char *src,
                                            size_t size);
void mica_dataplane_gpu_shim_copy_from_device(void *shim, uintptr_t queue,
                                              char *dst, uintptr_t src,
                                              size_t size);

#ifdef __cplusplus
}
#endif