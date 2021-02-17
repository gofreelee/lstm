use std::error::Error;
mod bindings;

type BlobType = bindings::BlobType;
type GPUType = bindings::GPUType;

#[derive(Debug)]
pub struct GPUInfo {
	gpu_type: GPUType,
	pci_device_id: u32,
	total_memory: u64,
	free_memory: u64,
	name: String,
}

//
// Minimal interfaces between MICA and the GPUs. All operations for each GPU run on a dedicated
// thread, so that it is possible to work with the NVIDIA driver.
//
// TODO: (1) Type different types of handles to enforce compile-time checks. (2) Implement safe versions of the memcpy APIs.
pub trait GPUShim {
	fn get_gpu_info(&self) -> GPUInfo;
	//
	// The NVIDIA implementation requires the caller thread to be initialized
	// before any operations.
	fn initialize(&self) -> Result<(), Box<dyn Error>>;
	//
	// Allocate Read-only blob
	// TODO: Add APIs for kernels?
	fn allocate_blob(&self, ty: BlobType, size: u64) -> u64;
	fn free_blob(&self, handle: u64);
	//
	// Allocate temporary buffer for a job
	fn allocate_bufffer(&self, ty: BlobType, size: u64) -> u64;
	fn free_buffer(&self, handle: u64);
	//
	// Allocate a hardware queue for QoS
	fn allocate_queue(&self, priority: u32) -> u64;
	fn free_queue(&self, handle: u64);
	//
	// Launching a kernel. The kernel blob might contain multiple kernels thus it requires an addtional
	// ID to identify the individual kernel to be launched.
	//
	// TODO: Specify the pre-condition for launching?
	fn launch_kernel(
		&self,
		queue: u64,
		blob: u64,
		id: u32,
		grid_size: u32,
		block_size: u32,
		args: &[u8],
	);
	fn queue_sync(&self, queue: u64);

	fn copy_to_device(&self, queue: u64, dst: u64, src: &[u8]);
	fn copy_from_device(&self, queue: u64, dst: &mut [u8], src: u64);
}

pub fn enumerate_gpu_devices() -> Vec<Box<dyn GPUShim>> {
	bindings::enumerate_gpu_devices()
}
