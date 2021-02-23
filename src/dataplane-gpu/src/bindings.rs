use libc;
use std::ffi::CStr;
use std::mem::transmute;

#[repr(u32)]
#[derive(Debug)]
pub enum BlobType {
	Kernel,
	Buffer,
}

#[repr(u32)]
#[derive(Debug)]
#[allow(dead_code)]
pub enum GPUType {
	Unknown,
	AMD,
	NVIDIA,
}

#[repr(C)]
#[derive(Debug)]
struct GPUInfoInternal {
	gpu_type: GPUType,
	pci_device_id: u32,
	total_memory: u64,
	free_memory: u64,
	name: [i8; 256],
}

extern "C" {
	fn mica_dataplane_enumerate_gpu_device(out: *mut libc::uintptr_t, size: u64) -> u32;
	fn mica_dataplane_gpu_shim_get_gpu_info(shim: libc::uintptr_t, out: *mut GPUInfoInternal);
	fn mica_dataplane_gpu_shim_initialize(shim: libc::uintptr_t) -> u32;
	fn mica_dataplane_gpu_shim_allocate_blob(
		shim: libc::uintptr_t,
		blob_type: u32,
		size: u64,
	) -> u64;
	fn mica_dataplane_gpu_shim_free_blob(shim: libc::uintptr_t, blob: libc::uintptr_t);
	fn mica_dataplane_gpu_shim_allocate_buffer(
		shim: libc::uintptr_t,
		blob_type: u32,
		size: u64,
	) -> u64;
	fn mica_dataplane_gpu_shim_free_buffer(shim: libc::uintptr_t, blob: libc::uintptr_t);
	fn mica_dataplane_gpu_shim_allocate_queue(shim: libc::uintptr_t, priority: u32) -> u64;
	fn mica_dataplane_gpu_shim_free_queue(shim: libc::uintptr_t, queue: libc::uintptr_t);
	fn mica_dataplane_gpu_shim_sync_queue(shim: libc::uintptr_t, queue: libc::uintptr_t);
	fn mica_dataplane_gpu_shim_launch_kernel(
		shim: libc::uintptr_t,
		queue: libc::uintptr_t,
		kernel_blob: libc::uintptr_t,
		kernel_id: u32,
		grid_size: u32,
		block_size: u32,
		args: *const u8,
		args_size: usize,
	);
	fn mica_dataplane_gpu_shim_copy_to_device(
		shim: libc::uintptr_t,
		queue: libc::uintptr_t,
		dst: libc::uintptr_t,
		src: *const u8,
		size: usize,
	);
	fn mica_dataplane_gpu_shim_copy_from_device(
		shim: libc::uintptr_t,
		queue: libc::uintptr_t,
		dst: *mut u8,
		src: libc::uintptr_t,
		size: usize,
	);
}

struct GPUShimImpl {
	handle: libc::uintptr_t,
}

impl crate::GPUShim for GPUShimImpl {
	fn get_gpu_info(&self) -> crate::GPUInfo {
		let mut info = GPUInfoInternal {
			gpu_type: GPUType::Unknown,
			pci_device_id: 0,
			total_memory: 0,
			free_memory: 0,
			name: [0i8; 256],
		};
		unsafe {
			let ptr = &mut info;
			mica_dataplane_gpu_shim_get_gpu_info(self.handle, ptr);
			let c_str = CStr::from_ptr(&info.name as *const libc::c_char)
				.to_str()
				.expect("Invalid GPU name");
			return crate::GPUInfo {
				gpu_type: info.gpu_type,
				pci_device_id: info.pci_device_id,
				total_memory: info.total_memory,
				free_memory: info.free_memory,
				name: c_str.to_owned(),
			};
		}
	}
	fn initialize(&self) -> Result<(), Box<dyn std::error::Error>> {
		unsafe {
			let res = mica_dataplane_gpu_shim_initialize(self.handle);
			if res == 0 {
				return Ok(());
			} else {
				return Err(Box::new(std::io::Error::new(
					std::io::ErrorKind::Other,
					"Unknown error",
				)));
			}
		}
	}

	fn allocate_blob(&self, ty: BlobType, size: u64) -> u64 {
		unsafe {
			let t = transmute(ty as u32);
			return mica_dataplane_gpu_shim_allocate_blob(self.handle, t, size);
		}
	}
	fn free_blob(&self, handle: u64) {
		unsafe { mica_dataplane_gpu_shim_free_blob(self.handle, handle as libc::uintptr_t) }
	}
	fn allocate_bufffer(&self, ty: BlobType, size: u64) -> u64 {
		unsafe {
			let t = transmute(ty as u32);
			return mica_dataplane_gpu_shim_allocate_buffer(self.handle, t, size);
		}
	}
	fn free_buffer(&self, handle: u64) {
		unsafe { mica_dataplane_gpu_shim_free_buffer(self.handle, handle as libc::uintptr_t) }
	}
	fn allocate_queue(&self, priority: u32) -> u64 {
		unsafe { mica_dataplane_gpu_shim_allocate_queue(self.handle, priority) }
	}
	fn free_queue(&self, handle: u64) {
		unsafe { mica_dataplane_gpu_shim_free_queue(self.handle, handle as libc::uintptr_t) }
	}
	fn launch_kernel(
		&self,
		queue: u64,
		blob: u64,
		id: u32,
		grid_size: u32,
		block_size: u32,
		args: &[u8],
	) {
		unsafe {
			let q = queue as libc::uintptr_t;
			let b = blob as libc::uintptr_t;
			mica_dataplane_gpu_shim_launch_kernel(
				self.handle,
				q,
				b,
				id,
				grid_size,
				block_size,
				args.as_ptr(),
				args.len(),
			);
		}
	}
	fn queue_sync(&self, queue: u64) {
		unsafe { mica_dataplane_gpu_shim_sync_queue(self.handle, queue as libc::uintptr_t) }
	}

	fn copy_to_device(&self, queue: u64, dst: u64, src: &[u8]) {
		unsafe {
			mica_dataplane_gpu_shim_copy_to_device(
				self.handle,
				queue as libc::uintptr_t,
				dst as libc::uintptr_t,
				src.as_ptr(),
				src.len(),
			)
		}
	}
	fn copy_from_device(&self, queue: u64, dst: &mut [u8], src: u64) {
		unsafe {
			mica_dataplane_gpu_shim_copy_from_device(
				self.handle,
				queue as libc::uintptr_t,
				dst.as_mut_ptr(),
				src as libc::uintptr_t,
				dst.len(),
			)
		}
	}
}

pub(crate) fn enumerate_gpu_devices() -> Vec<Box<dyn crate::GPUShim>> {
	unsafe {
		let mut res = [libc::uintptr_t::default(); 64];
		let num_devices = mica_dataplane_enumerate_gpu_device(res.as_mut_ptr(), 64);
		let mut r = Vec::<Box<dyn crate::GPUShim>>::new();
		for x in 0..num_devices {
			r.push(Box::new(GPUShimImpl {
				handle: res[x as usize],
			}));
		}
		return r;
	}
}
