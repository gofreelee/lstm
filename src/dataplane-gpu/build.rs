use cmake;

fn main() {
	let dst = cmake::build("lib");

	println!("cargo:rustc-link-search=native={}", dst.display());
	println!("cargo:rustc-link-lib=static=dataplane_gpu_shim");
}
