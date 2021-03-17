#[kernel, launch_bounds(128,1,1)]
struct foo {
	a: int,
	b: int,
}
