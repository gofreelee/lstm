#[inline]
fn inline1() {}

#[kernel, launch_bounds(128,1,1)]
fn kern() {
    inline1();
}