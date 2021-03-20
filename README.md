# MICA

MICA is a high performance, serverless framework for GPU workloads.

The key to MICA's performance is to enforce workloads running on the clusters cooperatively. MICA uses a language-based approach to ensure cooperative executions. All workloads running on MICA need to be written in a domain-specific language. The language constructs ensure that all workloads are memory-safe and will eventually terminate.

## Building MICA

MICA is written in Rust. You can build it with [cargo](https://doc.rust-lang.org/cargo) and the nightly version of Rust. You can use [Rustup](https://rustup.rs) to set up your building enviornments.

To build MICA, simply type the following command:

```
cargo +nightly build --release
```

## Coding styles

The project is implemented in both Rust and C++. Please follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) when implementing features in C++. 
