#!/bin/sh

export RUSTFLAGS="-Dwarnings ${RUSTFLAGS}"
cargo fmt --all -- --check \
  && cargo +nightly build --release \
  && cargo +nightly test

if [ "$?" -ne 0 ]; then
echo "Failed to pass cargo check/build/test"
exit 1
fi
