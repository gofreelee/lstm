#!/bin/sh

cargo fmt --all -- --check \
  && cargo +nightly build --release

# TODO: Run tests
