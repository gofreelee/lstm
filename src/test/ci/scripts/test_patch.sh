#!/bin/sh

cargo fmt --all -- --check \
  && cargo +nightly build --release \
  && cargo +nightly test

# TODO: Run tests
