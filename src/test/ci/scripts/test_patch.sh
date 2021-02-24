#!/bin/sh

cargo fmt --all --check \
  && cargo build --release

# TODO: Run tests
