#!/bin/sh

check_clang_format() {
  GIT_DIFF="git clang-format --diff HEAD^ HEAD"
  diff=$(${GIT_DIFF})
  if [ "$?" -ne 0 ]; then
    echo "Failed to run git-clang-format"
    exit 1
  fi
  MISFORMATED_FILES=$(${GIT_DIFF}|egrep -- "^diff --git")
  if [ "$?" -eq 0 ]; then
    echo "The following files do not conform with the styling guide: ${MISFORMATED_FILES}"
    exit 1
  fi
}

check_experiments() {
  rm -rf target/experiments && mkdir -p target/experiments && \
  cd target/experiments && cmake -DCMAKE_BUILD_TYPE=Release -GNinja ../../experiments && ninja && ninja test
  if [ "$?" -ne 0 ]; then
    echo "Failed to test the experiments"
    exit 1
  fi
}

check_clang_format

export RUSTFLAGS="-Dwarnings ${RUSTFLAGS}"
cargo fmt --all -- --check \
  && cargo +nightly build --release \
  && cargo +nightly test

if [ "$?" -ne 0 ]; then
echo "Failed to pass cargo check/build/test"
exit 1
fi

check_experiments
