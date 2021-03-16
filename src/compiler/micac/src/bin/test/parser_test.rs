use clap::{App, Arg};
use mica_compiler_interface::compile_to_ast;
use std::path::Path;
#[cfg(test)]
mod parser_test {
	#[test]
	fn test_parse_single_struct() {
		let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
		path.push("src/test/compiler/frontend/singlestruct.rs");
		assert!(mica_compiler_interface::compile_to_ast(
			&path.as_path(),
			|ast| match ast {
				Err(mut diag) => return false,
				Ok(_) => return true,
			}
		));
	}
}
