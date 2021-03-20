#[cfg(test)]
mod parser_test {
	#[test]
	fn test_parse_single_struct() {
		let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
		path.push("src/test/compiler/frontend/singlestruct.rs");
		assert!(mica_compiler_interface::compile_to_ast(
			&path.as_path(),
			|ast| match ast {
				Err(_diag) => return false,
				Ok(_) => return true,
			}
		));
	}
	#[test]

	fn test_parse_struct_with_attributes() {
		let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
		path.push("src/test/compiler/frontend/attributes_with_struct.rs");
		assert!(mica_compiler_interface::compile_to_ast(
			&path.as_path(),
			|ast| match ast {
				Err(_diag) => return false,
				Ok(_) => return true,
			}
		));
	}

	#[test]

	fn test_parse_function_without_body() {
		let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
		path.push("src/test/compiler/frontend/function_with_emptybody.rs");
		assert!(mica_compiler_interface::compile_to_ast(
			&path.as_path(),
			|ast| match ast {
				Err(_diag) => return false,
				Ok(_) => return true,
			}
		));
	}
}
