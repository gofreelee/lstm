use clap::{App, Arg};
use mica_compiler_interface::compile_to_ast;
use std::path::Path;
mod test;

fn main() {
	let matches = App::new("The MICA compiler")
		.version("0.1.0")
		.arg(
			Arg::with_name("output")
				.short("o")
				.takes_value(true)
				.required(true)
				.help("The output of the compilation"),
		)
		.arg(
			Arg::with_name("arch")
				.long("arch")
				.takes_value(true)
				.default_value("cuda")
				.help("The target backend (CUDA / OpenCL"),
		)
		.arg(Arg::with_name("input").required(true).help("Input files"))
		.get_matches();

	let input = matches.value_of("input").unwrap();
	let path = Path::new(&input);
	compile_to_ast(path, |ast| match ast {
		Err(mut diag) => diag.emit(),
		Ok(_) => println!("The file passed"),
	});
}
