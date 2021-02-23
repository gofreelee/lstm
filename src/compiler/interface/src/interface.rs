use mica_compiler_ast::ast;
use mica_compiler_parser::parse_crate_from_file;
use mica_compiler_session::session;
use mica_compiler_span::{with_compatible_session_globals, with_session_globals};
use rustc_errors::PResult;
use std::path::Path;

pub fn compile_to_ast<R>(path: &Path, f: impl FnOnce(PResult<ast::Crate>) -> R) -> R {
	let parse_sess = session::build_parse_session();
	with_compatible_session_globals(|| {
		with_session_globals(|| {
			let ast = parse_crate_from_file(&path, &parse_sess);
			f(ast)
		})
	})
}
