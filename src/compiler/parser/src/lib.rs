use mica_compiler_ast::ast;
use mica_compiler_session::ParseSess;
use mica_compiler_span::SourceFile;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{Diagnostic, FatalError, Level, PResult};
use std::path::Path;

mod lexer;
mod parser;
use mica_compiler_ast::tokenstream::TokenStream;
pub use parser::Parser;

/// A variant of 'panictry!' that works on a Vec<Diagnostic> instead of a single DiagnosticBuilder.
macro_rules! panictry_buffer {
	($handler:expr, $e:expr) => {{
		use rustc_errors::FatalError;
		use std::result::Result::{Err, Ok};
		match $e {
			Ok(e) => e,
			Err(errs) => {
				for e in errs {
					$handler.emit_diagnostic(&e);
				}
				FatalError.raise()
			}
		}
	}};
}

pub fn parse_crate_from_file<'a>(input: &Path, sess: &'a ParseSess) -> PResult<'a, ast::Crate> {
	let mut parser = new_parser_from_file(sess, input);
	parser.parse_crate_mod()
}

pub fn new_parser_from_file<'a>(sess: &'a ParseSess, path: &Path) -> Parser<'a> {
	source_file_to_parser(sess, file_to_source_file(sess, path))
}

/// Given a session and a path and an optional span (for error reporting),
/// add the path to the session's source_map and return the new source_file or
/// error when a file can't be read.
fn try_file_to_source_file(sess: &ParseSess, path: &Path) -> Result<Lrc<SourceFile>, Diagnostic> {
	sess.source_map().load_file(path).map_err(|e| {
		let msg = format!("couldn't read {}: {}", path.display(), e);
		let diag = Diagnostic::new(Level::Fatal, &msg);
		diag
	})
}

/// Given a session and a path and an optional span (for error reporting),
/// adds the path to the session's `source_map` and returns the new `source_file`.
fn file_to_source_file(sess: &ParseSess, path: &Path) -> Lrc<SourceFile> {
	match try_file_to_source_file(sess, path) {
		Ok(source_file) => source_file,
		Err(d) => {
			sess.span_diagnostic.emit_diagnostic(&d);
			FatalError.raise();
		}
	}
}

/// Given a `source_file` and config, returns a parser.
fn source_file_to_parser(sess: &ParseSess, source_file: Lrc<SourceFile>) -> Parser<'_> {
	panictry_buffer!(
		&sess.span_diagnostic,
		maybe_source_file_to_parser(sess, source_file)
	)
}

/// Given a `source_file` and config, return a parser. Returns any buffered errors from lexing the
/// initial token stream.
fn maybe_source_file_to_parser(
	sess: &ParseSess,
	source_file: Lrc<SourceFile>,
) -> Result<Parser<'_>, Vec<Diagnostic>> {
	let stream = maybe_file_to_stream(sess, source_file)?;
	let parser = stream_to_parser(sess, stream);

	Ok(parser)
}

/// Given a `source_file`, produces a sequence of token trees.
pub fn source_file_to_stream(sess: &ParseSess, source_file: Lrc<SourceFile>) -> TokenStream {
	panictry_buffer!(
		&sess.span_diagnostic,
		maybe_file_to_stream(sess, source_file)
	)
}

/// Given a source file, produces a sequence of token trees. Returns any buffered errors from
/// parsing the token stream.
pub fn maybe_file_to_stream(
	sess: &ParseSess,
	source_file: Lrc<SourceFile>,
) -> Result<TokenStream, Vec<Diagnostic>> {
	let src = source_file.src.as_ref().unwrap_or_else(|| {
		sess.span_diagnostic.bug(&format!(
			"cannot lex `source_file` without source: {}",
			source_file.name
		));
	});

	let (token_trees, _unmatched_braces) =
		lexer::parse_token_trees(sess, src.as_str(), source_file.start_pos, None);

	match token_trees {
		Ok(stream) => Ok(stream),
		Err(err) => {
			let mut buffer = Vec::with_capacity(1);
			err.buffer(&mut buffer);
			Err(buffer)
		}
	}
}

/// Given a stream and the `ParseSess`, produces a parser.
pub fn stream_to_parser<'a>(sess: &'a ParseSess, stream: TokenStream) -> Parser<'a> {
	Parser::new(sess, stream)
}
