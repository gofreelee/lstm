use super::{StringReader, UnmatchedBrace};
use mica_compiler_ast::token::{Token, TokenKind};
use mica_compiler_ast::tokenstream::TokenStream;
use rustc_errors::PResult;

#[derive(Default)]
struct TokenStreamBuilder {
	buf: Vec<Token>,
}

impl TokenStreamBuilder {
	fn push(&mut self, tok: Token) {
		if let Some(prev_token) = self.buf.last() {
			if let Some(glued) = prev_token.glue(&tok) {
				self.buf.pop();
				self.buf.push(glued);
				return;
			}
		}
		self.buf.push(tok)
	}

	fn into_token_stream(self) -> TokenStream {
		TokenStream::new(self.buf)
	}
}

impl<'a> StringReader<'a> {
	pub(super) fn into_token_trees(&mut self) -> (PResult<'a, TokenStream>, Vec<UnmatchedBrace>) {
		let mut buf = TokenStreamBuilder::default();
		loop {
			let (_, tok) = self.next_token();
			if tok == TokenKind::Eof {
				return (Ok(buf.into_token_stream()), Vec::with_capacity(1));
			}
			buf.push(tok);
		}
	}
}
