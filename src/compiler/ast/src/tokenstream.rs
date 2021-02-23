use crate::token::Token;
use rustc_data_structures::sync::Lrc;
use std::iter;

pub use rustc_ast::tokenstream::Spacing;

#[derive(Clone, Debug)]
pub struct TokenStream(pub(crate) Lrc<Vec<Token>>);

impl iter::FromIterator<Token> for TokenStream {
	fn from_iter<I: IntoIterator<Item = Token>>(iter: I) -> Self {
		TokenStream::new(iter.into_iter().map(Into::into).collect::<Vec<Token>>())
	}
}

impl TokenStream {
	pub fn new(streams: Vec<Token>) -> TokenStream {
		TokenStream(Lrc::new(streams))
	}

	pub fn is_empty(&self) -> bool {
		self.0.is_empty()
	}

	pub fn len(&self) -> usize {
		self.0.len()
	}
}

#[derive(Clone)]
pub struct Cursor {
	pub stream: TokenStream,
	index: usize,
}

impl Iterator for Cursor {
	type Item = Token;

	fn next(&mut self) -> Option<Token> {
		self.next_with_spacing()
	}
}

impl Cursor {
	pub fn new(stream: TokenStream) -> Self {
		Cursor { stream, index: 0 }
	}

	pub fn next_with_spacing(&mut self) -> Option<Token> {
		if self.index < self.stream.len() {
			self.index += 1;
			let t = &self.stream.0[self.index - 1];
			Some(t.clone())
		} else {
			None
		}
	}

	pub fn look_ahead(&self, n: usize) -> Option<&Token> {
		self.stream.0[self.index..].get(n)
	}
}
