mod diagnostics;
mod item;
mod path;
mod ty;

use diagnostics::Error;
use mica_compiler_ast::token::{Token, TokenKind};
use mica_compiler_ast::tokenstream::{Cursor, TokenStream};
use mica_compiler_pprust as pprust;
use mica_compiler_session::ParseSess;
use mica_compiler_span::symbol::{Ident, Symbol};
use mica_compiler_span::DUMMY_SP;
use rustc_errors::PResult;
use std::{mem, slice};

pub struct Parser<'a> {
	pub sess: &'a ParseSess,
	pub prev_token: Token,
	pub token: Token,
	pub token_cursor: Cursor,
	expected_tokens: Vec<TokenType>,
}

impl<'a> Parser<'a> {
	pub fn new(sess: &'a ParseSess, tokens: TokenStream) -> Self {
		let mut parser = Parser {
			sess,
			prev_token: Token::dummy(),
			token: Token::dummy(),
			token_cursor: Cursor::new(tokens),
			expected_tokens: Vec::new(),
		};

		// Make parser point to the first token.
		parser.bump();

		parser
	}

	fn next_tok(&mut self) -> Token {
		let next = self.token_cursor.next();
		next.unwrap_or_else(|| Token::new(TokenKind::Eof, DUMMY_SP))
	}

	/// Expects and consumes the token `t`. Signals an error if the next token is not `t`.
	pub fn expect(&mut self, t: &TokenKind) -> PResult<'a, bool /* recovered */> {
		if self.expected_tokens.is_empty() {
			if self.token == *t {
				self.bump();
				Ok(false)
			} else {
				let mut err = self.struct_span_err(self.token.span, "Unexpected token");
				err.span_label(self.token.span.to_rustc(), "Unexpected token");
				Err(err)
			}
		} else {
			self.expect_one_of(slice::from_ref(t), &[])
		}
	}

	/// Expect next token to be edible or inedible token.  If edible,
	/// then consume it; if inedible, then return without consuming
	/// anything.  Signal a fatal error if next token is unexpected.
	pub fn expect_one_of(
		&mut self,
		edible: &[TokenKind],
		inedible: &[TokenKind],
	) -> PResult<'a, bool /* recovered */> {
		if edible.contains(&self.token.kind) {
			self.bump();
			Ok(false)
		} else if inedible.contains(&self.token.kind) {
			// leave it in the input
			Ok(false)
		} else {
			self.expected_one_of_not_found(edible, inedible)
		}
	}

	fn check_or_expected(&mut self, ok: bool, typ: TokenType) -> bool {
		if ok {
			true
		} else {
			self.expected_tokens.push(typ);
			false
		}
	}

	#[allow(dead_code)]
	fn check_ident(&mut self) -> bool {
		self.check_or_expected(self.token.is_ident(), TokenType::Ident)
	}

	fn check_path(&mut self) -> bool {
		self.check_or_expected(self.token.is_path_start(), TokenType::Path)
	}

	/// Advance the parser by one token using provided token as the next one.
	fn bump_with(&mut self, next_token: Token) {
		// // Bumping after EOF is a bad sign, usually an infinite loop.
		// if self.prev_token.kind == TokenKind::Eof {
		// 	let msg = "attempted to bump the parser past EOF (may be stuck in a loop)";
		// // self.span_bug(self.token.span, msg);
		// }

		// Update the current and previous tokens.
		self.prev_token = mem::replace(&mut self.token, next_token);

		// Diagnostics.
		self.expected_tokens.clear();
	}

	/// Advance the parser by one token.
	pub fn bump(&mut self) {
		let next_token = self.next_tok();
		self.bump_with(next_token);
	}

	/// Look-ahead `dist` tokens of `self.token` and get access to that token there.
	/// When `dist == 0` then the current token is looked at.
	pub fn look_ahead<R>(&self, dist: usize, looker: impl FnOnce(&Token) -> R) -> R {
		if dist == 0 {
			return looker(&self.token);
		}

		match self.token_cursor.look_ahead(dist - 1) {
			Some(token) => looker(token),
			None => looker(&Token::dummy()),
		}
	}

	/// Checks if the next token is `tok`, and returns `true` if so.
	///
	/// This method will automatically add `tok` to `expected_tokens` if `tok` is not
	/// encountered.
	fn check(&mut self, tok: &TokenKind) -> bool {
		let is_present = self.token == *tok;
		if !is_present {
			self.expected_tokens.push(TokenType::Token(tok.clone()));
		}
		is_present
	}

	/// Consumes a token 'tok' if it exists. Returns whether the given token was present.
	pub fn eat(&mut self, tok: &TokenKind) -> bool {
		let is_present = self.check(tok);
		if is_present {
			self.bump()
		}
		is_present
	}

	/// If the next token is the given keyword, returns `true` without eating it.
	/// An expectation is also added for diagnostics purposes.
	fn check_keyword(&mut self, kw: Symbol) -> bool {
		self.expected_tokens.push(TokenType::Keyword(kw));
		self.token.is_keyword(kw)
	}

	/// If the next token is the given keyword, eats it and returns `true`.
	/// Otherwise, returns `false`. An expectation is also added for diagnostics purposes.
	// Public for rustfmt usage.
	pub fn eat_keyword(&mut self, kw: Symbol) -> bool {
		if self.check_keyword(kw) {
			self.bump();
			true
		} else {
			false
		}
	}

	// Public for rustfmt usage.
	pub fn parse_ident(&mut self) -> PResult<'a, Ident> {
		self.parse_ident_common(true)
	}

	fn parse_ident_common(&mut self, recover: bool) -> PResult<'a, Ident> {
		match self.token.ident() {
			Some((ident, is_raw)) => {
				if !is_raw && ident.is_reserved() {
					let mut err = self.expected_ident_found();
					if recover {
						err.emit();
					} else {
						return Err(err);
					}
				}
				self.bump();
				Ok(ident)
			}
			_ => Err(match self.prev_token.kind {
				TokenKind::DocComment(..) => {
					self.span_fatal_err(self.prev_token.span, Error::UselessDocComment)
				}
				_ => self.expected_ident_found(),
			}),
		}
	}
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
enum TokenType {
	Token(TokenKind),
	Keyword(Symbol),
	Operator,
	Ident,
	Path,
	Type,
	Const,
}

impl TokenType {
	#[allow(dead_code)]
	fn to_string(&self) -> String {
		match *self {
			TokenType::Token(ref t) => format!("`{}`", pprust::token_kind_to_string(t)),
			TokenType::Keyword(kw) => format!("`{}`", kw),
			TokenType::Operator => "an operator".to_string(),
			TokenType::Ident => "identifier".to_string(),
			TokenType::Path => "path".to_string(),
			TokenType::Type => "type".to_string(),
			TokenType::Const => "a const expression".to_string(),
		}
	}
}

fn token_descr_opt(token: &Token) -> Option<&'static str> {
	Some(match token.kind {
		_ if token.is_special_ident() => "reserved identifier",
		_ if token.is_used_keyword() => "keyword",
		_ if token.is_unused_keyword() => "reserved keyword",
		TokenKind::DocComment(..) => "doc comment",
		_ => return None,
	})
}

pub(super) fn token_descr(token: &Token) -> String {
	let token_str = pprust::token_to_string(token);
	match token_descr_opt(token) {
		Some(prefix) => format!("{} `{}`", prefix, token_str),
		_ => format!("`{}`", token_str),
	}
}
