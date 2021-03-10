use crate::ast;
use mica_compiler_span::symbol::{Ident, Symbol};
use mica_compiler_span::{Span, DUMMY_SP};
use std::fmt;

pub use rustc_ast::token::{BinOpToken, CommentKind, DelimToken, LitKind};

/// A literal token.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Lit {
	pub kind: LitKind,
	pub symbol: Symbol,
	pub suffix: Option<Symbol>,
}

impl fmt::Display for Lit {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		use LitKind::*;
		let Lit {
			kind,
			symbol,
			suffix,
		} = *self;
		match kind {
			Byte => write!(f, "b'{}'", symbol)?,
			Char => write!(f, "'{}'", symbol)?,
			Str => write!(f, "\"{}\"", symbol)?,
			StrRaw(n) => write!(
				f,
				"r{delim}\"{string}\"{delim}",
				delim = "#".repeat(n as usize),
				string = symbol
			)?,
			ByteStr => write!(f, "b\"{}\"", symbol)?,
			ByteStrRaw(n) => write!(
				f,
				"br{delim}\"{string}\"{delim}",
				delim = "#".repeat(n as usize),
				string = symbol
			)?,
			Integer | Float | Bool | Err => write!(f, "{}", symbol)?,
		}

		if let Some(suffix) = suffix {
			write!(f, "{}", suffix)?;
		}

		Ok(())
	}
}

#[derive(Clone, PartialEq, Debug)]
pub enum TokenKind {
	/* Expression-operator symbols. */
	Eq,
	Lt,
	Le,
	EqEq,
	Ne,
	Ge,
	Pound,
	Gt,
	AndAnd,
	OrOr,
	Not,
	Tilde,
	BinOp(BinOpToken),
	BinOpEq(BinOpToken),

	/* Structural symbols */
	Dot,
	Comma,
	Semi,
	Colon,
	Question,
	ModSep,
	/// An opening delimiter (e.g., `{`).
	OpenDelim(DelimToken),
	/// A closing delimiter (e.g., `}`).
	CloseDelim(DelimToken),

	/* Literals */
	Literal(Lit),

	/// Identifier token.
	/// Do not forget about `NtIdent` when you want to match on identifiers.
	/// It's recommended to use `Token::(ident,uninterpolate,uninterpolated_span)` to
	/// treat regular and interpolated identifiers in the same way.
	Ident(Symbol, /* is_raw */ bool),

	/// A doc comment token.
	/// `Symbol` is the doc comment's data excluding its "quotes" (`///`, `/**`, etc)
	/// similarly to symbols in string literal tokens.
	DocComment(CommentKind, ast::AttrStyle, Symbol),

	Eof,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Token {
	pub kind: TokenKind,
	pub span: Span,
}

impl Token {
	pub fn new(kind: TokenKind, span: Span) -> Self {
		Token { kind, span }
	}

	/// Some token that will be thrown away later.
	pub fn dummy() -> Self {
		Token::new(TokenKind::Question, DUMMY_SP)
	}

	pub fn is_path_start(&self) -> bool {
		self == &TokenKind::ModSep || self.is_ident()
	}

	/// Returns `true` if the token is a given keyword, `kw`.
	pub fn is_keyword(&self, kw: Symbol) -> bool {
		self.is_non_raw_ident_where(|id| id.name == kw)
	}

	// Returns true for reserved identifiers used internally for elided lifetimes,
	// unnamed method parameters, crate root module, error recovery etc.
	pub fn is_special_ident(&self) -> bool {
		false
	}

	/// Returns `true` if the token is a keyword used in the language.
	pub fn is_used_keyword(&self) -> bool {
		self.is_non_raw_ident_where(Ident::is_used_keyword)
	}

	/// Returns `true` if the token is a keyword reserved for possible future use.
	pub fn is_unused_keyword(&self) -> bool {
		false
	}

	/// Returns an identifier if this token is an identifier.
	pub fn ident(&self) -> Option<(Ident, /* is_raw */ bool)> {
		match self.kind {
			TokenKind::Ident(name, is_raw) => Some((Ident::new(name, self.span), is_raw)),
			_ => None,
		}
	}

	/// Returns `true` if the token is an identifier.
	pub fn is_ident(&self) -> bool {
		self.ident().is_some()
	}

	/// Returns `true` if the token is a non-raw identifier for which `pred` holds.
	pub fn is_non_raw_ident_where(&self, pred: impl FnOnce(Ident) -> bool) -> bool {
		match self.ident() {
			Some((id, false)) => pred(id),
			_ => false,
		}
	}

	pub fn glue(&self, joint: &Token) -> Option<Token> {
		use BinOpToken::*;
		use TokenKind::*;
		let kind = match self.kind {
			Eq => match joint.kind {
				Eq => EqEq,
				_ => return None,
			},
			Lt => match joint.kind {
				Eq => Le,
				Lt => BinOp(Shl),
				Le => BinOpEq(Shl),
				_ => return None,
			},
			Gt => match joint.kind {
				Eq => Ge,
				Gt => BinOp(Shr),
				Ge => BinOpEq(Shr),
				_ => return None,
			},
			Not => match joint.kind {
				Eq => Ne,
				_ => return None,
			},
			BinOp(op) => match joint.kind {
				Eq => BinOpEq(op),
				BinOp(And) if op == And => AndAnd,
				BinOp(Or) if op == Or => OrOr,
				_ => return None,
			},
			Colon => match joint.kind {
				Colon => ModSep,
				_ => return None,
			},

			Dot | Le | EqEq | Ne | Ge | AndAnd | OrOr | Tilde | BinOpEq(..) | Comma | Semi
			| ModSep | Pound | Question | OpenDelim(..) | CloseDelim(..) | Literal(..)
			| Ident(..) | DocComment(..) | Eof => return None,
		};

		Some(Token::new(kind, self.span.to(joint.span)))
	}
}

impl PartialEq<TokenKind> for Token {
	fn eq(&self, rhs: &TokenKind) -> bool {
		self.kind == *rhs
	}
}
