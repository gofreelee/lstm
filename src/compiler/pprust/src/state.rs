use mica_compiler_ast::ast;
use mica_compiler_ast::token::{self, BinOpToken, CommentKind, DelimToken, LitKind, TokenKind};
use mica_compiler_span::symbol::{IdentPrinter, Symbol};

fn binop_to_string(op: BinOpToken) -> &'static str {
	match op {
		BinOpToken::Plus => "+",
		BinOpToken::Minus => "-",
		BinOpToken::Star => "*",
		BinOpToken::Slash => "/",
		BinOpToken::Percent => "%",
		BinOpToken::Caret => "^",
		BinOpToken::And => "&",
		BinOpToken::Or => "|",
		BinOpToken::Shl => "<<",
		BinOpToken::Shr => ">>",
	}
}

pub fn literal_to_string(lit: token::Lit) -> String {
	let token::Lit {
		kind,
		symbol,
		suffix,
	} = lit;
	let mut out = match kind {
		LitKind::Byte => format!("b'{}'", symbol),
		LitKind::Char => format!("'{}'", symbol),
		LitKind::Str => format!("\"{}\"", symbol),
		LitKind::StrRaw(n) => {
			format!(
				"r{delim}\"{string}\"{delim}",
				delim = "#".repeat(n as usize),
				string = symbol
			)
		}
		LitKind::ByteStr => format!("b\"{}\"", symbol),
		LitKind::ByteStrRaw(n) => {
			format!(
				"br{delim}\"{string}\"{delim}",
				delim = "#".repeat(n as usize),
				string = symbol
			)
		}
		LitKind::Integer | LitKind::Float | LitKind::Bool | LitKind::Err => symbol.to_string(),
	};

	if let Some(suffix) = suffix {
		out.push_str(&suffix.as_str())
	}

	out
}

fn doc_comment_to_string(
	comment_kind: CommentKind,
	attr_style: ast::AttrStyle,
	data: Symbol,
) -> String {
	match (comment_kind, attr_style) {
		(CommentKind::Line, ast::AttrStyle::Outer) => format!("///{}", data),
		(CommentKind::Line, ast::AttrStyle::Inner) => format!("//!{}", data),
		(CommentKind::Block, ast::AttrStyle::Outer) => format!("/**{}*/", data),
		(CommentKind::Block, ast::AttrStyle::Inner) => format!("/*!{}*/", data),
	}
}

fn token_kind_to_string_ext(tok: &TokenKind) -> String {
	match *tok {
		TokenKind::Eq => "=".to_string(),
		TokenKind::Lt => "<".to_string(),
		TokenKind::Le => "<=".to_string(),
		TokenKind::EqEq => "==".to_string(),
		TokenKind::Ne => "!=".to_string(),
		TokenKind::Ge => ">=".to_string(),
		TokenKind::Gt => ">".to_string(),
		TokenKind::Not => "!".to_string(),
		TokenKind::Pound => "#".to_string(),
		TokenKind::Tilde => "~".to_string(),
		TokenKind::OrOr => "||".to_string(),
		TokenKind::AndAnd => "&&".to_string(),
		TokenKind::BinOp(op) => binop_to_string(op).to_string(),
		TokenKind::BinOpEq(op) => format!("{}=", binop_to_string(op)),

		/* Structural symbols */
		TokenKind::Dot => ".".to_string(),
		TokenKind::Comma => ",".to_string(),
		TokenKind::Semi => ";".to_string(),
		TokenKind::Colon => ":".to_string(),
		TokenKind::ModSep => "::".to_string(),
		TokenKind::OpenDelim(DelimToken::Paren) => "(".to_string(),
		TokenKind::CloseDelim(DelimToken::Paren) => ")".to_string(),
		TokenKind::OpenDelim(DelimToken::Bracket) => "[".to_string(),
		TokenKind::CloseDelim(DelimToken::Bracket) => "]".to_string(),
		TokenKind::OpenDelim(DelimToken::Brace) => "{".to_string(),
		TokenKind::CloseDelim(DelimToken::Brace) => "}".to_string(),
		TokenKind::OpenDelim(DelimToken::NoDelim) | TokenKind::CloseDelim(DelimToken::NoDelim) => {
			"".to_string()
		}
		TokenKind::Question => "?".to_string(),

		/* Literals */
		TokenKind::Literal(lit) => literal_to_string(lit),

		/* Name components */
		TokenKind::Ident(s, is_raw) => IdentPrinter::new(s, is_raw).to_string(),

		/* Other */
		TokenKind::DocComment(comment_kind, attr_style, data) => {
			doc_comment_to_string(comment_kind, attr_style, data)
		}
		TokenKind::Eof => "<eof>".to_string(),
	}
}

/// Print the token precisely, without converting `$crate` into its respective crate name.
pub fn token_to_string(token: &token::Token) -> String {
	token_kind_to_string_ext(&token.kind)
}

/// Print the token kind precisely, without converting `$crate` into its respective crate name.
pub fn token_kind_to_string(tok: &TokenKind) -> String {
	token_kind_to_string_ext(tok)
}
