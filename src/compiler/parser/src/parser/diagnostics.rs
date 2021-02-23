use crate::Parser;
use mica_compiler_ast::token::{self, DelimToken, TokenKind};
use mica_compiler_span::MultiSpan;
use rustc_errors::{struct_span_err, DiagnosticBuilder, Handler, PResult};

pub enum Error {
	UselessDocComment,
}

/// Control whether the closing delimiter should be consumed when calling `Parser::consume_block`.
#[allow(dead_code)]
pub(crate) enum ConsumeClosingDelim {
	Yes,
	No,
}

impl Error {
	fn span_err(self, sp: impl Into<MultiSpan>, handler: &Handler) -> DiagnosticBuilder<'_> {
		match self {
			Error::UselessDocComment => {
				let mut err = struct_span_err!(
					handler,
					sp,
					E0585,
					"found a documentation comment that doesn't document anything",
				);
				err.help(
					"doc comments must come before what they document, maybe a comment was \
                          intended with `//`?",
				);
				err
			}
		}
	}
}

impl<'a> Parser<'a> {
	pub(super) fn span_fatal_err<S: Into<MultiSpan>>(
		&self,
		sp: S,
		err: Error,
	) -> DiagnosticBuilder<'a> {
		err.span_err(sp, self.diagnostic())
	}

	pub fn struct_span_err<S: Into<MultiSpan>>(&self, sp: S, m: &str) -> DiagnosticBuilder<'a> {
		self.sess.span_diagnostic.struct_span_err(sp, m)
	}

	pub(super) fn diagnostic(&self) -> &'a Handler {
		&self.sess.span_diagnostic
	}

	pub(super) fn expected_ident_found(&self) -> DiagnosticBuilder<'a> {
		let mut err = self.struct_span_err(
			self.token.span,
			&format!(
				"expected identifier, found {}",
				super::token_descr(&self.token)
			),
		);
		if let Some(token_descr) = super::token_descr_opt(&self.token) {
			err.span_label(
				self.token.span.to_rustc(),
				format!("expected identifier, found {}", token_descr),
			);
		} else {
			err.span_label(self.token.span.to_rustc(), "expected identifier");
		}
		err
	}

	pub(super) fn expected_one_of_not_found(
		&mut self,
		_edible: &[TokenKind],
		_inedible: &[TokenKind],
	) -> PResult<'a, bool /* recovered */> {
		let mut err = self.struct_span_err(self.token.span, "Unexpected token");
		err.span_label(self.token.span.to_rustc(), "Unexpected token");
		Err(err)
	}

	pub(super) fn consume_block(
		&mut self,
		delim: token::DelimToken,
		consume_close: ConsumeClosingDelim,
	) {
		let mut brace_depth = 0;
		loop {
			if self.eat(&TokenKind::OpenDelim(delim)) {
				brace_depth += 1;
			} else if self.check(&TokenKind::CloseDelim(delim)) {
				if brace_depth == 0 {
					if let ConsumeClosingDelim::Yes = consume_close {
						// Some of the callers of this method expect to be able to parse the
						// closing delimiter themselves, so we leave it alone. Otherwise we advance
						// the parser.
						self.bump();
					}
					return;
				} else {
					self.bump();
					brace_depth -= 1;
					continue;
				}
			} else if self.token == TokenKind::Eof
				|| self.eat(&TokenKind::CloseDelim(DelimToken::NoDelim))
			{
				return;
			} else {
				self.bump();
			}
		}
	}
}
