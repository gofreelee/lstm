use crate::parser::path::PathStyle;
use crate::parser::Parser;
use mica_compiler_ast::ast::{Ty, TyKind, P};
use mica_compiler_ast::node_id::DUMMY_NODE_ID;
use mica_compiler_span::Span;
use rustc_errors::PResult;

impl<'a> Parser<'a> {
	/// Parses a type.
	pub fn parse_ty(&mut self) -> PResult<'a, P<Ty>> {
		self.parse_ty_common()
	}

	/// Parses a type starting with a path.
	fn parse_path_start_ty(&mut self) -> PResult<'a, TyKind> {
		// Simple path
		let path = self.parse_path(PathStyle::Type)?;
		Ok(TyKind::Path(path))
	}

	fn parse_ty_common(&mut self) -> PResult<'a, P<Ty>> {
		let lo = self.token.span;
		let kind = if self.check_path() {
			self.parse_path_start_ty()?
		} else {
			let msg = format!("expected type, found {}", super::token_descr(&self.token));
			let mut err = self.struct_span_err(self.token.span.to_rustc(), &msg);
			err.span_label(self.token.span.to_rustc(), "expected type");
			return Err(err);
		};

		let span = lo.to(self.prev_token.span);
		let ty = self.mk_ty(span, kind);
		Ok(ty)
	}

	pub(super) fn mk_ty(&self, span: Span, kind: TyKind) -> P<Ty> {
		P(Ty {
			kind,
			span,
			id: DUMMY_NODE_ID,
		})
	}
}
