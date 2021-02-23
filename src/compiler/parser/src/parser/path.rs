use crate::parser::Parser;
use mica_compiler_ast::ast::{Path, PathSegment};
use mica_compiler_ast::token::TokenKind;
use mica_compiler_span::symbol::Ident;
use rustc_errors::PResult;

/// Specifies how to parse a path.
#[derive(Copy, Clone, PartialEq)]
#[allow(dead_code)]
pub enum PathStyle {
	/// In some contexts, notably in expressions, paths with generic arguments are ambiguous
	/// with something else. For example, in expressions `segment < ....` can be interpreted
	/// as a comparison and `segment ( ....` can be interpreted as a function call.
	/// In all such contexts the non-path interpretation is preferred by default for practical
	/// reasons, but the path interpretation can be forced by the disambiguator `::`, e.g.
	/// `x<y>` - comparisons, `x::<y>` - unambiguously a path.
	Expr,
	/// In other contexts, notably in types, no ambiguity exists and paths can be written
	/// without the disambiguator, e.g., `x<y>` - unambiguously a path.
	/// Paths with disambiguators are still accepted, `x::<Y>` - unambiguously a path too.
	Type,
	/// A path with generic arguments disallowed, e.g., `foo::bar::Baz`, used in imports,
	/// visibilities or attributes.
	/// Technically, this variant is unnecessary and e.g., `Expr` can be used instead
	/// (paths in "mod" contexts have to be checked later for absence of generic arguments
	/// anyway, due to macros), but it is used to avoid weird suggestions about expected
	/// tokens when something goes wrong.
	Mod,
}

impl<'a> Parser<'a> {
	/// Parses simple paths.
	///
	/// `path = [::] segment+`
	/// `segment = ident | ident[::]<args> | ident[::](args) [-> type]`
	///
	/// # Examples
	/// `a::b::C<D>` (without disambiguator)
	/// `a::b::C::<D>` (with disambiguator)
	/// `Fn(Args)` (without disambiguator)
	/// `Fn::(Args)` (with disambiguator)
	pub(super) fn parse_path(&mut self, _style: PathStyle) -> PResult<'a, Path> {
		let lo = self.token.span;
		let mut segments = Vec::new();
		let mod_sep_ctxt = self.token.span.ctxt();
		if self.eat(&TokenKind::ModSep) {
			segments.push(PathSegment::path_root(
				lo.shrink_to_lo().with_ctxt(mod_sep_ctxt),
			));
		}
		self.parse_path_segments(&mut segments, _style)?;

		Ok(Path {
			segments,
			span: lo.to(self.prev_token.span),
		})
	}

	pub(super) fn parse_path_segments(
		&mut self,
		segments: &mut Vec<PathSegment>,
		_style: PathStyle,
	) -> PResult<'a, ()> {
		loop {
			let segment = self.parse_path_segment(_style)?;
			segments.push(segment);

			if !self.eat(&TokenKind::ModSep) {
				return Ok(());
			}
		}
	}

	pub(super) fn parse_path_segment(&mut self, _style: PathStyle) -> PResult<'a, PathSegment> {
		let ident = self.parse_path_segment_ident()?;
		Ok(
			// Generic arguments are not found.
			PathSegment::from_ident(ident),
		)
	}

	pub(super) fn parse_path_segment_ident(&mut self) -> PResult<'a, Ident> {
		self.parse_ident()
	}
}
