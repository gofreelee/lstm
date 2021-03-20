use crate::Parser;
use mica_compiler_ast::ast::P;
use mica_compiler_ast::ast::{AttrParam, Attribute, ParamContent};
use mica_compiler_ast::node_id::DUMMY_NODE_ID;
use mica_compiler_ast::token::{DelimToken, TokenKind};
use rustc_errors::PResult;

impl<'a> Parser<'a> {
	pub fn parse_attributes(&mut self) -> PResult<'a, Vec<P<Attribute>>> {
		let mut attributes = vec![];
		while let Some(attribute) = self.parse_attribute()? {
			attributes.push(attribute);
			if self.token == TokenKind::Comma {
				self.bump();
			}
		}
		// TO DO: deal errors

		Ok(attributes)
	}

	pub fn parse_attribute(&mut self) -> PResult<'a, Option<P<Attribute>>> {
		self.parse_attribute_common().map(|i| i.map(P))
	}

	pub fn parse_attribute_common(&mut self) -> PResult<'a, Option<Attribute>> {
		if self.token == TokenKind::CloseDelim(DelimToken::Bracket) {
			return Ok(None);
		}
		let attribute_name = self.parse_ident()?;
		let attrparams = if self.token == TokenKind::OpenDelim(DelimToken::Paren) {
			self.bump();
			let params = self.parse_attrparams()?;
			self.eat(&TokenKind::CloseDelim(DelimToken::Paren));
			params
		} else if self.token == TokenKind::Comma {
			self.eat(&TokenKind::Comma);
			vec![]
		} else {
			vec![]
		};

		Ok(Some(Attribute {
			ident: attribute_name,
			span: self.token.span,
			attr_params: attrparams,
			id: DUMMY_NODE_ID,
		}))
	}

	pub fn parse_attrparams(&mut self) -> PResult<'a, Vec<AttrParam>> {
		let mut attrparams = vec![];
		while self.token != TokenKind::CloseDelim(DelimToken::Paren) {
			let param = self.parse_attrparam()?;
			attrparams.push(param);
			if self.token == TokenKind::Comma {
				self.bump();
			}
		}
		Ok(attrparams)
	}

	pub fn parse_attrparam(&mut self) -> PResult<'a, AttrParam> {
		let content = match self.token.kind {
			TokenKind::Literal(literal_param) => {
				// let literal_param = TokenKind::Literal(self.token.kind);
				self.bump();
				ParamContent::Literal(literal_param)
			}
			TokenKind::Ident(_symbol_name, _flag) => {
				let ident_param = self.parse_ident()?;
				ParamContent::Ident(ident_param)
			}
			_ => ParamContent::None,
		};
		match content {
			ParamContent::None => {
				let mut err = self.struct_span_err(self.token.span, "msg");
				err.span_label(self.token.span.to_rustc(), "expected `{` after struct name");
				return Err(err);
			}
			_ => Ok(AttrParam {
				param_content: content,
				id: DUMMY_NODE_ID,
			}),
		}
	}
}
