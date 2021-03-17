use crate::parser::diagnostics::ConsumeClosingDelim;
use crate::Parser;
use mica_compiler_ast::ast::P;
use mica_compiler_ast::ast::{
	self, Attribute, Function, FunctionParam, Item, ItemKind, Statement, StructField,
};
use mica_compiler_ast::node_id::DUMMY_NODE_ID;
use mica_compiler_ast::token::{self, DelimToken, TokenKind};
use mica_compiler_span::symbol::{kw, Ident};
use mica_compiler_span::Span;
use rustc_errors::{Applicability, PResult};

pub(super) type ItemInfo = (Ident, ItemKind);

impl<'a> Parser<'a> {
	/// Parses a source module as a crate. This is the main entry point for the parser.
	pub fn parse_crate_mod(&mut self) -> PResult<'a, ast::Crate> {
		let (items, span) = self.parse_mod(&token::TokenKind::Eof)?;
		Ok(ast::Crate { items, span })
	}
	/// Parses the contents of a module (inner attributes followed by module items).
	pub fn parse_mod(&mut self, term: &TokenKind) -> PResult<'a, (Vec<P<Item>>, Span)> {
		let lo = self.token.span;

		let mut items = vec![];
		while let Some(item) = self.parse_item()? {
			items.push(item);
		}

		if !self.eat(term) {
			let token_str = super::token_descr(&self.token);
			let msg = &format!("expected item, found {}", token_str);
			let mut err = self.struct_span_err(self.token.span, msg);
			err.span_label(self.token.span.to_rustc(), "expected item");
			return Err(err);
		}

		Ok((items, lo.to(self.prev_token.span)))
	}
}

impl<'a> Parser<'a> {
	pub fn parse_item(&mut self) -> PResult<'a, Option<P<Item>>> {
		self.parse_item_common().map(|i| i.map(P))
	}

	fn parse_item_common(&mut self) -> PResult<'a, Option<Item>> {
		let lo = self.token.span;
		let (kind, attributes) = self.parse_item_kind(lo)?;
		if let Some((ident, kind)) = kind {
			// self.error_on_unconsumed_default(def, &kind);
			let span = lo.to(self.prev_token.span);
			let id = DUMMY_NODE_ID;
			let item = Item {
				id,
				span,
				ident,
				kind,
				attributes,
			};
			return Ok(Some(item));
		}
		// At this point, we have failed to parse an item.
		Ok(None)
	}

	/// Parses one of the items allowed by the flags.
	fn parse_item_kind(&mut self, _lo: Span) -> PResult<'a, (Option<ItemInfo>, Vec<P<Attribute>>)> {
		let attributes;
		let empty = Vec::new();
		if self.token == TokenKind::Pound {
			//parse attribute
			self.bump();
			if self.eat(&TokenKind::OpenDelim(DelimToken::Bracket)) {
				attributes = self.parse_attributes()?;
				if self.token == TokenKind::CloseDelim(DelimToken::Bracket) {
					self.bump();
				}
			} else {
				attributes = empty;
			}
		} else {
			attributes = empty;
		}
		let info = if self.eat_keyword(kw::Struct) {
			// STRUCT ITEM
			self.parse_item_struct()?
		} else if self.eat_keyword(kw::Fn) {
			self.parse_item_function()?
		} else {
			return Ok((None, vec![]));
		};
		Ok((Some(info), attributes))
	}

	/// Parses `struct Foo { ... }`.
	fn parse_item_struct(&mut self) -> PResult<'a, ItemInfo> {
		let class_name = self.parse_ident()?;

		let vdata = if self.token == TokenKind::OpenDelim(DelimToken::Brace) {
			let (fields, _recovered) = self.parse_record_struct_body()?;
			fields
		} else {
			let token_str = super::token_descr(&self.token);
			let msg = &format!(
				"expected `where`, `{{`, `(`, or `;` after struct name, found {}",
				token_str
			);
			let mut err = self.struct_span_err(self.token.span, msg);
			err.span_label(self.token.span.to_rustc(), "expected `{` after struct name");
			return Err(err);
		};

		Ok((class_name, ItemKind::Struct(vdata)))
	}

	fn parse_record_struct_body(
		&mut self,
	) -> PResult<'a, (Vec<StructField>, /* recovered */ bool)> {
		let mut fields = Vec::new();
		let mut recovered = false;
		if self.eat(&TokenKind::OpenDelim(DelimToken::Brace)) {
			while self.token != TokenKind::CloseDelim(DelimToken::Brace) {
				let field = self.parse_struct_decl_field().map_err(|e| {
					self.consume_block(DelimToken::Brace, ConsumeClosingDelim::No);
					recovered = true;
					e
				});
				match field {
					Ok(field) => fields.push(field),
					Err(mut err) => {
						err.emit();
						break;
					}
				}
			}
			self.eat(&TokenKind::CloseDelim(DelimToken::Brace));
		} else {
			let token_str = super::token_descr(&self.token);
			let msg = &format!(
				"expected `where`, or `{{` after struct name, found {}",
				token_str
			);
			let mut err = self.struct_span_err(self.token.span, msg);
			err.span_label(
				self.token.span.to_rustc(),
				"expected `where`, or `{` after struct name",
			);
			return Err(err);
		}

		Ok((fields, recovered))
	}

	/// Parses an element of a struct declaration.
	fn parse_struct_decl_field(&mut self) -> PResult<'a, StructField> {
		let lo = self.token.span;
		Ok(self.parse_single_struct_field(lo)?)
	}

	/// Parses a structure field declaration.
	fn parse_single_struct_field(&mut self, lo: Span) -> PResult<'a, StructField> {
		let a_var = self.parse_name_and_ty(lo)?;
		match self.token.kind {
			TokenKind::Comma => {
				self.bump();
			}
			TokenKind::CloseDelim(DelimToken::Brace) => {}
			_ => {
				let sp = self.prev_token.span.shrink_to_hi();
				let mut err = self.struct_span_err(
					sp,
					&format!(
						"expected `,`, or `}}`, found {}",
						super::token_descr(&self.token)
					),
				);

				// Try to recover extra trailing angle brackets
				let mut recovered = false;
				if self.token.is_ident() {
					// This is likely another field; emit the diagnostic and keep going
					err.span_suggestion(
						sp.to_rustc(),
						"try adding a comma",
						",".into(),
						Applicability::MachineApplicable,
					);
					err.emit();
					recovered = true;
				}

				if recovered {
					// Make sure an error was emitted (either by recovering an angle bracket,
					// or by finding an identifier as the next token), since we're
					// going to continue parsing
					assert!(self.sess.span_diagnostic.has_errors());
				} else {
					return Err(err);
				}
			}
		}
		Ok(a_var)
	}

	/// Parses a structure field.
	fn parse_name_and_ty(&mut self, lo: Span) -> PResult<'a, StructField> {
		let name = self.parse_ident_common(false)?;
		self.expect(&TokenKind::Colon)?;
		let ty = self.parse_ty()?;
		Ok(StructField {
			span: lo.to(self.prev_token.span),
			ident: name,
			id: DUMMY_NODE_ID,
			ty,
		})
	}
}

impl<'a> Parser<'a> {
	// Parse 'fn Func() -> type {....}'
	pub fn parse_item_function(&mut self) -> PResult<'a, ItemInfo> {
		let func_name = self.parse_ident()?;
		let vdata = if self.token == TokenKind::OpenDelim(DelimToken::Paren) {
			let (params, _recovered) = self.parse_function_param()?;
			params
		} else {
			let token_str = super::token_descr(&self.token);
			let msg = &format!("expected '(' after function name, found{}", token_str);
			let mut err = self.struct_span_err(self.token.span, msg);
			err.span_label(self.token.span.to_rustc(), "expected '(' after struct name");
			return Err(err);
		};
		let function_body = if self.token == TokenKind::OpenDelim(DelimToken::Brace) {
			self.bump();
			let statements: Vec<Statement> = Vec::new();
			//  parse statements will be added by zhangyangyu
			self.eat(&TokenKind::CloseDelim(DelimToken::Brace));
			statements
		} else {
			vec![]
		};

		Ok((
			func_name,
			ItemKind::Fn(Function {
				func_name: func_name,
				params: vdata,
				function_body: function_body,
			}),
		))
	}

	fn parse_function_param(&mut self) -> PResult<'a, (Vec<FunctionParam>, bool)> {
		// let lo = self.token.span;
		// Ok(self.parse_single_function_param(lo)?)
		let mut params = Vec::new();
		let mut recovered = false;
		if self.eat(&TokenKind::OpenDelim(DelimToken::Paren)) {
			while self.token != TokenKind::CloseDelim(DelimToken::Paren) {
				let param = self
					.parse_single_function_param(self.token.span)
					.map_err(|e| {
						self.consume_block(DelimToken::Paren, ConsumeClosingDelim::No);
						recovered = true;
						e
					});
				match param {
					Ok(param) => params.push(param),
					Err(mut err) => {
						err.emit();
						break;
					}
				}
			}
			self.eat(&TokenKind::CloseDelim(DelimToken::Paren));
		} else {
			let token_str = super::token_descr(&self.token);
			let msg = &format!("expected '(' after function name, found{}", token_str);
			let mut err = self.struct_span_err(self.token.span, msg);
			err.span_label(
				self.token.span.to_rustc(),
				"expected '(' after function name, found{}",
			);
			return Err(err);
		}

		Ok((params, recovered))
	}

	fn parse_single_function_param(&mut self, lo: Span) -> PResult<'a, FunctionParam> {
		let a_var = self.parse_name_and_ty(lo)?;
		match self.token.kind {
			TokenKind::Comma => {
				self.bump();
			}
			TokenKind::CloseDelim(DelimToken::Paren) => {}
			_ => {
				let sp = self.prev_token.span.shrink_to_hi();
				let mut err = self.struct_span_err(
					sp,
					&format!(
						"expected `,`, or `)`, found {}",
						super::token_descr(&self.token)
					),
				);

				let mut recovered = false;
				if self.token.is_ident() {
					// This is likely another field; emit the diagnostic and keep going
					err.span_suggestion(
						sp.to_rustc(),
						"try adding a Paren",
						")".into(),
						Applicability::MachineApplicable,
					);
					err.emit();
					recovered = true;
				}

				if recovered {
					// Make sure an error was emitted (either by recovering an angle Paren,
					// or by finding an identifier as the next token), since we're
					// going to continue parsing
					assert!(self.sess.span_diagnostic.has_errors());
				} else {
					return Err(err);
				}
			}
		}
		Ok(a_var)
	}
}
