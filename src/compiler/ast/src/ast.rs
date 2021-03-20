use crate::token::Lit;
use mica_compiler_span::symbol::kw;
use mica_compiler_span::{symbol::Ident, Span};
pub use rustc_ast::node_id::DUMMY_NODE_ID;
pub use rustc_ast::ptr::P;
pub use rustc_ast::{node_id::NodeId, AttrStyle};

pub type FunctionParam = StructField;
/// The various kinds of type recognized by the compiler.
#[derive(Clone, Debug)]
pub enum TyKind {
	Path(Path),
}

#[derive(Clone, Debug)]
pub struct Ty {
	pub id: NodeId,
	pub span: Span,
	pub kind: TyKind,
}

/// Field of a struct.
///
/// E.g., `bar: usize` as in `struct Foo { bar: usize }`.
#[derive(Clone, Debug)]
pub struct StructField {
	pub span: Span,
	pub id: NodeId,
	pub ident: Ident,
	pub ty: P<Ty>,
}

#[derive(Clone, Debug)]
pub enum ItemKind {
	Fn(Function),
	Enum,
	Struct(Vec<StructField>),
}

#[derive(Clone, Debug)]
pub struct Crate {
	pub items: Vec<P<Item>>,
	pub span: Span,
}

/// An item definition.
#[derive(Clone, Debug)]
pub struct Item<K = ItemKind> {
	pub id: NodeId,
	pub span: Span,
	/// The name of the item.
	/// It might be a dummy name in case of anonymous items.
	pub ident: Ident,
	pub kind: K,
	pub attributes: Vec<P<Attribute>>,
}

/// A "Path" is essentially Rust's notion of a name.
///
/// It's represented as a sequence of identifiers,
/// along with a bunch of supporting information.
///
/// E.g., `std::cmp::PartialEq`.
#[derive(Clone, Debug)]
pub struct Path {
	pub span: Span,
	/// The segments in the path: the things separated by `::`.
	/// Global paths begin with `kw::PathRoot`.
	pub segments: Vec<PathSegment>,
}

impl Path {
	// Convert a span and an identifier to the corresponding
	// one-segment path.
	pub fn from_ident(ident: Ident) -> Path {
		Path {
			segments: vec![PathSegment::from_ident(ident)],
			span: ident.span,
		}
	}
}

/// A segment of a path: an identifier, an optional lifetime, and a set of types.
///
/// E.g., `std`, `String` or `Box<T>`.
#[derive(Clone, Debug)]
pub struct PathSegment {
	/// The identifier portion of this path segment.
	pub ident: Ident,
	pub id: NodeId,
}

impl PathSegment {
	pub fn from_ident(ident: Ident) -> Self {
		PathSegment {
			ident,
			id: DUMMY_NODE_ID,
		}
	}
	pub fn path_root(span: Span) -> Self {
		PathSegment::from_ident(Ident::new(kw::PathRoot, span))
	}
}

/// will be used to prestent
#[derive(Clone, Debug)]
pub struct Attribute {
	pub ident: Ident,
	pub span: Span,
	pub attr_params: Vec<AttrParam>,
	pub id: NodeId,
}

#[derive(Clone, Debug)]
pub struct AttrParam {
	pub param_content: ParamContent,
	pub id: NodeId,
}

#[derive(Clone, Debug)]
pub enum ParamContent {
	Ident(Ident),
	Literal(Lit),
	None,
}

#[derive(Clone, Debug)]
pub struct Function {
	pub func_name: Ident,
	pub params: Vec<FunctionParam>,
	pub function_body: Vec<Statement>,
}

#[derive(Clone, Debug)]
pub struct Statement {}
