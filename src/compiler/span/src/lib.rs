#![feature(const_panic)]
#![feature(negative_impls)]

mod span_encoding;
pub mod symbol;
use rustc_data_structures::sync::Lock;
use rustc_data_structures::AtomicRef;
use rustc_span::hygiene::SyntaxContext;
pub use rustc_span::{BytePos, FileName, MultiSpan, Pos, SourceFile};
pub use span_encoding::{Span, DUMMY_SP};
use std::cmp;
use std::fmt;

pub mod source_map {
	pub use rustc_span::source_map::{FilePathMapping, SourceMap};
}
pub mod hygiene {
	pub use rustc_span::hygiene::HygieneData;
}

// Per-session global variables: this struct is stored in thread-local storage
// in such a way that it is accessible without any kind of handle to all
// threads within the compilation session, but is not accessible outside the
// session.
pub struct SessionGlobals {
	symbol_interner: Lock<symbol::Interner>,
	span_interner: Lock<span_encoding::SpanInterner>,
	// hygiene_data: Lock<hygiene::HygieneData>,
	// source_map: Lock<Option<Lrc<SourceMap>>>,
}

impl SessionGlobals {
	pub fn new() -> SessionGlobals {
		SessionGlobals {
			symbol_interner: Lock::new(symbol::Interner::fresh()),
			span_interner: Lock::new(span_encoding::SpanInterner::default()),
		}
	}
}

// If this ever becomes non thread-local, `decode_syntax_context`
// and `decode_expn_id` will need to be updated to handle concurrent
// deserialization.
scoped_tls::scoped_thread_local!(pub static SESSION_GLOBALS: SessionGlobals);

pub fn with_session_globals<R>(f: impl FnOnce() -> R) -> R {
	let session_globals = SessionGlobals::new();
	SESSION_GLOBALS.set(&session_globals, f)
}

pub fn with_compatible_session_globals<R>(f: impl FnOnce() -> R) -> R {
	rustc_span::with_default_session_globals(f)
}

/// Represents a span.
///
/// Spans represent a region of code, used for error reporting. Positions in spans
/// are *absolute* positions from the beginning of the [`SourceMap`], not positions
/// relative to [`SourceFile`]s. Methods on the `SourceMap` can be used to relate spans back
/// to the original source.
///
/// You must be careful if the span crosses more than one file, since you will not be
/// able to use many of the functions on spans in source_map and you cannot assume
/// that the length of the span is equal to `span.hi - span.lo`; there may be space in the
/// [`BytePos`] range between files.
///
/// `SpanData` is public because `Span` uses a thread-local interner and can't be
/// sent to other threads, but some pieces of performance infra run in a separate thread.
/// Using `Span` is generally preferred.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub struct SpanData {
	pub lo: BytePos,
	pub hi: BytePos,
	/// Information about where the macro came from, if this piece of
	/// code was created by a macro expansion.
	pub ctxt: SyntaxContext,
}

impl SpanData {
	#[inline]
	pub fn span(&self) -> Span {
		Span::new(self.lo, self.hi, self.ctxt)
	}
	#[inline]
	pub fn with_lo(&self, lo: BytePos) -> Span {
		Span::new(lo, self.hi, self.ctxt)
	}
	#[inline]
	pub fn with_hi(&self, hi: BytePos) -> Span {
		Span::new(self.lo, hi, self.ctxt)
	}
	#[inline]
	pub fn with_ctxt(&self, ctxt: SyntaxContext) -> Span {
		Span::new(self.lo, self.hi, ctxt)
	}
}

impl Span {
	#[inline]
	pub fn lo(self) -> BytePos {
		self.data().lo
	}
	#[inline]
	pub fn with_lo(self, lo: BytePos) -> Span {
		self.data().with_lo(lo)
	}
	#[inline]
	pub fn hi(self) -> BytePos {
		self.data().hi
	}
	#[inline]
	pub fn with_hi(self, hi: BytePos) -> Span {
		self.data().with_hi(hi)
	}
	#[inline]
	pub fn ctxt(self) -> SyntaxContext {
		self.data().ctxt
	}
	#[inline]
	pub fn with_ctxt(self, ctxt: SyntaxContext) -> Span {
		self.data().with_ctxt(ctxt)
	}

	/// Returns a new span representing an empty span at the beginning of this span.
	#[inline]
	pub fn shrink_to_lo(self) -> Span {
		let span = self.data();
		span.with_hi(span.lo)
	}
	/// Returns a new span representing an empty span at the end of this span.
	#[inline]
	pub fn shrink_to_hi(self) -> Span {
		let span = self.data();
		span.with_lo(span.hi)
	}

	/// Returns `true` if this is a dummy span with any hygienic context.
	#[inline]
	pub fn is_dummy(self) -> bool {
		let span = self.data();
		span.lo.0 == 0 && span.hi.0 == 0
	}

	#[inline]
	pub fn with_root_ctxt(lo: BytePos, hi: BytePos) -> Span {
		Span::new(lo, hi, SyntaxContext::root())
	}

	/// Returns a `Span` that would enclose both `self` and `end`.
	///
	/// ```text
	///     ____             ___
	///     self lorem ipsum end
	///     ^^^^^^^^^^^^^^^^^^^^
	/// ```
	pub fn to(self, end: Span) -> Span {
		let span_data = self.data();
		let end_data = end.data();
		// FIXME(jseyfried): `self.ctxt` should always equal `end.ctxt` here (cf. issue #23480).
		// Return the macro span on its own to avoid weird diagnostic output. It is preferable to
		// have an incomplete span than a completely nonsensical one.
		if span_data.ctxt != end_data.ctxt {
			if span_data.ctxt == SyntaxContext::root() {
				return end;
			} else if end_data.ctxt == SyntaxContext::root() {
				return self;
			}
			// Both spans fall within a macro.
			// FIXME(estebank): check if it is the *same* macro.
		}
		Span::new(
			cmp::min(span_data.lo, end_data.lo),
			cmp::max(span_data.hi, end_data.hi),
			if span_data.ctxt == SyntaxContext::root() {
				end_data.ctxt
			} else {
				span_data.ctxt
			},
		)
	}
}

pub static SPAN_DEBUG: AtomicRef<fn(Span, &mut fmt::Formatter<'_>) -> fmt::Result> =
	AtomicRef::new(&(default_span_debug as fn(_, &mut fmt::Formatter<'_>) -> _));

pub fn default_span_debug(span: Span, f: &mut fmt::Formatter<'_>) -> fmt::Result {
	f.debug_struct("Span")
		.field("lo", &span.lo())
		.field("hi", &span.hi())
		.field("ctxt", &span.ctxt())
		.finish()
}

impl fmt::Debug for Span {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		(*SPAN_DEBUG)(*self, f)
	}
}

impl fmt::Debug for SpanData {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		(*SPAN_DEBUG)(Span::new(self.lo, self.hi, self.ctxt), f)
	}
}

impl From<Span> for MultiSpan {
	fn from(span: Span) -> MultiSpan {
		let s: rustc_span::Span = unsafe { std::mem::transmute::<Span, rustc_span::Span>(span) };
		MultiSpan::from_span(s)
	}
}

impl From<Span> for rustc_span::Span {
	fn from(span: Span) -> rustc_span::Span {
		unsafe { std::mem::transmute::<Span, rustc_span::Span>(span) }
	}
}
