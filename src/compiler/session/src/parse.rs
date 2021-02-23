use mica_compiler_span::source_map::{FilePathMapping, SourceMap};
use mica_compiler_span::symbol::Symbol;
use mica_compiler_span::Span;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::{Lock, Lrc};
use rustc_errors::emitter::ColorConfig;
use rustc_errors::Handler;

/// Info about a parsing session.
pub struct ParseSess {
	pub span_diagnostic: Handler,
	source_map: Lrc<SourceMap>,
	pub symbol_gallery: SymbolGallery,
}

impl ParseSess {
	pub fn new(file_path_mapping: FilePathMapping) -> Self {
		let sm = Lrc::new(SourceMap::new(file_path_mapping));
		let handler = Handler::with_tty_emitter(ColorConfig::Auto, true, None, Some(sm.clone()));
		ParseSess::with_span_handler(handler, sm)
	}

	pub fn with_span_handler(handler: Handler, source_map: Lrc<SourceMap>) -> Self {
		Self {
			span_diagnostic: handler,
			source_map,
			symbol_gallery: SymbolGallery::default(),
		}
	}

	#[inline]
	pub fn source_map(&self) -> &SourceMap {
		&self.source_map
	}
}

#[derive(Default)]
pub struct SymbolGallery {
	/// All symbols occurred and their first occurrence span.
	pub symbols: Lock<FxHashMap<Symbol, Span>>,
}

impl SymbolGallery {
	/// Insert a symbol and its span into symbol gallery.
	/// If the symbol has occurred before, ignore the new occurance.
	pub fn insert(&self, symbol: Symbol, span: Span) {
		self.symbols.lock().entry(symbol).or_insert(span);
	}
}
