use super::ParseSess;
use mica_compiler_span::source_map::FilePathMapping;

pub fn build_parse_session() -> ParseSess {
	let parse_sess = ParseSess::new(FilePathMapping::empty());
	parse_sess
}
