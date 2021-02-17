fn main() {
	tonic_build::configure()
		.compile(&["../../protos/services.proto"], &["../../protos"])
		.unwrap();
}
