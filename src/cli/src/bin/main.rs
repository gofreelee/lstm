use log;

fn main() -> Result<(), std::io::Error> {
	env_logger::init();

	log::info!("Mica CLI");
	Ok(())
}
