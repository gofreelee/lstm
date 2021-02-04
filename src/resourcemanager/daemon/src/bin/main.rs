use log;
use std::{thread, time};

fn main() -> Result<(), std::io::Error> {
	env_logger::init();

	log::info!("Resource manager manager started");
	loop {
		thread::sleep(time::Duration::from_millis(1000));
	}
}
