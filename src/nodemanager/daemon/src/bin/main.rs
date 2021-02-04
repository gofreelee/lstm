use clap::{App, Arg};
use log;
use std::net::TcpListener;

use mica_nodemanager::config::Configuration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	env_logger::init();

	let matches = App::new("Node manager")
		.arg(
			Arg::with_name("config")
				.long("config")
				.takes_value(true)
				.required(true),
		)
		.get_matches();
	let conf = Configuration::load(matches.value_of("config").unwrap())?;
	let listener = TcpListener::bind(conf.endpoint)?;
	let node_manager = mica_nodemanager::new_node_manager(conf);
	log::info!("Node manager started at {}", listener.local_addr().unwrap());
	node_manager.start(listener).await?;
	Ok(())
}
