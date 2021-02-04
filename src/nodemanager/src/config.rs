use toml::Value;
use uuid::Uuid;

use std::fs;
use std::io::ErrorKind;
use std::net::SocketAddr;

#[derive(Debug)]
pub struct Configuration {
	pub uuid: Vec<u8>,
	pub endpoint: SocketAddr,
	pub hostname: String,
}

impl Configuration {
	pub fn load(filename: &str) -> Result<Configuration, std::io::Error> {
		let contents = fs::read_to_string(filename)?;
		let value = contents.parse::<Value>()?;

		let uuid_str = value["uuid"]
			.as_str()
			.ok_or(std::io::Error::new(ErrorKind::Other, "UUID unspecified"));
		let uuid_bytes = Uuid::parse_str(uuid_str?)
			.map_err(|x| std::io::Error::new(ErrorKind::Other, x))?
			.as_bytes()
			.to_vec();

		let endpoint = value["endpoint"].as_str().ok_or(std::io::Error::new(
			ErrorKind::Other,
			"Endpoint not specified",
		));
		let endpoint_addr = endpoint?
			.to_string()
			.parse()
			.map_err(|x| std::io::Error::new(ErrorKind::Other, x))?;
		let hostname = value["hostname"].as_str().ok_or(std::io::Error::new(
			ErrorKind::Other,
			"Hostname not specified",
		));
		return Ok(Configuration {
			uuid: uuid_bytes,
			endpoint: endpoint_addr,
			hostname: hostname?.to_string(),
		});
	}
}
