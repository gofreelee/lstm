pub mod config;
mod rpc;

use crate::config::Configuration;
use crate::rpc::NodeManagerServerImpl;
use std::sync::Arc;

pub fn new_node_manager(conf: Configuration) -> Arc<NodeManagerServerImpl> {
	crate::rpc::new_node_manager(conf)
}
