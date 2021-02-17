use log;
use mica_rpc_api::node_manager_service_server::{NodeManagerService, NodeManagerServiceServer};
use mica_rpc_api::{DeployJobRequest, DeployJobResponse};
use tonic::{transport::Server, Request, Response, Status};

use crate::config::Configuration;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};
use std::sync::{Arc, RwLock};
use weak_self::WeakSelf;

#[derive(Debug)]
struct NodeManagerServerStub {
	me: Arc<NodeManagerServerImpl>,
}

#[derive(Debug)]
pub struct NodeManagerServerImpl {
	me: WeakSelf<NodeManagerServerImpl>,
	state: RwLock<NodeManagerServerState>,
}

#[derive(Debug)]
pub struct NodeManagerServerState {
	pub conf: Configuration,
	local_addr: SocketAddr,
}

#[tonic::async_trait]
impl NodeManagerService for NodeManagerServerStub {
	async fn deploy_job(
		&self,
		request: Request<DeployJobRequest>,
	) -> Result<Response<DeployJobResponse>, Status> {
		log::info!("Got Deploy Job: {:?}", request);
		let reply = DeployJobResponse::default();

		Ok(Response::new(reply))
	}
}

impl NodeManagerServerImpl {
	pub async fn start(&self, listener: TcpListener) -> Result<(), Box<dyn std::error::Error>> {
		let mut state = self.state.write().unwrap();
		state.local_addr = *&listener.local_addr().unwrap();
		let incoming = tokio_stream::wrappers::TcpListenerStream::new(
			tokio::net::TcpListener::from_std(listener)?,
		);
		let me = self.me.get().upgrade().unwrap();

		let this = NodeManagerServerStub { me: me };
		let stubs = NodeManagerServiceServer::new(this);
		let server = Server::builder().add_service(stubs);
		server.serve_with_incoming(incoming).await?;
		Ok(())
	}
}

pub(crate) fn new_node_manager(conf: Configuration) -> Arc<NodeManagerServerImpl> {
	let state = RwLock::new(NodeManagerServerState {
		conf: conf,
		local_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 0),
	});
	let this = Arc::new(NodeManagerServerImpl {
		me: WeakSelf::default(),
		state: state,
	});
	this.me.init(&this);
	return this;
}
