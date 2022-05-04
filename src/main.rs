extern crate libvktypes;

use libvktypes::instance::LibHandler;

const BUFFER_SIZE: usize = 64_000;

mod cipher_type;
mod worker;
mod key_manager;

use cipher_type::CipherType;
use key_manager::Key;

use worker::{
	GPUHandler,
	GPUWorker,
	WorkerType
};

use std::net::{
	IpAddr,
	Ipv4Addr,
	SocketAddr,
	UdpSocket
};

fn main() {
	// Special for NVIDIA GeForce GTX 750 Ti
	let vk_lib = LibHandler::new(1, 2, 0, true).unwrap();

	let dev = GPUHandler::new(&vk_lib);

	let speck_key = Key::<16>::from_file("speck_128_128_test.key");

	#[cfg(debug_assertions)]
	println!("[Debug] received key speck_128_128_test.key: {:02x?}", speck_key.data());

	let udp_socket = UdpSocket::bind("127.0.0.1:34254").unwrap();

	let out_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

	let create_info = WorkerType {
		shader_path: "compiled_shaders/speck_128_128.spv",
		buffer_size: BUFFER_SIZE as u32,
		cipher_type: CipherType::Speck128_128,
		key: speck_key.data()
	};

	let worker = GPUWorker::new(&dev, &create_info);

	let mut receive_data = |bytes: &mut [u8]| {
		udp_socket.recv_from(bytes).expect("Failed to read from UDP socket");

		#[cfg(debug_assertions)]
		println!("[Debug] received data from udp socket: {:02x?}", bytes);
	};

	worker.apply(&mut receive_data);

	worker.exec();

	let mut send_data = |bytes: &mut [u8]| {
		udp_socket.send_to(bytes, out_addr).expect("Failed to write to UDP socket");

		#[cfg(debug_assertions)]
		println!("[Debug] worker result: {:02x?}", bytes);
	};

	worker.apply(&mut send_data);
}
