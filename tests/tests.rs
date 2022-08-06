//! All test vectorf form 'SIMON and SPECK Implementation Guide'

extern crate libvktypes;

use libvktypes::instance::LibHandler;
use libvktypes::hardware::{
	HWDescription,
	MemoryProperty
};
use libvktypes::logical_device::LogicalDevice;
use libvktypes::memory::{
	Memory,
	BufferType
};
use libvktypes::pipeline::ComputePipeline;
use libvktypes::shader::Shader;
use libvktypes::cmd_queue::{
	ComputeQueue,
	AccessType,
	PipelineStage
};
use libvktypes::specialization_constants::SpecializationConstant;

pub const DEV_INDEX: usize = 0;
pub const QUEUE_FAMILY_INDEX: usize = 0;
pub const SPECK_128_128_SHADER_PATH: &str = "compiled_shaders/speck_128_128.spv";
pub const ATTEMPTS_NUM: usize = 100;
pub const SLEEP_DURATION: Duration = Duration::from_secs(1);

use std::io::Read;
use std::fs::{
	File,
	remove_file,
};
use std::path::Path;
use std::time::Duration;
use std::thread;
use std::io::Write;

fn validate_shader(shader_path: &str, push_const: &[u8], src: &[u8], exp_result: &[u8]) {
	assert_eq!(src.len(), exp_result.len());

	// Special for NVIDIA GeForce GTX 750 Ti
	let vk_lib = LibHandler::new(1, 2, 0, true).unwrap();

	let hw_list = HWDescription::list(&vk_lib).unwrap();

	let dev = LogicalDevice::new(&vk_lib, &hw_list[DEV_INDEX], QUEUE_FAMILY_INDEX).unwrap();

	let memory = Memory::new(&dev, src.len() as u64,
		MemoryProperty::HOST_VISIBLE | MemoryProperty::HOST_COHERENT | MemoryProperty::DEVICE_LOCAL,
		BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST
	).unwrap();

	let shader = Shader::from_src(&dev, shader_path, String::from("main")).unwrap();

	let pipeline = ComputePipeline::new(&dev, &[&memory], &shader, &SpecializationConstant::empty(), push_const.len() as u32).unwrap();

	let cmd_queue = ComputeQueue::new(&dev).unwrap();

	if !push_const.is_empty() {
		cmd_queue.update_push_constants(&pipeline, push_const);
	}

	cmd_queue.cmd_bind_pipeline(&pipeline);
	cmd_queue.dispatch(1, 1, 1);

	cmd_queue.submit().unwrap();

	let mut f = |bytes: &mut [u8]| {
		bytes.copy_from_slice(src);
	};

	memory.write(&mut f).unwrap();

	cmd_queue.exec(PipelineStage::COMPUTE_SHADER, u64::MAX).unwrap();

	let data: &[u8] = memory.read().unwrap();

	assert_eq!(*data, *exp_result);
}

#[cfg(test)]
mod tests {
	use crate::*;

	#[test]
	fn lcs() {
		const BUFFER_SIZE: usize = 8;

		let test: [u8; BUFFER_SIZE] =
		[
			0x6c, 0x61, 0x76, 0x69, 0x75, 0x71, 0x65, 0x20
		];

		let exp_result: [u8; BUFFER_SIZE] =
		[
			0x61, 0x0b, 0xb3, 0x4b, 0xab, 0x8b, 0x2b, 0x03
		];

		validate_shader("compiled_shaders/tests/rot_left.spv", &[], &test, &exp_result);
	}

	#[test]
	fn rcs() {
		const BUFFER_SIZE: usize = 8;

		let test: [u8; BUFFER_SIZE] =
		[
			0x6c, 0x61, 0x76, 0x69, 0x75, 0x71, 0x65, 0x20,
		];

		let exp_result: [u8; BUFFER_SIZE] =
		[
			0x61, 0x76, 0x69, 0x75, 0x71, 0x65, 0x20, 0x6c
		];

		validate_shader("compiled_shaders/tests/rot_right.spv", &[], &test, &exp_result);
	}

	#[test]
	fn base_add() {
		const BUFFER_SIZE: usize = 8;
		const PUSH_CONST_SIZE: usize = 20;

		let test: [u8; BUFFER_SIZE] = [0; BUFFER_SIZE];

		let exp_result: [u8; BUFFER_SIZE] =
		[
			0x09, 0x04, 0x1f, 0x1a, 0x35, 0x40, 0x2b, 0x3e
		];

		let push_constant: [u8; PUSH_CONST_SIZE] =
		[
			0x00, 0x00, 0x00, 0x10,
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
			0x09, 0x03, 0x1d, 0x17, 0x31, 0x3b, 0x25, 0x37
		];

		validate_shader("compiled_shaders/tests/add.spv", &push_constant, &test, &exp_result);
	}

	#[test]
	fn speck_128_128_ks() {
		const BUFFER_SIZE: usize = 16;
		const PUSH_CONST_SIZE: usize = 4;

		let test: [u8; BUFFER_SIZE] =
		[
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
			0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f
		];

		let exp_result: [u8; BUFFER_SIZE] =
		[
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
			0x09, 0x03, 0x1d, 0x17, 0x31, 0x3b, 0x25, 0x37
		];

		let push_constant: [u8; PUSH_CONST_SIZE] =
		[
			0x00, 0x00, 0x00, 0x10
		];

		validate_shader("compiled_shaders/tests/speck_128_128_ks_round.spv", &push_constant, &test, &exp_result);
	}

	#[test]
	fn speck_128_128_round() {
		const BUFFER_SIZE: usize = 16;
		const PUSH_CONST_SIZE: usize = 20;

		let test: [u8; BUFFER_SIZE] =
		[
			0x20, 0x6d, 0x61, 0x64, 0x65, 0x20, 0x69, 0x74,
			0x20, 0x65, 0x71, 0x75, 0x69, 0x76, 0x61, 0x6c
		];

		let exp_result: [u8; BUFFER_SIZE] =
		[
			0x86, 0xb6, 0xdf, 0xed, 0xf4, 0x87, 0x9a, 0x30,
			0x85, 0xdf, 0xd4, 0xce, 0xdf, 0x84, 0xd3, 0x93
		];

		let push_constant: [u8; PUSH_CONST_SIZE] =
		[
			0x00, 0x00, 0x00, 0x20,
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
			0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f
		];

		validate_shader("compiled_shaders/tests/speck_128_128_round.spv", &push_constant, &test, &exp_result);
	}

	#[test]
	fn speck_128_128() {
		const BUFFER_SIZE: usize = 16;
		const PUSH_CONST_SIZE: usize = 20;

		let test: [u8; BUFFER_SIZE] =
		[
			0x20, 0x6d, 0x61, 0x64, 0x65, 0x20, 0x69, 0x74,
			0x20, 0x65, 0x71, 0x75, 0x69, 0x76, 0x61, 0x6c
		];

		let exp_result: [u8; BUFFER_SIZE] =
		[
			0x18, 0x0d, 0x57, 0x5c, 0xdf, 0xfe, 0x60, 0x78,
		 	0x65, 0x32, 0x78, 0x79, 0x51, 0x98, 0x5d, 0xa6
		];

		let push_constant: [u8; PUSH_CONST_SIZE] =
		[
			0x00, 0x00, 0x00, 0x20,
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
			0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f
		];

		validate_shader("compiled_shaders/speck_128_128.spv", &push_constant, &test, &exp_result);
	}
}

#[cfg(test)]
mod performance {
	use crate::*;

	// Approx. values
	const BLOCK_0_5_GB: usize = 512*1024*1024;
	const BLOCK_1_GB: usize = 2*BLOCK_0_5_GB;

	const PUSH_CONST_SIZE: usize = 20;

	fn encrypt_data(data_size: usize, block_size: usize, out_file: &str) {
		let blocks_num: usize = data_size / block_size;

		// Special for NVIDIA GeForce GTX 750 Ti
		let vk_lib = LibHandler::new(1, 2, 0, true).unwrap();

		let hw_list = HWDescription::list(&vk_lib).unwrap();

		let dev = LogicalDevice::new(&vk_lib, &hw_list[DEV_INDEX], QUEUE_FAMILY_INDEX).unwrap();

		let memory = Memory::new(&dev, block_size as u64,
			MemoryProperty::HOST_VISIBLE | MemoryProperty::HOST_COHERENT | MemoryProperty::HOST_CACHED,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST
		).unwrap();

		let shader = Shader::from_src(&dev, SPECK_128_128_SHADER_PATH, String::from("main")).unwrap();

		let pipeline = ComputePipeline::new(&dev, &[&memory], &shader, &SpecializationConstant::empty(), PUSH_CONST_SIZE as u32).unwrap();

		let cmd_queue = ComputeQueue::new(&dev).unwrap();

		let push_constant: [u8; PUSH_CONST_SIZE] =
		[
			0x00, 0x00, 0x00, 0x20,
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
			0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f
		];

		cmd_queue.update_push_constants(&pipeline, &push_constant);

		cmd_queue.cmd_bind_pipeline(&pipeline);
		cmd_queue.dispatch(1, 1, 1);

		cmd_queue.submit().unwrap();

		let out_path = Path::new(out_file);

		if out_path.exists() {
			remove_file(out_path).unwrap();
		}

		let mut file = File::options().append(true).create(true).open(out_file).unwrap();

		for i in 0..ATTEMPTS_NUM {
			let mut set_memory = |bytes: &mut [u8]| {
				bytes.fill(i as u8);
			};

			let now = std::time::Instant::now();

			for _ in 0..blocks_num {
				memory.write(&mut set_memory).unwrap();

				cmd_queue.exec(PipelineStage::COMPUTE_SHADER, u64::MAX).unwrap();

				memory.read().unwrap();
			}

			writeln!(file, "{}", now.elapsed().as_millis());

			println!("[{}/{}]", i+1, ATTEMPTS_NUM);

			thread::sleep(SLEEP_DURATION);
		}
	}

	#[test]
	fn encrypt_0_5gb() {
		encrypt_data(BLOCK_0_5_GB, BLOCK_0_5_GB, "tests/misc/perf_results/0_5gb.csv");
	}

	#[test]
	fn encrypt_1gb() {
		encrypt_data(BLOCK_1_GB, BLOCK_1_GB, "tests/misc/perf_results/1gb.csv");
	}

	#[test]
	fn encrypt_1_5gb() {
		encrypt_data(BLOCK_1_GB + BLOCK_0_5_GB, BLOCK_1_GB + BLOCK_0_5_GB, "tests/misc/perf_results/1_5gb.csv");
	}

	#[test]
	fn encrypt_2gb() {
		encrypt_data(2*BLOCK_1_GB, 2*BLOCK_1_GB, "tests/misc/perf_results/2gb.csv");
	}

	#[test]
	fn encrypt_2_5gb() {
		encrypt_data(2*BLOCK_1_GB + BLOCK_0_5_GB, 2*BLOCK_1_GB + BLOCK_0_5_GB, "tests/misc/perf_results/2_5gb.csv");
	}

	#[test]
	fn encrypt_3gb() {
		encrypt_data(3*BLOCK_1_GB, 3*BLOCK_1_GB, "tests/misc/perf_results/3gb.csv");
	}
}

#[cfg(test)]
mod throughput {
	use crate::*;

	const BLOCK_256MB: usize = 128*1024*1024;
	const BLOCK_0_5_GB: usize = 512*1024*1024;
	const BLOCK_1_GB: usize = 2*BLOCK_0_5_GB;

	const PUSH_CONST_SIZE: usize = 20;

	fn encrypt_data(data_size: usize, block_size: usize) {
		let blocks_num: usize = data_size / block_size;

		// Special for NVIDIA GeForce GTX 750 Ti
		let vk_lib = LibHandler::new(1, 2, 0, true).unwrap();

		let hw_list = HWDescription::list(&vk_lib).unwrap();

		let dev = LogicalDevice::new(&vk_lib, &hw_list[DEV_INDEX], QUEUE_FAMILY_INDEX).unwrap();

		let memory = Memory::new(&dev, block_size as u64,
			MemoryProperty::HOST_VISIBLE | MemoryProperty::HOST_COHERENT | MemoryProperty::HOST_CACHED,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST
		).unwrap();

		let shader = Shader::from_src(&dev, SPECK_128_128_SHADER_PATH, String::from("main")).unwrap();

		let pipeline = ComputePipeline::new(&dev, &[&memory], &shader, &SpecializationConstant::empty(), PUSH_CONST_SIZE as u32).unwrap();

		let cmd_queue = ComputeQueue::new(&dev).unwrap();

		let push_constant: [u8; PUSH_CONST_SIZE] =
		[
			0x00, 0x00, 0x00, 0x20,
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
			0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f
		];

		cmd_queue.update_push_constants(&pipeline, &push_constant);

		cmd_queue.cmd_bind_pipeline(&pipeline);
		cmd_queue.dispatch(1, 1, 1);

		cmd_queue.submit().unwrap();


		let mut set_memory = |bytes: &mut [u8]| {
			bytes.fill(0xaf);
		};

		let now = std::time::Instant::now();

		for _ in 0..blocks_num {
			memory.write(&mut set_memory).unwrap();

			cmd_queue.exec(PipelineStage::COMPUTE_SHADER, u64::MAX).unwrap();

			memory.read().unwrap();
		}

		println!("{}", now.elapsed().as_millis());
	}

	#[test]
	fn determine_throughput() {
		for i in 10..20 {
			println!("{} GB", i);

			encrypt_data(i*BLOCK_1_GB, BLOCK_256MB);

			thread::sleep(SLEEP_DURATION);
		}
	}
}
