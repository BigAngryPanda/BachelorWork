//! All test vectorf form 'SIMON and SPECK Implementation Guide'

#[cfg(test)]
mod tests {
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

	pub static DEV_INDEX: usize = 0;
	pub static QUEUE_FAMILY_INDEX: usize = 0;

	#[test]
	fn lcs() {
		// Special for NVIDIA GeForce GTX 750 Ti
		let vk_lib = LibHandler::new(1, 2, 0, true).unwrap();

		let hw_list = HWDescription::list(&vk_lib).unwrap();

		let dev = LogicalDevice::new(&vk_lib, &hw_list[DEV_INDEX], QUEUE_FAMILY_INDEX).unwrap();

		const BUFFER_SIZE: u64 = 8;

		let test: [u8; BUFFER_SIZE as usize] =
		[
			0x6c, 0x61, 0x76, 0x69, 0x75, 0x71, 0x65, 0x20
		];

		let exp_result: [u8; BUFFER_SIZE as usize] =
		[
			0x61, 0x0b, 0xb3, 0x4b, 0xab, 0x8b, 0x2b, 0x03
		];

		let host_memory = Memory::new(&dev, BUFFER_SIZE,
			MemoryProperty::HOST_VISIBLE,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST
		).unwrap();

		let mut f = |bytes: &mut [u8]| {
			bytes.clone_from_slice(&test);
		};

		host_memory.write(&mut f).unwrap();

		let dev_memory = Memory::new(&dev, BUFFER_SIZE,
			MemoryProperty::DEVICE_LOCAL,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST).unwrap();

		let shader = Shader::from_src(&dev, "compiled_shaders/tests/rot_left.spv", String::from("main")).unwrap();

		let pipeline = ComputePipeline::new(&dev, &[&dev_memory], &shader, &SpecializationConstant::empty(), 0).unwrap();

		let cmd_queue = ComputeQueue::new(&dev).unwrap();

		cmd_queue.cmd_copy(&host_memory, &dev_memory);

		cmd_queue.cmd_set_barrier(&dev_memory,
			AccessType::HOST_WRITE,
			AccessType::SHADER_READ,
			PipelineStage::HOST,
			PipelineStage::COMPUTE_SHADER);

		cmd_queue.cmd_bind_pipeline(&pipeline);
		cmd_queue.dispatch(BUFFER_SIZE as u32, 1, 1);

		cmd_queue.cmd_set_barrier(&dev_memory,
			AccessType::SHADER_WRITE,
			AccessType::TRANSFER_READ,
			PipelineStage::COMPUTE_SHADER,
			PipelineStage::TRANSFER);

		cmd_queue.cmd_copy(&dev_memory, &host_memory);

		cmd_queue.cmd_set_barrier(&host_memory,
			AccessType::TRANSFER_WRITE,
			AccessType::HOST_READ,
			PipelineStage::TRANSFER,
			PipelineStage::HOST);

		cmd_queue.submit().unwrap();

		cmd_queue.exec(PipelineStage::TRANSFER, u64::MAX).unwrap();

		let data: &[u8] = host_memory.read().unwrap();

		assert_eq!(*data, exp_result);
	}

	#[test]
	fn rcs() {
		// Special for NVIDIA GeForce GTX 750 Ti
		let vk_lib = LibHandler::new(1, 2, 0, true).unwrap();

		let hw_list = HWDescription::list(&vk_lib).unwrap();

		let dev = LogicalDevice::new(&vk_lib, &hw_list[DEV_INDEX], QUEUE_FAMILY_INDEX).unwrap();

		const BUFFER_SIZE: u64 = 8;

		let test: [u8; BUFFER_SIZE as usize] =
		[
			0x6c, 0x61, 0x76, 0x69, 0x75, 0x71, 0x65, 0x20,
		];

		let exp_result: [u8; BUFFER_SIZE as usize] =
		[
			0x61, 0x76, 0x69, 0x75, 0x71, 0x65, 0x20, 0x6c
		];

		let host_memory = Memory::new(&dev, BUFFER_SIZE,
			MemoryProperty::HOST_VISIBLE,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST
		).unwrap();

		let mut f = |bytes: &mut [u8]| {
			bytes.clone_from_slice(&test);
		};

		host_memory.write(&mut f).unwrap();

		let dev_memory = Memory::new(&dev, BUFFER_SIZE,
			MemoryProperty::DEVICE_LOCAL,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST).unwrap();

		let shader = Shader::from_src(&dev, "compiled_shaders/tests/rot_right.spv", String::from("main")).unwrap();

		let pipeline = ComputePipeline::new(&dev, &[&dev_memory], &shader, &SpecializationConstant::empty(), 0).unwrap();

		let cmd_queue = ComputeQueue::new(&dev).unwrap();

		cmd_queue.cmd_copy(&host_memory, &dev_memory);

		cmd_queue.cmd_set_barrier(&dev_memory,
			AccessType::HOST_WRITE,
			AccessType::SHADER_READ,
			PipelineStage::HOST,
			PipelineStage::COMPUTE_SHADER);

		cmd_queue.cmd_bind_pipeline(&pipeline);
		cmd_queue.dispatch(BUFFER_SIZE as u32, 1, 1);

		cmd_queue.cmd_set_barrier(&dev_memory,
			AccessType::SHADER_WRITE,
			AccessType::TRANSFER_READ,
			PipelineStage::COMPUTE_SHADER,
			PipelineStage::TRANSFER);

		cmd_queue.cmd_copy(&dev_memory, &host_memory);

		cmd_queue.cmd_set_barrier(&host_memory,
			AccessType::TRANSFER_WRITE,
			AccessType::HOST_READ,
			PipelineStage::TRANSFER,
			PipelineStage::HOST);

		cmd_queue.submit().unwrap();

		cmd_queue.exec(PipelineStage::TRANSFER, u64::MAX).unwrap();

		let data: &[u8] = host_memory.read().unwrap();

		assert_eq!(*data, exp_result);
	}

	#[test]
	fn base_add() {
		// Special for NVIDIA GeForce GTX 750 Ti
		let vk_lib = LibHandler::new(1, 2, 0, true).unwrap();

		let hw_list = HWDescription::list(&vk_lib).unwrap();

		let dev = LogicalDevice::new(&vk_lib, &hw_list[DEV_INDEX], QUEUE_FAMILY_INDEX).unwrap();

		const BUFFER_SIZE: u64 = 8;

		const PUSH_CONST_SIZE: usize = 20;

		let exp_result: [u8; BUFFER_SIZE as usize] =
		[
			0x09, 0x04, 0x1f, 0x1a, 0x35, 0x40, 0x2b, 0x3e
		];

		let host_memory = Memory::new(&dev, BUFFER_SIZE,
			MemoryProperty::HOST_VISIBLE,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST
		).unwrap();

		let dev_memory = Memory::new(&dev, BUFFER_SIZE,
			MemoryProperty::DEVICE_LOCAL,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST).unwrap();

		let shader = Shader::from_src(&dev, "compiled_shaders/tests/add.spv", String::from("main")).unwrap();

		let pipeline = ComputePipeline::new(&dev, &[&dev_memory], &shader, &SpecializationConstant::empty(), PUSH_CONST_SIZE as u32).unwrap();

		let cmd_queue = ComputeQueue::new(&dev).unwrap();

		let push_constant: [u8; PUSH_CONST_SIZE] =
		[
			0x00, 0x00, 0x00, 0x10,
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
			0x09, 0x03, 0x1d, 0x17, 0x31, 0x3b, 0x25, 0x37
		];

		cmd_queue.update_push_constants(&pipeline, &push_constant);

		cmd_queue.cmd_bind_pipeline(&pipeline);
		cmd_queue.dispatch(BUFFER_SIZE as u32, 1, 1);

		cmd_queue.cmd_set_barrier(&dev_memory,
			AccessType::SHADER_WRITE,
			AccessType::TRANSFER_READ,
			PipelineStage::COMPUTE_SHADER,
			PipelineStage::TRANSFER);

		cmd_queue.cmd_copy(&dev_memory, &host_memory);

		cmd_queue.cmd_set_barrier(&host_memory,
			AccessType::TRANSFER_WRITE,
			AccessType::HOST_READ,
			PipelineStage::TRANSFER,
			PipelineStage::HOST);

		cmd_queue.submit().unwrap();

		cmd_queue.exec(PipelineStage::TRANSFER, u64::MAX).unwrap();

		let data: &[u8] = host_memory.read().unwrap();

		assert_eq!(*data, exp_result);
	}

	#[test]
	fn speck_128_128_ks() {
		const BUFFER_SIZE: u64 = 16;

		const PUSH_CONST_SIZE: usize = 4;

		let pt: [u8; BUFFER_SIZE as usize] =
		[
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
			0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f
		];

		let ct: [u8; BUFFER_SIZE as usize] =
		[
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
			0x09, 0x03, 0x1d, 0x17, 0x31, 0x3b, 0x25, 0x37
		];

		// Special for NVIDIA GeForce GTX 750 Ti
		let vk_lib = LibHandler::new(1, 2, 0, true).unwrap();

		let hw_list = HWDescription::list(&vk_lib).unwrap();

		let dev = LogicalDevice::new(&vk_lib, &hw_list[DEV_INDEX], QUEUE_FAMILY_INDEX).unwrap();

		let host_memory = Memory::new(&dev, BUFFER_SIZE,
			MemoryProperty::HOST_VISIBLE,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST
		).unwrap();

		let mut f = |bytes: &mut [u8]| {
			bytes.clone_from_slice(&pt);
		};

		host_memory.write(&mut f).unwrap();

		let dev_memory = Memory::new(&dev, BUFFER_SIZE,
			MemoryProperty::DEVICE_LOCAL,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST).unwrap();

		let shader = Shader::from_src(&dev, "compiled_shaders/tests/speck_128_128_ks_round.spv", String::from("main")).unwrap();

		let pipeline = ComputePipeline::new(&dev, &[&dev_memory], &shader, &SpecializationConstant::empty(), PUSH_CONST_SIZE as u32).unwrap();

		let cmd_queue = ComputeQueue::new(&dev).unwrap();

		cmd_queue.cmd_copy(&host_memory, &dev_memory);

		cmd_queue.cmd_set_barrier(&dev_memory,
			AccessType::HOST_WRITE,
			AccessType::SHADER_READ,
			PipelineStage::HOST,
			PipelineStage::COMPUTE_SHADER);

		let push_constant: [u8; PUSH_CONST_SIZE] =
		[
			0x00, 0x00, 0x00, 0x10
		];

		cmd_queue.update_push_constants(&pipeline, &push_constant);

		cmd_queue.cmd_bind_pipeline(&pipeline);
		cmd_queue.dispatch(BUFFER_SIZE as u32, 1, 1);

		cmd_queue.cmd_set_barrier(&dev_memory,
			AccessType::SHADER_WRITE,
			AccessType::TRANSFER_READ,
			PipelineStage::COMPUTE_SHADER,
			PipelineStage::TRANSFER);

		cmd_queue.cmd_copy(&dev_memory, &host_memory);

		cmd_queue.cmd_set_barrier(&host_memory,
			AccessType::TRANSFER_WRITE,
			AccessType::HOST_READ,
			PipelineStage::TRANSFER,
			PipelineStage::HOST);

		cmd_queue.submit().unwrap();

		cmd_queue.exec(PipelineStage::TRANSFER, u64::MAX).unwrap();

		let data: &[u8] = host_memory.read().unwrap();

		assert_eq!(*data, ct);
	}

	#[test]
	fn speck_128_128_round() {
		const BUFFER_SIZE: u64 = 16;

		const PUSH_CONST_SIZE: usize = 20;

		let pt: [u8; BUFFER_SIZE as usize] =
		[
			0x20, 0x6d, 0x61, 0x64, 0x65, 0x20, 0x69, 0x74,
			0x20, 0x65, 0x71, 0x75, 0x69, 0x76, 0x61, 0x6c
		];

		let ct: [u8; BUFFER_SIZE as usize] =
		[
			0x86, 0xb6, 0xdf, 0xed, 0xf4, 0x87, 0x9a, 0x30,
			0x85, 0xdf, 0xd4, 0xce, 0xdf, 0x84, 0xd3, 0x93
		];

		// Special for NVIDIA GeForce GTX 750 Ti
		let vk_lib = LibHandler::new(1, 2, 0, true).unwrap();

		let hw_list = HWDescription::list(&vk_lib).unwrap();

		let dev = LogicalDevice::new(&vk_lib, &hw_list[DEV_INDEX], QUEUE_FAMILY_INDEX).unwrap();

		let host_memory = Memory::new(&dev, BUFFER_SIZE,
			MemoryProperty::HOST_VISIBLE,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST
		).unwrap();

		let mut f = |bytes: &mut [u8]| {
			bytes.clone_from_slice(&pt);
		};

		host_memory.write(&mut f).unwrap();

		let dev_memory = Memory::new(&dev, BUFFER_SIZE,
			MemoryProperty::DEVICE_LOCAL,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST).unwrap();

		let shader = Shader::from_src(&dev, "compiled_shaders/tests/speck_128_128_round.spv", String::from("main")).unwrap();

		let pipeline = ComputePipeline::new(&dev, &[&dev_memory], &shader, &SpecializationConstant::empty(), PUSH_CONST_SIZE as u32).unwrap();

		let cmd_queue = ComputeQueue::new(&dev).unwrap();

		cmd_queue.cmd_copy(&host_memory, &dev_memory);

		cmd_queue.cmd_set_barrier(&dev_memory,
			AccessType::HOST_WRITE,
			AccessType::SHADER_READ,
			PipelineStage::HOST,
			PipelineStage::COMPUTE_SHADER);

		let push_constant: [u8; PUSH_CONST_SIZE] =
		[
			0x00, 0x00, 0x00, 0x20,
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
			0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f
		];

		cmd_queue.update_push_constants(&pipeline, &push_constant);

		cmd_queue.cmd_bind_pipeline(&pipeline);
		cmd_queue.dispatch(BUFFER_SIZE as u32, 1, 1);

		cmd_queue.cmd_set_barrier(&dev_memory,
			AccessType::SHADER_WRITE,
			AccessType::TRANSFER_READ,
			PipelineStage::COMPUTE_SHADER,
			PipelineStage::TRANSFER);

		cmd_queue.cmd_copy(&dev_memory, &host_memory);

		cmd_queue.cmd_set_barrier(&host_memory,
			AccessType::TRANSFER_WRITE,
			AccessType::HOST_READ,
			PipelineStage::TRANSFER,
			PipelineStage::HOST);

		cmd_queue.submit().unwrap();

		cmd_queue.exec(PipelineStage::TRANSFER, u64::MAX).unwrap();

		let data: &[u8] = host_memory.read().unwrap();

		assert_eq!(*data, ct);
	}

	#[test]
	fn speck_128_128() {
		const BUFFER_SIZE: u64 = 16;

		const PUSH_CONST_SIZE: usize = 20;

		let pt: [u8; BUFFER_SIZE as usize] =
		[
			0x20, 0x6d, 0x61, 0x64, 0x65, 0x20, 0x69, 0x74,
			0x20, 0x65, 0x71, 0x75, 0x69, 0x76, 0x61, 0x6c
		];

		let ct: [u8; BUFFER_SIZE as usize] =
		[
			0x18, 0x0d, 0x57, 0x5c, 0xdf, 0xfe, 0x60, 0x78,
		 	0x65, 0x32, 0x78, 0x79, 0x51, 0x98, 0x5d, 0xa6
		];

		// Special for NVIDIA GeForce GTX 750 Ti
		let vk_lib = LibHandler::new(1, 2, 0, true).unwrap();

		let hw_list = HWDescription::list(&vk_lib).unwrap();

		let dev = LogicalDevice::new(&vk_lib, &hw_list[DEV_INDEX], QUEUE_FAMILY_INDEX).unwrap();

		let host_memory = Memory::new(&dev, BUFFER_SIZE,
			MemoryProperty::HOST_VISIBLE,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST
		).unwrap();

		let mut f = |bytes: &mut [u8]| {
			bytes.clone_from_slice(&pt);
		};

		host_memory.write(&mut f).unwrap();

		let dev_memory = Memory::new(&dev, BUFFER_SIZE,
			MemoryProperty::DEVICE_LOCAL,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST).unwrap();

		let shader = Shader::from_src(&dev, "compiled_shaders/speck_128_128.spv", String::from("main")).unwrap();

		let pipeline = ComputePipeline::new(&dev, &[&dev_memory], &shader, &SpecializationConstant::empty(), PUSH_CONST_SIZE as u32).unwrap();

		let cmd_queue = ComputeQueue::new(&dev).unwrap();

		cmd_queue.cmd_copy(&host_memory, &dev_memory);

		cmd_queue.cmd_set_barrier(&dev_memory,
			AccessType::HOST_WRITE,
			AccessType::SHADER_READ,
			PipelineStage::HOST,
			PipelineStage::COMPUTE_SHADER);

		let push_constant: [u8; PUSH_CONST_SIZE] =
		[
			0x00, 0x00, 0x00, 0x20,
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
			0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f
		];

		cmd_queue.update_push_constants(&pipeline, &push_constant);

		cmd_queue.cmd_bind_pipeline(&pipeline);
		cmd_queue.dispatch(BUFFER_SIZE as u32, 1, 1);

		cmd_queue.cmd_set_barrier(&dev_memory,
			AccessType::SHADER_WRITE,
			AccessType::TRANSFER_READ,
			PipelineStage::COMPUTE_SHADER,
			PipelineStage::TRANSFER);

		cmd_queue.cmd_copy(&dev_memory, &host_memory);

		cmd_queue.cmd_set_barrier(&host_memory,
			AccessType::TRANSFER_WRITE,
			AccessType::HOST_READ,
			PipelineStage::TRANSFER,
			PipelineStage::HOST);

		cmd_queue.submit().unwrap();

		cmd_queue.exec(PipelineStage::TRANSFER, u64::MAX).unwrap();

		let data: &[u8] = host_memory.read().unwrap();

		assert_eq!(*data, ct);
	}
}

// Run this tests separately!
#[cfg(test)]
mod performance {
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

	pub static DEV_INDEX: usize = 0;
	pub static QUEUE_FAMILY_INDEX: usize = 0;

	use std::io::Read;
	use std::fs::File;
	use std::path::Path;

	fn encrypt_data<const BUFFER_SIZE: usize>() {
		const PUSH_CONST_SIZE: usize = 20;

		let mut pt = File::open(Path::new("tests/misc/plaintext")).expect("Failed to open file");

		// Special for NVIDIA GeForce GTX 750 Ti
		let vk_lib = LibHandler::new(1, 2, 0, false).unwrap();

		let hw_list = HWDescription::list(&vk_lib).unwrap();

		let dev = LogicalDevice::new(&vk_lib, &hw_list[DEV_INDEX], QUEUE_FAMILY_INDEX).unwrap();

		let host_memory = Memory::new(&dev, BUFFER_SIZE as u64,
			MemoryProperty::HOST_VISIBLE,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST
		).unwrap();

		let dev_memory = Memory::new(&dev, BUFFER_SIZE as u64,
			MemoryProperty::DEVICE_LOCAL,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST).unwrap();

		let shader = Shader::from_src(&dev, "compiled_shaders/speck_128_128.spv", String::from("main")).unwrap();

		let pipeline = ComputePipeline::new(&dev, &[&dev_memory], &shader, &SpecializationConstant::empty(), PUSH_CONST_SIZE as u32).unwrap();

		let cmd_queue = ComputeQueue::new(&dev).unwrap();

		cmd_queue.cmd_copy(&host_memory, &dev_memory);

		cmd_queue.cmd_set_barrier(&dev_memory,
			AccessType::HOST_WRITE,
			AccessType::SHADER_READ,
			PipelineStage::HOST,
			PipelineStage::COMPUTE_SHADER);

		let push_constant: [u8; PUSH_CONST_SIZE] =
		[
			0x00, 0x00, 0x00, 0x20,
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
			0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f
		];

		cmd_queue.update_push_constants(&pipeline, &push_constant);

		cmd_queue.cmd_bind_pipeline(&pipeline);
		cmd_queue.dispatch(1, 1, 1);

		cmd_queue.cmd_set_barrier(&dev_memory,
			AccessType::SHADER_WRITE,
			AccessType::TRANSFER_READ,
			PipelineStage::COMPUTE_SHADER,
			PipelineStage::TRANSFER);

		cmd_queue.cmd_copy(&dev_memory, &host_memory);

		cmd_queue.cmd_set_barrier(&host_memory,
			AccessType::TRANSFER_WRITE,
			AccessType::HOST_READ,
			PipelineStage::TRANSFER,
			PipelineStage::HOST);

		cmd_queue.submit().unwrap();

		let mut f = |bytes: &mut [u8]| {
			pt.read_exact(bytes).expect("Failed to read data");
		};

		host_memory.write(&mut f).unwrap();

		let now = std::time::Instant::now();

		cmd_queue.exec(PipelineStage::TRANSFER, u64::MAX).unwrap();

		let elapsed_time = now.elapsed();

		println!(
			"Running cypher with copy took {} sec, {} millisec, {} nanosec.",
			elapsed_time.as_secs(),
			elapsed_time.as_millis(),
			elapsed_time.as_nanos()
		);
	}

	#[test]
	fn data_512mb() {
		const BUFFER_SIZE: usize = 512*1024*1024;

		encrypt_data::<BUFFER_SIZE>();
	}

	#[test]
	fn data_1gb() {
		const BUFFER_SIZE: usize = 1024*1024*1024;

		encrypt_data::<BUFFER_SIZE>();
	}

	#[test]
	fn data_1_5gb() {
		const BUFFER_SIZE: usize = 1024*1024*1024 + 512*1024*1024;

		encrypt_data::<BUFFER_SIZE>();
	}

	#[test]
	fn data_2gb() {
		const BUFFER_SIZE: usize = 2*1024*1024*1024;

		encrypt_data::<BUFFER_SIZE>();
	}

	#[test]
	fn data_2_5gb() {
		const BUFFER_SIZE: usize = 2*1024*1024*1024 + 512*1024*1024;

		encrypt_data::<BUFFER_SIZE>();
	}

	#[test]
	fn data_3gb() {
		const BUFFER_SIZE: usize = 3*1024*1024*1024;

		encrypt_data::<BUFFER_SIZE>();
	}

}