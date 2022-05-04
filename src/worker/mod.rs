extern crate libvktypes;

use libvktypes::instance::LibHandler;
use libvktypes::hardware::{
	HWDescription,
	MemoryProperty
};
use libvktypes::utils::filters::{
	select_hw,
	any_memory,
	is_compute_family,
	dedicated_hw
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

use std::mem::size_of;

use crate::cipher_type::CipherType;

pub struct WorkerType<'b> {
	pub shader_path: &'b str,
	pub buffer_size: u32,
	pub cipher_type: CipherType,
	pub key: &'b [u8]
}

pub struct GPUHandler<'a> {
	i_dev: LogicalDevice<'a>,
}

impl<'a> GPUHandler<'a> {
	pub fn new(lib: &LibHandler) -> GPUHandler {
		let hw_list =
		HWDescription::list(lib).expect("Failed to get list of available hardware");

		let hw_info = select_hw(hw_list.iter(), dedicated_hw, is_compute_family, any_memory)
			.expect("Failed to get device information");

		let hw_dev_ref = &hw_list[hw_info.device];

		let dev = LogicalDevice::new(lib, hw_dev_ref, hw_info.queue)
			.expect("Failed to create logical device");

		GPUHandler {
			i_dev: dev
		}
	}

	pub fn device(&self) -> &LogicalDevice<'a> {
		&self.i_dev
	}
}

pub struct GPUWorker<'a> {
	i_host_memory: Memory<'a>,
	i_device_memory: Memory<'a>,
	i_shader: Shader<'a>,
	i_pipeline: ComputePipeline<'a>,
	i_queue: ComputeQueue<'a>
}

impl<'a> GPUWorker<'a> {
	pub fn new(dev: &'a GPUHandler, create_info: &WorkerType) -> GPUWorker<'a> {
		let host_memory = Memory::new(
			dev.device(),
			create_info.buffer_size as u64,
			MemoryProperty::HOST_VISIBLE,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST
		)
		.expect("Failed to allocate host memory");

		let dev_memory = Memory::new(
			dev.device(),
			create_info.buffer_size as u64,
			MemoryProperty::DEVICE_LOCAL,
			BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST
		)
		.expect("Failed to allocate device memory");

		let shader = Shader::from_src(
			dev.device(),
			create_info.shader_path,
			String::from("main")
		)
		.expect("Failed to create shader");

		let pipeline = ComputePipeline::new(
			dev.device(),
			&[&dev_memory],
			&shader,
			&SpecializationConstant::empty(),
			(create_info.key.len() + size_of::<u32>()) as u32
		)
		.expect("Failed to allocate host memory");

		let cmd_queue = ComputeQueue::new(dev.device()).expect("Failed to allocate command buffer");

		cmd_queue.cmd_copy(&host_memory, &dev_memory);

		cmd_queue.cmd_set_barrier(
			&dev_memory,
			AccessType::HOST_WRITE,
			AccessType::SHADER_READ,
			PipelineStage::HOST,
			PipelineStage::COMPUTE_SHADER
		);

		let len_bytes: [u8; 4] = (create_info.key.len() as u32).to_le_bytes();
		let push_data: Vec<u8> = len_bytes
			.iter()
			.copied()
			.chain(create_info.key.iter().copied())
			.collect();

		cmd_queue.update_push_constants(&pipeline, push_data.as_slice());

		cmd_queue.cmd_bind_pipeline(&pipeline);

		// TODO
		cmd_queue.dispatch(1, 1, 1);

		cmd_queue.cmd_set_barrier(
			&dev_memory,
			AccessType::SHADER_WRITE,
			AccessType::TRANSFER_READ,
			PipelineStage::COMPUTE_SHADER,
			PipelineStage::TRANSFER
		);

		cmd_queue.cmd_copy(&dev_memory, &host_memory);

		cmd_queue.cmd_set_barrier(
			&host_memory,
			AccessType::TRANSFER_WRITE,
			AccessType::HOST_READ,
			PipelineStage::TRANSFER,
			PipelineStage::HOST
		);

		cmd_queue.submit().expect("Failed to submit cmd buffer");

		GPUWorker {
			i_host_memory: host_memory,
			i_device_memory: dev_memory,
			i_shader: shader,
			i_pipeline: pipeline,
			i_queue: cmd_queue
		}
	}

	pub fn copy_into(&self, buffer: &mut [u8]) {
		buffer.clone_from_slice(
			self.i_host_memory
			.read()
			.expect("Failed to access host memory")
		);
	}

	pub fn write_into(&self, buffer: &[u8]) {
		let mut f = |bytes: &mut [u8]| {
			bytes.clone_from_slice(buffer);
		};

		self.i_host_memory.write(&mut f).expect("Failed writing to host memory");
	}

	pub fn apply<F>(&self, f: &mut F)
		where F: FnMut(&mut [u8])
	{
		self.i_host_memory.write(f).expect("Failed to process host memory");
	}

	pub fn exec(&self) {
		self.i_queue.exec(PipelineStage::TRANSFER, u64::MAX).unwrap();
	}
}