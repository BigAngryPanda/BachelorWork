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

// Special for NVIDIA GeForce GTX 750 Ti
static DEV_INDEX: usize = 0;
static QUEUE_FAMILY_INDEX: usize = 0;
//static DEV_MEMORY_INDEX: usize = 7;
//static HOST_MEMORY_INDEX: usize = 9;

static BUFFER_SIZE: u64 = 32;

const PUSH_CONST_SIZE: usize = 20;

fn main() {
	// Special for NVIDIA GeForce GTX 750 Ti
	let vk_lib = LibHandler::new(1, 2, 0, true).unwrap();

	let hw_list = HWDescription::list(&vk_lib).unwrap();
/*
	for (i, hw) in hw_list.iter().enumerate() {
		print!("\nDevice number {}\n", i);
		print!("{}", hw);
	}
*/
	let dev = LogicalDevice::new(&vk_lib, &hw_list[DEV_INDEX], QUEUE_FAMILY_INDEX).unwrap();

	let host_memory = Memory::new(&dev, BUFFER_SIZE,
		MemoryProperty::HOST_VISIBLE,
		BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST).unwrap();

	let f = |bytes: &mut [u8]| {
		bytes.fill(1);
	};

	host_memory.write(f).unwrap();

	let dev_memory = Memory::new(&dev, BUFFER_SIZE,
		MemoryProperty::DEVICE_LOCAL,
		BufferType::STORAGE_BUFFER | BufferType::TRANSFER_SRC | BufferType::TRANSFER_DST).unwrap();

	let shader = Shader::from_src(&dev, "src/shaders/comp.spv", String::from("main")).unwrap();

	let pipeline = ComputePipeline::new(&dev, &[&dev_memory], &shader, &SpecializationConstant::empty(), PUSH_CONST_SIZE as u32).unwrap();

	let cmd_queue = ComputeQueue::new(&dev).unwrap();

	cmd_queue.cmd_copy(&host_memory, &dev_memory);

	cmd_queue.cmd_set_barrier(&dev_memory,
		AccessType::HOST_WRITE,
		AccessType::SHADER_READ,
		PipelineStage::HOST,
		PipelineStage::COMPUTE_SHADER);

	let push_constant: [u8; PUSH_CONST_SIZE] =[
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

	let p = |bytes: &mut [u8]| {
		println!("{:02x?}", bytes);
	};

	host_memory.write(p).unwrap();

//	let data: &[u8] = host_memory.read().unwrap();

//	print!("{:?}", data);
}
