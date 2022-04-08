#[derive(Debug)]
pub enum CipherType {
	Speck128_128,
}

impl CipherType {
	pub fn key_size(&self) -> u64 {
		match self {
			CipherType::Speck128_128 => 16,
		}
	}

	pub fn block_size(&self) -> u64 {
		match self {
			CipherType::Speck128_128 => 16,
		}
	}
}