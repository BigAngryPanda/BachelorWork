use std::io::Read;
use std::fs::File;
use std::path::Path;

pub struct Key<const N: usize> {
	i_key: [u8; N],
}

impl<const N: usize> Key<N> {
	pub fn from_file(path: &str) -> Key<N> {
		let mut raw_key = File::open(Path::new(path)).expect("Failed to read key");

		let mut buffer: [u8; N] = [0; N];

		raw_key.read_exact(&mut buffer).expect("Failed to read key from keyfile");

		Key {
			i_key: buffer
		}
	}

	pub fn data(&self) -> &[u8] {
		&self.i_key
	}
}

#[cfg(test)]
mod test {
	use crate::key_manager::Key;
	#[test]
	fn from_file_128() {
		let target: [u8; 16] =
		[
			0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
			0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f
		];

		let key = Key::<16>::from_file("speck_128_128_test.key");

		assert_eq!(key.data(), target);
	}
}