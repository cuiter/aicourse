use super::matrix::{Float, Matrix};
use std::convert::TryInto;
use std::io::{Error, ErrorKind};

#[derive(Copy, Clone)]
enum DataType {
    U8,
    I8,
    I16,
    I32,
    F32,
    F64,
}

#[derive(Copy, Clone)]
struct MagicNumber {
    data_type: DataType,
    dimensions: u8,
}

fn data_type_size(data_type: DataType) -> usize {
    match data_type {
        DataType::U8 => 1,
        DataType::I8 => 1,
        DataType::I16 => 2,
        DataType::I32 => 4,
        DataType::F32 => 4,
        DataType::F64 => 8,
    }
}

fn read_u32(data: &Vec<u8>, index: usize) -> Result<u32, Error> {
    if index > data.len() - 4 {
        return Err(Error::new(ErrorKind::InvalidData, "index is out of range"));
    }
    let word: [u8; 4] = data[index..(index + 4)].try_into().unwrap();
    Ok(u32::from_be_bytes(word))
}

fn read_element<T: Float>(data: &Vec<u8>, data_type: DataType, index: usize) -> Result<T, Error> {
    let error = Error::new(ErrorKind::InvalidData, "data is corrupted");
    match data_type {
        DataType::U8 => T::from_u8(data[index]).ok_or(error),
        DataType::I8 => T::from_i8(i8::from_be_bytes(
            data[index..(index + 1)].try_into().unwrap(),
        ))
        .ok_or(error),
        DataType::I16 => T::from_i16(i16::from_be_bytes(
            data[index..(index + 2)].try_into().unwrap(),
        ))
        .ok_or(error),
        DataType::I32 => T::from_i32(i32::from_be_bytes(
            data[index..(index + 4)].try_into().unwrap(),
        ))
        .ok_or(error),
        DataType::F32 => T::from_f32(f32::from_bits(u32::from_be_bytes(
            data[index..(index + 4)].try_into().unwrap(),
        )))
        .ok_or(error),
        DataType::F64 => T::from_f64(f64::from_bits(u64::from_be_bytes(
            data[index..(index + 8)].try_into().unwrap(),
        )))
        .ok_or(error),
    }
}

fn read_magic_number(data: &Vec<u8>) -> Result<MagicNumber, Error> {
    if data.len() < 9 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "data is too small to be valid",
        ));
    }
    if data[0] != 0 || data[1] != 0 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "magic number prefix is invalid",
        ));
    }

    let data_type = match data[2] {
        0x08 => DataType::U8,
        0x09 => DataType::I8,
        0x0B => DataType::I16,
        0x0C => DataType::I32,
        0x0D => DataType::F32,
        0x0E => DataType::F64,
        _ => {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "magic number type is invalid",
            ))
        }
    };
    let dimensions = match data[3] {
        0 => {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "magic number n# dimensions is invalid",
            ))
        }
        n => n,
    };

    Ok(MagicNumber {
        data_type,
        dimensions,
    })
}

fn read_dimension_sizes(data: &Vec<u8>, dimensions: u8) -> Result<Vec<usize>, Error> {
    if data.len() < 4 + dimensions as usize * 4 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "data is too small to be valid",
        ));
    }

    let mut dimension_sizes: Vec<usize> = Vec::with_capacity(dimensions as usize);
    for d in 0..dimensions {
        dimension_sizes.push(read_u32(&data, 4 + d as usize * 4)? as usize);
    }

    Ok(dimension_sizes)
}

fn read_matrix_data<T: Float>(
    data: &Vec<u8>,
    magic: MagicNumber,
    dimension_sizes: &Vec<usize>,
) -> Result<Matrix<T>, Error> {
    let expected_data_len = 4
        + magic.dimensions as usize * 4
        + data_type_size(magic.data_type) * dimension_sizes.iter().product::<usize>();
    if data.len() != expected_data_len {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "data length does not match. expected: {}, got: {}",
                expected_data_len,
                data.len()
            ),
        ));
    }

    let n_rows = dimension_sizes[0];
    let n_cols: usize = dimension_sizes[1..].iter().product();

    let mut matrix = Matrix::zero(n_rows as u32, n_cols as u32);
    for m in 0..n_rows {
        for n in 0..n_cols {
            let index = 4
                + 4 * magic.dimensions as usize
                + data_type_size(magic.data_type) * (m * n_cols + n);
            matrix[(m as u32, n as u32)] = read_element(data, magic.data_type, index)?;
        }
    }

    Ok(matrix)
}

/// Loads the matrix data from a byte array in the IDX format.
/// For details, see: http://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html
pub fn load_idx<T: Float>(data: &Vec<u8>) -> Result<Matrix<T>, Error> {
    let magic = read_magic_number(data)?;

    let dimension_sizes = read_dimension_sizes(data, magic.dimensions)?;

    read_matrix_data(data, magic, &dimension_sizes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata::idx::*;

    #[test]
    fn load_idx_ok() {
        let inputs = data_inputs();
        let expected_outputs = matrix_outputs();
        for i in 0..inputs.len() {
            let output = load_idx(&inputs[i]).unwrap();
            assert_eq!(output, expected_outputs[i]);
        }
    }

    #[test]
    fn load_idx_error() {
        let inputs = wrong_data_inputs();
        for i in 0..inputs.len() {
            let output = load_idx::<f64>(&inputs[i]);
            assert!(output.is_err());
        }
    }
}
