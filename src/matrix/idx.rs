use super::matrix::{Float, Matrix};
use std::convert::TryInto;
use std::io::{Error, ErrorKind, Result};

#[derive(Copy, Clone)]
pub enum DataType {
    U8,
    I8,
    I16,
    I32,
    F32,
    F64,
}

impl DataType {
    fn from_u8(code: u8) -> Result<Self> {
        match code {
            0x08 => Ok(DataType::U8),
            0x09 => Ok(DataType::I8),
            0x0B => Ok(DataType::I16),
            0x0C => Ok(DataType::I32),
            0x0D => Ok(DataType::F32),
            0x0E => Ok(DataType::F64),
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "magic number type is invalid",
                ))
            }
        }
    }

    fn to_u8(&self) -> u8 {
        match self {
            DataType::U8 => 0x08,
            DataType::I8 => 0x09,
            DataType::I16 => 0x0B,
            DataType::I32 => 0x0C,
            DataType::F32 => 0x0D,
            DataType::F64 => 0x0E,
        }
    }
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

fn write_element<T: Float>(data: &mut Vec<u8>, value: T, data_type: DataType) {
    match data_type {
        DataType::U8 => data.push(value.to_u8().unwrap()),
        DataType::I8 => data.push(value.to_i8().unwrap() as u8),
        DataType::I16 => data.extend_from_slice(&value.to_i16().unwrap().to_be_bytes()),
        DataType::I32 => data.extend_from_slice(&value.to_i32().unwrap().to_be_bytes()),
        DataType::F32 => data.extend_from_slice(&value.to_f32().unwrap().to_bits().to_be_bytes()),
        DataType::F64 => data.extend_from_slice(&value.to_f64().unwrap().to_bits().to_be_bytes()),
    }
}

fn read_u32(data: &Vec<u8>, index: usize) -> Result<u32> {
    if index > data.len() - 4 {
        return Err(Error::new(ErrorKind::InvalidData, "index is out of range"));
    }
    let word: [u8; 4] = data[index..(index + 4)].try_into().unwrap();
    Ok(u32::from_be_bytes(word))
}

fn read_element<T: Float>(element_data: &[u8], data_type: DataType) -> T {
    let convert_result = match data_type {
        DataType::U8 => T::from_u8(element_data[0]),
        DataType::I8 => T::from_i8(i8::from_be_bytes(element_data.try_into().unwrap())),
        DataType::I16 => T::from_i16(i16::from_be_bytes(element_data.try_into().unwrap())),
        DataType::I32 => T::from_i32(i32::from_be_bytes(element_data.try_into().unwrap())),
        DataType::F32 => T::from_f32(f32::from_bits(u32::from_be_bytes(
            element_data.try_into().unwrap(),
        ))),
        DataType::F64 => T::from_f64(f64::from_bits(u64::from_be_bytes(
            element_data.try_into().unwrap(),
        ))),
    };
    // For performance reasons, don't return a Result or panic.
    // Speculation: If the parsing failed, which could only happen
    // on a floating-point number, the result can be set to NaN.
    convert_result.unwrap_or(T::nan())
}

fn read_elements<T: Float>(data: &Vec<u8>, magic: MagicNumber, n_items: usize) -> Vec<T> {
    let mut elements = Vec::with_capacity(n_items);
    let elements_data = &data[(4 + 4 * magic.dimensions as usize)..];

    for element_data in elements_data.chunks(data_type_size(magic.data_type)) {
        elements.push(read_element(element_data, magic.data_type));
    }

    elements
}

fn read_magic_number(data: &Vec<u8>) -> Result<MagicNumber> {
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

    let data_type = DataType::from_u8(data[2])?;
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

fn read_dimension_sizes(data: &Vec<u8>, dimensions: u8) -> Result<Vec<usize>> {
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
) -> Result<Matrix<T>> {
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

    let elements = read_elements(data, magic, n_rows * n_cols);

    let matrix = Matrix::new(n_rows as u32, n_cols as u32, elements);

    Ok(matrix)
}

/// Loads the matrix data from a byte array in the IDX format.
/// For details, see: http://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html
pub fn load_idx<T: Float>(data: &Vec<u8>) -> Result<Matrix<T>> {
    let magic = read_magic_number(data)?;

    let dimension_sizes = read_dimension_sizes(data, magic.dimensions)?;

    read_matrix_data(data, magic, &dimension_sizes)
}

/// Writes the matrix data to a byte array in the IDX format.
pub fn write_idx<T: Float>(matrix: &Matrix<T>, data_type: DataType) -> Vec<u8> {
    let mut data = vec![
        0x00, 0x00,
        data_type.to_u8(),
        2 // Two-dimensional matrix.
    ];
    data.extend_from_slice(&matrix.get_m().to_be_bytes());
    data.extend_from_slice(&matrix.get_n().to_be_bytes());
    for value in matrix.iter() {
        write_element(&mut data, *value, data_type);
    }
    data
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

    #[test]
    fn write_idx_correct() {
        // Re-using data from other tests.
        use crate::testdata::{
            util::{classify_outputs, batch_inputs},
        };
        let inputs: Vec<Matrix<f64>> =
            classify_outputs().clone().into_iter()
            .chain(batch_inputs().clone())
            .collect();
        for i in 0..inputs.len() {
            for data_type in [DataType::U8, DataType::I8, DataType::I16, DataType::I32, DataType::F32, DataType::F64].iter() {
                let written_idx = write_idx(&inputs[i], *data_type);
                let output = load_idx::<f64>(&written_idx).unwrap();
                assert_eq!(inputs[i], output);
            }
        }
    }

    #[test]
    fn write_idx_correct_f64() {
        // Re-using data from other tests.
        use crate::testdata::{
            util::{classify_inputs},
            logreg, logregm
        };
        let inputs: Vec<Matrix<f64>> =
            classify_inputs().clone().into_iter()
            .chain(logreg::tests_inputs().clone())
            .chain(logregm::tests_inputs().clone())
            .collect();

        for i in 0..inputs.len() {
            let written_idx = write_idx(&inputs[i], DataType::F64);
            let output = load_idx::<f64>(&written_idx).unwrap();
            assert_eq!(inputs[i], output);
        }
    }
}
