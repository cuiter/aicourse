use crate::matrix::{Float, Matrix};
use rand::Rng;

// Good constants for increasing/decreasing gradient descent learning rate values.
// Based on the "bold diver" algorithm, see:
// https://willamette.edu/~gorr/classes/cs449/momrate.html
pub const LEARNING_RATE_INCREASE: f32 = 1.05;
pub const LEARNING_RATE_DECREASE: f32 = 0.50;

/// The logistic sigmoid function.
/// If z < 0, then 0 < output < 0.5
/// If z >= 0, then 0.5 <= output < 1.0
pub fn sigmoid<T: Float>(z: T) -> T {
    T::from_u8(1).unwrap() / (T::from_u8(1).unwrap() + T::powf(T::E(), -z))
}

/// For each row, returns the 1-indexed column of the item with the highest value.
/// If two items have the same value, the lowest column is chosen.
pub fn classify<T: Float>(predictions: &Matrix<T>) -> Matrix<T> {
    let mut result = Matrix::zero(predictions.get_m(), 1);
    for m in 0..predictions.get_m() {
        let mut max_class = 1u32;
        let mut max_value = T::neg_infinity();
        for n in 0..predictions.get_n() {
            if predictions[(m, n)] > max_value {
                max_class = n + 1;
                max_value = predictions[(m, n)];
            }
        }
        result[(m, 0)] = T::from_u32(max_class).unwrap();
    }

    result
}

/// Reverses a classification into a prediction with the class having a
/// 100% probability and the rest 0%.
pub fn unclassify<T: Float>(classes: &Matrix<T>, max_class: u32) -> Matrix<T> {
    let mut result = Matrix::zero(classes.get_m(), max_class);
    for m in 0..classes.get_m() {
        let class = classes[(m, 0)].to_u32().unwrap();
        if class > max_class {
            panic!(
                "Class {} is higher than the maximum class {}",
                class, max_class
            );
        }
        result[(m, class - 1)] = T::from_u8(1).unwrap();
    }

    result
}

/// Same as unclassify, but automatically determines the number of classes
/// (highest class occuring in the dataset).
pub fn unclassify_auto<T: Float>(classes: &Matrix<T>) -> Matrix<T> {
    let mut max_class = T::neg_infinity();
    for m in 0..classes.get_m() {
        let class = classes[(m, 0)];
        assert_eq!(class.floor(), class, "{} is an integer", class);
        if class > max_class {
            max_class = class;
        }
    }

    unclassify(classes, max_class.to_u32().unwrap())
}

/// Calculates the accuracy, the amount of correct outputs / total outputs.
pub fn accuracy<T: Float>(classes: &Matrix<T>, expected_classes: &Matrix<T>) -> f32 {
    assert_eq!(classes.get_dimensions(), expected_classes.get_dimensions());
    assert_eq!(classes.get_n(), 1);
    classes
        .iter()
        .zip(expected_classes.iter())
        .map(|(a, b)| if a == b { 1.0f32 } else { 0.0f32 })
        .sum::<f32>()
        / classes.get_m() as f32
}

/// Splits inputs into batches of a specific size.
/// The last batch may be shorter if there are not enough inputs.
pub fn batch<T: Float>(inputs: &Matrix<T>, batch_size: u32) -> Vec<Matrix<T>> {
    if batch_size >= inputs.get_m() {
        return vec![inputs.clone()];
    }

    let mut result = Vec::with_capacity((inputs.get_m() / batch_size) as usize);
    for batch in 0..(inputs.get_m() / batch_size) {
        result.push(inputs.get_sub_matrix(batch * batch_size, 0, batch_size, inputs.get_n()));
    }
    if inputs.get_m() % batch_size != 0 {
        let rows_processed = (inputs.get_m() / batch_size) * batch_size;
        result.push(inputs.get_sub_matrix(
            rows_processed,
            0,
            inputs.get_m() - rows_processed,
            inputs.get_n(),
        ));
    }

    result
}

/// Swaps two rows in a matrix.
fn swap_row<T: Float>(inputs: &mut Matrix<T>, row_1: u32, row_2: u32) {
    for n in 0..inputs.get_n() {
        let temp = inputs[(row_1, n)];
        inputs[(row_1, n)] = inputs[(row_2, n)];
        inputs[(row_2, n)] = temp;
    }
}

/// Shuffles the inputs in a random order.
/// Uses an optimized Fisher-Yates shuffle algorithm.
pub fn shuffle<T: Float, R: Rng>(inputs: &Matrix<T>, rng: &mut R) -> Matrix<T> {
    let mut results = inputs.clone();
    for m in (1..=(results.get_m() - 1)).rev() {
        let other_m = rng.gen_range(0, m + 1);
        swap_row(&mut results, m, other_m);
    }
    return results;
}

/// Splits inputs by distribution (for example: 60%, 20%, 20%).
/// Distribution sizes should sum up to 1. Errs in favor of the first batch.
pub fn split<T: Float>(inputs: &Matrix<T>, distribution: &Vec<f32>) -> Vec<Matrix<T>> {
    let mut sizes: Vec<u32> = distribution
        .iter()
        .map(|x| (x * inputs.get_m() as f32) as u32)
        .collect();
    let sizes_sum: u32 = sizes.iter().sum();
    assert!(sizes_sum <= inputs.get_m());
    let leftovers: u32 = inputs.get_m() - sizes_sum;
    sizes[0] += leftovers;

    let mut results = vec![];
    let mut index = 0;
    for size in sizes.iter() {
        results.push(inputs.get_sub_matrix(index, 0, *size, inputs.get_n()));
        index += size;
    }

    results
}

/// Returns the first n rows of the matrix.
pub fn first_rows<T: Float>(matrix: &Matrix<T>, n_rows: u32) -> Matrix<T> {
    matrix.get_sub_matrix(0, 0, n_rows, matrix.get_n())
}

/// Muxes multiple matrices into one.
pub fn mux_matrices<T: Float>(matrices: &Vec<Matrix<T>>) -> Matrix<T> {
    let mut result = Matrix::zero(1, matrices.len() as u32 * 2);
    for (idx, matrix) in matrices.iter().enumerate() {
        result[(0, idx as u32 * 2)] = T::from_u32(matrix.get_m()).unwrap();
        result[(0, idx as u32 * 2 + 1)] = T::from_u32(matrix.get_n()).unwrap();
    }

    for new_matrix in matrices.iter() {
        let mut matrix = new_matrix.clone();
        if result.get_n() < matrix.get_n() {
            result = result.h_concat(&Matrix::zero(
                result.get_m(),
                matrix.get_n() - result.get_n(),
            ));
        }
        if matrix.get_n() != result.get_n() {
            matrix = matrix.h_concat(&Matrix::zero(
                matrix.get_m(),
                result.get_n() - matrix.get_n(),
            ));
        }
        result = result.v_concat(&matrix);
    }

    result
}

/// Demuxes a single matrix into multiple.
pub fn demux_matrices<T: Float>(matrix: &Matrix<T>) -> Vec<Matrix<T>> {
    // The first row contains the sizes of the matrices.
    // Example: [ 2 3 3 5 ], the first matrix has 2 rows and 3 columns, the second 3 rows and 5
    // columns.
    // The proceeding rows are the matrices, vertically concatenated.
    // Trailing columns may occur.
    let mut matrix_sizes: Vec<(u32, u32)> = vec![];
    for n in (0..matrix.get_n()).step_by(2) {
        if matrix[(0, n)] == T::zero() {
            break;
        }

        matrix_sizes.push((
            matrix[(0, n)].to_u32().unwrap(),
            matrix[(0, n + 1)].to_u32().unwrap(),
        ));
    }

    let mut matrices = vec![];
    let mut m = 1;
    for matrix_size in matrix_sizes {
        matrices.push(matrix.get_sub_matrix(m, 0, matrix_size.0, matrix_size.1));
        m += matrix_size.0;
    }

    matrices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata::util::*;

    #[test]
    fn sigmoid_correct() {
        let inputs = [-1.0, 0.0, 1.0];
        let expected_outputs = [0.26894, 0.5, 0.73106];
        for i in 0..inputs.len() {
            let output = sigmoid(inputs[i]);
            assert!(
                Matrix::new(1, 1, vec![output])
                    .approx_eq(&Matrix::<f32>::new(1, 1, vec![expected_outputs[i]]), 0.001),
                "sigmoid({}) = {} == {}",
                inputs[i],
                output,
                expected_outputs[i]
            );
        }
    }

    #[test]
    fn classify_correct() {
        for i in 0..classify_inputs().len() {
            assert_eq!(
                classify(&classify_inputs()[i]),
                classify_outputs()[i],
                "test case {}",
                i
            );
        }
    }

    #[test]
    fn unclassify_correct() {
        for i in 0..unclassify_inputs().len() {
            assert_eq!(
                unclassify_auto(&unclassify_inputs()[i]),
                unclassify_outputs()[i],
                "test case {}",
                i
            );
        }
    }

    #[test]
    fn accuracy_correct() {
        for i in 0..accuracy_inputs().len() {
            let (classes, expected_classes) = &accuracy_inputs()[i];
            assert_eq!(
                accuracy(&classes, &expected_classes),
                accuracy_outputs()[i],
                "test case {}",
                i
            );
        }
    }

    #[test]
    fn batch_correct() {
        for i in 0..batch_inputs().len() {
            assert_eq!(
                batch(&batch_inputs()[i], 4),
                batch_outputs()[i],
                "test case {}",
                i
            );
        }
    }

    #[test]
    fn split_correct() {
        for i in 0..split_inputs().len() {
            let (inputs, distribution) = &split_inputs()[i];
            assert_eq!(
                split(inputs, distribution),
                split_outputs()[i],
                "test case {}",
                i
            );
        }
    }

    #[test]
    fn swap_row_correct() {
        for i in 0..swap_row_inputs().len() {
            let (inputs, row_1, row_2) = &swap_row_inputs()[i];
            let mut outputs = inputs.clone();
            swap_row(&mut outputs, *row_1, *row_2);
            assert_eq!(outputs, swap_row_outputs()[i], "test case {}", i);
        }
    }

    #[test]
    fn shuffle_correct() {
        for i in 0..shuffle_inputs().len() {
            use rand::SeedableRng;
            let (inputs, seed) = &shuffle_inputs()[i];
            let mut rng = rand_pcg::Pcg64::seed_from_u64(*seed);
            assert_eq!(
                shuffle(&inputs, &mut rng),
                shuffle_outputs()[i],
                "test case {}",
                i
            );
        }
    }

    #[test]
    fn mux_matrices_correct() {
        for i in 0..mux_matrices_inputs().len() {
            let inputs = &mux_matrices_inputs()[i];
            assert_eq!(
                mux_matrices(&inputs),
                mux_matrices_outputs()[i],
                "test case {}",
                i
            );
        }
    }

    #[test]
    fn demux_matrices_correct() {
        for i in 0..mux_matrices_outputs().len() {
            let inputs = &mux_matrices_outputs()[i];
            assert_eq!(
                demux_matrices(&inputs),
                mux_matrices_inputs()[i],
                "test case {}",
                i
            );
        }
    }
}
