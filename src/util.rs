use crate::matrix::{Matrix, Float};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_correct() {
        let inputs = [ -1.0, 0.0, 1.0 ];
        let expected_outputs = [ 0.26894, 0.5, 0.73106 ];
        for i in 0..inputs.len() {
            let output = sigmoid(inputs[i]);
            assert!(Matrix::new(1, 1, vec![output]).approx_eq(&Matrix::<f32>::new(1, 1, vec![expected_outputs[i]]), 0.001), "sigmoid({}) = {} == {}", inputs[i], output, expected_outputs[i]);
        }
    }

    #[test]
    fn classify_correct() {
        let input = Matrix::new(3, 4, vec![ 1.0, 0.5, 0.8, 0.0,
                                            0.0, 0.2, 0.1, 0.4,
                                            0.9, 0.2, 1.0, 1.0 ]);
        let expected_outputs = Matrix::new(3, 1, vec![ 1.0,
                                                       4.0,
                                                       3.0 ]);
        assert_eq!(classify(&input), expected_outputs);
    }
}
