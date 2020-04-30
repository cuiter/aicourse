use crate::matrix::{Float, Matrix};

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
pub fn unclassify<T: Float>(classes: &Matrix<T>) -> Matrix<T> {
    let mut max_class = T::neg_infinity();
    for m in 0..classes.get_m() {
        let class = classes[(m, 0)];
        assert_eq!(class.floor(), class, "{} is an integer", class);
        if class > max_class {
            max_class = class;
        }
    }

    let mut result = Matrix::zero(classes.get_m(), max_class.to_u32().unwrap());
    for m in 0..classes.get_m() {
        result[(m, classes[(m, 0)].to_u32().unwrap() - 1)] = T::from_u8(1).unwrap();
    }

    result
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
                unclassify(&unclassify_inputs()[i]),
                unclassify_outputs()[i],
                "test case {}",
                i
            );
        }
    }
}
