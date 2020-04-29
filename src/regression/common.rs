use crate::matrix::{Float, Matrix};

/// The type of method to use when solving a regression problem.
/// Gradient descent is an iterative optimisation algorithm.
/// Normal equation solves the problem by inverting matrices.
pub enum SolveMethod {
    GradientDescent,
    NormalEquation,
}

/// Adds an "x0" feature to the left side of the input matrix.
/// The "x0" feature is always assigned to the constant 1,
/// so that it can be used as a base offset.
pub fn add_zero_feature<T: Float>(inputs: &Matrix<T>) -> Matrix<T> {
    Matrix::one(inputs.get_m(), 1).h_concat(inputs)
}

/// Performs gradient descent with feature scaling.
/// Results in more accurate and quicker convergence.
pub fn train_gradient_descent_feature_scaling<T, F>(
    n_inputs: &Matrix<T>,
    outputs: &Matrix<T>,
    regularize_param: T,
    train_gradient_descent: F,
) -> Option<Matrix<T>>
where
    T: Float,
    F: Fn(&Matrix<T>, &Matrix<T>, T) -> Option<Matrix<T>>,
{
    let mut inputs_scale = Matrix::one(1, n_inputs.get_n());
    for n in 0..n_inputs.get_n() {
        let mut max_value = T::zero();
        for m in 0..n_inputs.get_m() {
            if T::abs(n_inputs[(m, n)]) > T::abs(max_value) {
                max_value = n_inputs[(m, n)];
            }
        }
        if max_value != T::zero() {
            inputs_scale[(0, n)] = T::from_f32(1.0).unwrap() / max_value;
        }
    }

    let mut scaled_n_inputs = n_inputs.clone();
    for n in 0..n_inputs.get_n() {
        for m in 0..n_inputs.get_m() {
            scaled_n_inputs[(m, n)] = scaled_n_inputs[(m, n)] * inputs_scale[(0, n)];
        }
    }

    let mut configuration = (train_gradient_descent)(&scaled_n_inputs, outputs, regularize_param)?;

    for m in 0..configuration.get_m() {
        configuration[(m, 0)] = configuration[(m, 0)] * inputs_scale[(0, m)];
    }

    Some(configuration)
}
