use crate::matrix::{Matrix, Float};

pub enum SolveMethod {
    GradientDescent,
    NormalEquation
}

/// A linear regression problem solver.
/// It needs to be trained before it can make predictions.
#[derive(Debug, PartialEq, Clone)]
pub struct Solver<T : Float> {
    configuration: Option<Matrix<T>>
}

/// Adds an "x0" feature to the left side of the input matrix.
/// The "x0" feature is always assigned to the constant 1,
/// so that it can be used as a base offset.
fn add_zero_feature<T : Float>(inputs: &Matrix<T>) -> Matrix<T> {
    Matrix::one(inputs.get_m(), 1).h_concat(inputs)
}

impl<T : Float> Solver<T> {
    /// Creates a new Solver with an empty configuration.
    pub fn new() -> Solver<T> {
        Solver { configuration: None }
    }

    /// Computes the "cost" (mean square error) between the calculated output
    /// and correct output given a configuration.
    fn cost(configuration: &Matrix<T>, n_inputs: &Matrix<T>, correct_outputs: &Matrix<T>) -> T {
        let outputs = Solver::<T>::run_n(configuration, n_inputs);

        let diff = &outputs - correct_outputs;
        let diffsquared = Matrix::new(diff.get_m(), diff.get_n(), diff.iter().map(|x| *x * *x).collect());
        let costsum = diffsquared.sum();

        costsum / T::from_u32(n_inputs.get_m() * 2).unwrap()
    }

    /// Performs gradient descent without feature scaling.
    fn train_gradient_descent(n_inputs: &Matrix<T>, outputs: &Matrix<T>) -> Option<Matrix<T>> {
        let n_inputs_trans = n_inputs.transpose();
        let mut current_configuration = Matrix::<T>::zero(n_inputs.get_n(), 1);

        let cost_epsilon = T::from_f32(0.00001).unwrap();
        let mut learning_rate = T::from_f32(1.0).unwrap();

        loop {
            let hypothesis = Solver::<T>::run_n(&current_configuration, n_inputs);
            let loss = &hypothesis - outputs;
            let cost = Solver::<T>::cost(&current_configuration, n_inputs, outputs);
            let gradient = &(&n_inputs_trans * &loss) / T::from_u32(n_inputs.get_m()).unwrap();

            let new_configuration = &current_configuration - &(&gradient * learning_rate);
            let new_cost = Solver::<T>::cost(&new_configuration, n_inputs, outputs);

            if T::abs(new_cost - cost) < cost_epsilon {
                break;
            }

            if new_cost < cost {
                // Heading in the right direction.
                // After leaving the "top" of a parabola, it is usually safe
                // to speed up the learning rate.
                learning_rate = learning_rate * T::from_f32(1.1).unwrap();
                current_configuration = new_configuration;
            } else {
                // If the new cost is higher than the previous cost,
                // the learning rate is too high. This makes the algorithm jump
                // over the perfect result into the wrong direction.
                // In this case, keep the old configuration and decrease the
                // learning rate significantly.
                learning_rate = learning_rate * T::from_f32(0.5).unwrap();
            }
        }

        Some(current_configuration)
    }

    /// Performs gradient descent with feature scaling.
    /// Results in more accurate and quicker convergence.
    fn train_gradient_descent_feature_scaling(n_inputs: &Matrix<T>, outputs: &Matrix<T>) -> Option<Matrix<T>> {
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

        let mut configuration = Solver::<T>::train_gradient_descent(&scaled_n_inputs, outputs)?;

        for m in 0..configuration.get_m() {
            configuration[(m, 0)] = configuration[(m, 0)] * inputs_scale[(0, m)];
        }

        Some(configuration)
    }

    /// Computes an optimal configuration by inverting a matrix.
    /// This may fail because the matrix may not be invertible.
    /// In this case, a technique called SVD is needed to compute a result,
    /// but is not yet implemented.
    fn train_normal_equation(n_inputs: &Matrix<T>, outputs: &Matrix<T>) -> Option<Matrix<T>> {
        let a = &n_inputs.transpose() * n_inputs;
        let ainv = &a.pinv()?; // TODO: Figure out SVD in Matrix::pinv
        let int = ainv * &n_inputs.transpose();
        Some(&int * outputs)
    }

    /// Runs the hypothesis on the inputs (with zero feature added) given the configuration.
    fn run_n(configuration: &Matrix<T>, n_inputs: &Matrix<T>) -> Matrix<T> {
        (&configuration.transpose() * &n_inputs.transpose()).transpose()
    }

    /// Returns the configuration if it is set (after at least
    /// one successful training), panics otherwise.
    pub fn get_configuration(&self) -> &Matrix<T> {
        self.configuration.as_ref().unwrap()
    }

    /// Trains the solver with the given inputs and outputs.
    /// The solve method can either be GradientDescent or NormalEquation.
    /// Returns whether the training has succeeded.
    /// If the training has not succeeded, the configuration will be None.
    /// ```
    /// let inputs = aicourse::matrix::Matrix::new(3, 2, vec![3.0, 5.0,
    ///                                                       9.0, 2.0,
    ///                                                      -3.0, 4.0]);
    /// let outputs = aicourse::matrix::Matrix::new(3, 1, vec![35.0,
    ///                                                        92.0,
    ///                                                       -26.0]);
    ///
    /// let correct_configuration = aicourse::matrix::Matrix::new(3, 1, vec![0.0,
    ///                                                                      10.0,
    ///                                                                      1.0]);
    ///
    /// let mut solver = aicourse::linreg::Solver::new();
    /// solver.train(&inputs, &outputs, aicourse::linreg::SolveMethod::GradientDescent);
    /// assert!(solver.get_configuration().approx_eq(&correct_configuration, 0.1));
    ///
    /// solver.train(&inputs, &outputs, aicourse::linreg::SolveMethod::NormalEquation);
    /// assert!(solver.get_configuration().approx_eq(&correct_configuration, 0.1));
    /// ```
    pub fn train(&mut self, inputs: &Matrix<T>, outputs: &Matrix<T>, method: SolveMethod) -> bool {
        assert_eq!(inputs.get_m(), outputs.get_m());
        assert_eq!(outputs.get_n(), 1);

        let n_inputs = add_zero_feature(inputs);

        self.configuration = match method {
            SolveMethod::NormalEquation => Solver::<T>::train_normal_equation(&n_inputs, outputs),
            SolveMethod::GradientDescent => Solver::<T>::train_gradient_descent_feature_scaling(&n_inputs, outputs),
        };

        self.configuration != None
    }

    /// Runs a prediction on the given inputs to form desired outputs.
    /// ```
    /// let inputs = aicourse::matrix::Matrix::new(3, 2, vec![3.0, 5.0,
    ///                                                       9.0, 2.0,
    ///                                                      -3.0, 4.0]);
    /// let outputs = aicourse::matrix::Matrix::new(3, 1, vec![35.0,
    ///                                                        92.0,
    ///                                                       -26.0]);
    ///
    /// let mut solver = aicourse::linreg::Solver::new();
    /// solver.train(&inputs, &outputs, aicourse::linreg::SolveMethod::GradientDescent);
    ///
    /// let predicted_outputs = solver.run(&inputs);
    ///
    /// assert!(predicted_outputs.approx_eq(&outputs, 0.1));
    /// ```
    pub fn run(&self, inputs: &Matrix<T>) -> Matrix<T> {
        assert!(self.configuration != None, "solver needs to be trained first");
        let n_configuration = self.get_configuration();
        assert_eq!(inputs.get_n(), n_configuration.get_m() - 1);

        let n_inputs = add_zero_feature(inputs);

        Solver::<T>::run_n(self.get_configuration(), &n_inputs)
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    fn tests_inputs() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(9, 2, vec![50.0, 6.0,
                                    100.0, 5.0,
                                    200.0, 3.0,
                                    400.0, 0.0,
                                    500.0, 0.0,
                                    600.0, 500.0,
                                    700.0, 0.0,
                                    800.0, 0.0,
                                   -100.0, 0.0]),
             Matrix::new(3, 1, vec![0.0,
                                    1.0,
                                    2.0])]
    }

    fn tests_outputs() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(9, 1, vec![1512.0,
                                    2010.0,
                                    3006.0,
                                    5000.0,
                                    6000.0,
                                    8000.0,
                                    8000.0,
                                    9000.0,
                                       0.0]),
             Matrix::new(3, 1, vec![0.0,
                                    1.0,
                                    2.0])]
    }

    fn tests_configuration() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(3, 1, vec![1000.0,
                                    10.0,
                                    2.0]),
             Matrix::new(2, 1, vec![0.0,
                                    1.0])]
    }

    #[test]
    fn train_gradient_descent() {
        for i in 0..tests_inputs().len() {
            let mut solver = Solver::<f64>::new();
            solver.train(&tests_inputs()[i], &tests_outputs()[i], SolveMethod::GradientDescent);

            assert!(solver.get_configuration().approx_eq(&tests_configuration()[i], 0.01), "test case {}\nconfiguration =\n{}\nexpected =\n{}", i, solver.get_configuration(), tests_configuration()[i]);
        }
    }

    #[test]
    fn train_normal_equation() {
        for i in 0..tests_inputs().len() {
            let mut solver = Solver::<f64>::new();
            solver.train(&tests_inputs()[i], &tests_outputs()[i], SolveMethod::NormalEquation);

            assert!(solver.get_configuration().approx_eq(&tests_configuration()[i], 0.01), "test case {}\nconfiguration =\n{}\nexpected =\n{}", i, solver.get_configuration(), tests_configuration()[i]);
        }
    }

    #[test]
    fn run() {
        for i in 0..tests_inputs().len() {
            let solver = Solver { configuration: Some(tests_configuration()[i].clone()) };

            assert_eq!(solver.run(&tests_inputs()[i]), tests_outputs()[i], "test case {}", i);
        }
    }
}
