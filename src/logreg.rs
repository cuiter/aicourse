use crate::matrix::{Float, Matrix};

/// A logistic regression problem solver.
/// It needs to be trained before it can make predictions.
#[derive(Debug, PartialEq, Clone)]
pub struct Solver<T: Float> {
    configuration: Option<Matrix<T>>,
}

/// Adds an "x0" feature to the left side of the input matrix.
/// The "x0" feature is always assigned to the constant 1,
/// so that it can be used as a base offset.
fn add_zero_feature<T: Float>(inputs: &Matrix<T>) -> Matrix<T> {
    Matrix::one(inputs.get_m(), 1).h_concat(inputs)
}

fn assert_output_is_integer<T: Float>(outputs: &Matrix<T>) {
    for x in outputs.iter() {
        assert!(
            *x == T::zero() || *x == T::from_u8(1).unwrap(),
            "{} is not equal to 0.0 or 1.0",
            x
        );
    }
}

pub fn discrete_outputs<T: Float>(outputs: &Matrix<T>) -> Matrix<T> {
    Matrix::new(
        outputs.get_m(),
        outputs.get_n(),
        outputs
            .iter()
            .map(|x| {
                if *x >= T::from_f32(0.5).unwrap() {
                    T::from_u8(1).unwrap()
                } else {
                    T::zero()
                }
            })
            .collect(),
    )
}

impl<T: Float> Solver<T> {
    /// Creates a new Solver with an empty configuration.
    pub fn new() -> Solver<T> {
        Solver {
            configuration: None,
        }
    }

    /// Computes the "cost" (mean square error) between the calculated output
    /// and correct output given a configuration.
    fn cost(configuration: &Matrix<T>, n_inputs: &Matrix<T>, correct_outputs: &Matrix<T>) -> T {
        let outputs = Solver::<T>::run_n(configuration, n_inputs);

        let logdiff = Matrix::new(
            outputs.get_m(),
            outputs.get_n(),
            outputs
                .iter()
                .zip(correct_outputs.iter())
                .map(|(output, correct_output)| {
                    -*correct_output * T::ln(*output)
                        - (T::from_u8(1).unwrap() - *correct_output)
                            * T::ln(T::from_u8(1).unwrap() - *output)
                })
                .collect(),
        );
        let costsum = logdiff.sum();

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
    fn train_gradient_descent_feature_scaling(
        n_inputs: &Matrix<T>,
        outputs: &Matrix<T>,
    ) -> Option<Matrix<T>> {
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

    /// Runs the hypothesis on the inputs (with zero feature added) given the configuration.
    fn run_n(configuration: &Matrix<T>, n_inputs: &Matrix<T>) -> Matrix<T> {
        let hypothesis = (&configuration.transpose() * &n_inputs.transpose()).transpose();
        let sigmoid_vect = hypothesis
            .iter()
            .map(|x| T::from_u8(1).unwrap() / (T::from_u8(1).unwrap() + T::powf(T::E(), -*x)))
            .collect();
        Matrix::new(hypothesis.get_m(), hypothesis.get_n(), sigmoid_vect)
    }

    /// Returns the configuration if it is set (after at least
    /// one successful training), panics otherwise.
    pub fn get_configuration(&self) -> &Matrix<T> {
        self.configuration.as_ref().unwrap()
    }

    /// Trains the solver with the given inputs and outputs.
    /// Returns whether the training has succeeded.
    /// If the training has not succeeded, the configuration will be None.
    /// ```
    /// let inputs = aicourse::matrix::Matrix::new(5, 1, vec![3.0,
    ///                                                       4.9,
    ///                                                       5.0,
    ///                                                       9.0,
    ///                                                      -3.0]);
    /// let outputs = aicourse::matrix::Matrix::new(5, 1, vec![1.0,
    ///                                                        1.0,
    ///                                                        0.0,
    ///                                                        0.0,
    ///                                                        1.0]);
    ///
    /// let mut solver = aicourse::logreg::Solver::new();
    /// solver.train(&inputs, &outputs);
    /// ```
    pub fn train(&mut self, inputs: &Matrix<T>, outputs: &Matrix<T>) -> bool {
        assert_eq!(inputs.get_m(), outputs.get_m());
        assert_eq!(outputs.get_n(), 1);
        assert_output_is_integer(outputs);

        let n_inputs = add_zero_feature(inputs);

        self.configuration =
            Solver::<T>::train_gradient_descent_feature_scaling(&n_inputs, outputs);

        self.configuration != None
    }

    /// Runs a prediction on the given inputs to form desired outputs.
    /// ```
    /// let inputs = aicourse::matrix::Matrix::new(3, 1, vec![3.0,
    ///                                                       9.0,
    ///                                                      -3.0]);
    /// let outputs = aicourse::matrix::Matrix::new(3, 1, vec![1.0,
    ///                                                        0.0,
    ///                                                        1.0]);
    ///
    /// let mut solver = aicourse::logreg::Solver::new();
    /// solver.train(&inputs, &outputs);
    ///
    /// let predicted_outputs = solver.run(&inputs);
    ///
    /// assert!(predicted_outputs.approx_eq(&outputs, 0.1));
    /// ```
    pub fn run(&self, inputs: &Matrix<T>) -> Matrix<T> {
        assert!(
            self.configuration != None,
            "solver needs to be trained first"
        );
        let n_configuration = self.get_configuration();
        assert_eq!(inputs.get_n(), n_configuration.get_m() - 1);

        let n_inputs = add_zero_feature(inputs);

        Solver::<T>::run_n(self.get_configuration(), &n_inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata::logreg::*;

    #[test]
    fn train_gradient_descent() {
        for i in 0..tests_inputs().len() {
            let mut solver = Solver::<f64>::new();
            solver.train(&tests_inputs()[i], &tests_outputs()[i]);

            // Because classification only differentiates between above or below 0, the scaling
            // of the resulting configuration doesn't matter. To compare it, the
            // configuration needs to be normalized before comparing it with the expected
            // configuration.
            let unnormalized_configuration = solver.get_configuration();
            let configuration_factor: f64 = tests_configuration()[i].iter().next().unwrap()
                / *unnormalized_configuration.iter().next().unwrap();
            let normalized_configuration = unnormalized_configuration * configuration_factor;

            assert!(
                normalized_configuration.approx_eq(&tests_configuration()[i], 0.06),
                "test case {}\nconfiguration =\n{}\nexpected =\n{}",
                i,
                normalized_configuration,
                tests_configuration()[i]
            );
        }
    }

    #[test]
    fn run() {
        for i in 0..tests_inputs().len() {
            let solver = Solver {
                configuration: Some(tests_configuration()[i].clone()),
            };

            assert_eq!(
                discrete_outputs(&solver.run(&tests_inputs()[i])),
                tests_outputs()[i],
                "test case {}",
                i
            );
        }
    }
}
