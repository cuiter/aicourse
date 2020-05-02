use crate::matrix::{Float, Matrix};
use crate::regression::common::{add_zero_feature, train_gradient_descent_feature_scaling};
use crate::util::sigmoid;

/// A logistic regression problem solver.
/// It needs to be trained before it can make predictions.
#[derive(Debug, PartialEq, Clone)]
pub struct Solver<T: Float> {
    configuration: Option<Matrix<T>>,
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

fn discrete_outputs<T: Float>(outputs: &Matrix<T>) -> Matrix<T> {
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
    fn cost(
        configuration: &Matrix<T>,
        n_inputs: &Matrix<T>,
        correct_outputs: &Matrix<T>,
        regularize_param: T,
    ) -> T {
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

        let normal_cost = costsum / T::from_u32(n_inputs.get_m()).unwrap();

        if regularize_param == T::zero() {
            normal_cost
        } else {
            let configuration_squared_sum = Matrix::new(
                configuration.get_m() - 1,
                1,
                configuration
                    .get_sub_matrix(1, 0, configuration.get_m() - 1, 1)
                    .iter()
                    .map(|x| *x * *x)
                    .collect(),
            )
            .sum();
            normal_cost
                + regularize_param / T::from_u32(n_inputs.get_m() * 2).unwrap()
                    * configuration_squared_sum
        }
    }

    /// Performs gradient descent without feature scaling.
    /// For mathematical formulas, see:
    /// https://medium.com/ml-ai-study-group/31c17bca9181
    fn train_gradient_descent(
        n_inputs: &Matrix<T>,
        outputs: &Matrix<T>,
        regularize_param: T,
    ) -> Option<Matrix<T>> {
        let n_inputs_trans = n_inputs.transpose();
        let mut current_configuration = Matrix::<T>::zero(n_inputs.get_n(), 1);

        let cost_epsilon = T::from_f32(0.00001).unwrap();
        let mut learning_rate = T::from_f32(1.0).unwrap();

        loop {
            let hypothesis = Solver::<T>::run_n(&current_configuration, n_inputs);
            let gradient_simple = &(&n_inputs_trans * &(&hypothesis - outputs))
                / T::from_u32(n_inputs.get_m()).unwrap();
            let gradient = if regularize_param == T::zero() {
                gradient_simple
            } else {
                let configuration_without_first = current_configuration.with_first_zero();

                // NOTE: I have no idea whether to add or subtract the regularization product.
                // In https://www.youtube.com/watch?v=IXPgm1e0IOo&t=6m40s it is a subtraction,
                // but on the Medium article it is an addition.
                // I go with the one that gives the best results given anecdotal testing...
                &gradient_simple
                    - &(&configuration_without_first
                        * (learning_rate / T::from_u32(n_inputs.get_m()).unwrap()))
            };

            let cost =
                Solver::<T>::cost(&current_configuration, n_inputs, outputs, regularize_param);
            let new_configuration = &current_configuration - &(&gradient * learning_rate);
            let new_cost =
                Solver::<T>::cost(&new_configuration, n_inputs, outputs, regularize_param);

            if T::abs(new_cost - cost) < cost_epsilon {
                break;
            }

            if new_cost < cost {
                // Heading in the right direction.
                // After leaving the "top" of a parabola, it is usually safe
                // to speed up the learning rate.
                learning_rate *= T::from_f32(1.1).unwrap();
                current_configuration = new_configuration;
            } else {
                // If the new cost is higher than the previous cost,
                // the learning rate is too high. This makes the algorithm jump
                // over the perfect result into the wrong direction.
                // In this case, keep the old configuration and decrease the
                // learning rate significantly.
                learning_rate *= T::from_f32(0.5).unwrap();
            }
        }

        Some(current_configuration)
    }

    /// Runs the hypothesis on the inputs (with zero feature added) given the configuration.
    fn run_n(configuration: &Matrix<T>, n_inputs: &Matrix<T>) -> Matrix<T> {
        let hypothesis = (&configuration.transpose() * &n_inputs.transpose()).transpose();
        hypothesis.map(sigmoid)
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
    /// let mut solver = aicourse::regression::logistic::Solver::new();
    /// solver.train(&inputs, &outputs, 0.0);
    /// ```
    pub fn train(&mut self, inputs: &Matrix<T>, outputs: &Matrix<T>, regularize_param: T) -> bool {
        assert_eq!(inputs.get_m(), outputs.get_m());
        assert_eq!(outputs.get_n(), 1);
        assert_output_is_integer(outputs);

        let n_inputs = add_zero_feature(inputs);

        self.configuration = train_gradient_descent_feature_scaling(
            &n_inputs,
            outputs,
            regularize_param,
            Solver::<T>::train_gradient_descent,
        );

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
    /// let mut solver = aicourse::regression::logistic::Solver::new();
    /// solver.train(&inputs, &outputs, 0.0);
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

    /// Runs a prediction on the given inputs to form discrete outputs (either 0.0 or 1.0).
    /// ```
    /// let inputs = aicourse::matrix::Matrix::new(3, 1, vec![3.0,
    ///                                                       9.0,
    ///                                                      -3.0]);
    /// let outputs = aicourse::matrix::Matrix::new(3, 1, vec![1.0,
    ///                                                        0.0,
    ///                                                        1.0]);
    ///
    /// let mut solver = aicourse::regression::logistic::Solver::new();
    /// solver.train(&inputs, &outputs, 0.0);
    ///
    /// let predicted_outputs = solver.run_discrete(&inputs);
    ///
    /// assert_eq!(outputs, predicted_outputs);
    /// ```
    pub fn run_discrete(&self, inputs: &Matrix<T>) -> Matrix<T> {
        discrete_outputs(&self.run(inputs))
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
            solver.train(&tests_inputs()[i], &tests_outputs()[i], 0.0);

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

    #[test]
    fn run_discrete() {
        for i in 0..tests_inputs().len() {
            let solver = Solver {
                configuration: Some(tests_configuration()[i].clone()),
            };

            assert_eq!(
                solver.run_discrete(&tests_inputs()[i]),
                tests_outputs()[i],
                "test case {}",
                i
            );
        }
    }
}
