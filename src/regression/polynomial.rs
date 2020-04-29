use crate::regression::linear;
use crate::regression::common::SolveMethod;
use crate::matrix::{Float, Matrix};

/// A polynomial regression problem solver.
/// Works by transforming the input and passing it to a linear regression solver.
/// Needs to be trained before it can make predictions.
#[derive(Debug, PartialEq, Clone)]
pub struct Solver<T: Float, F: Fn(&Matrix<T>) -> Matrix<T>> {
    solver: Option<linear::Solver<T>>,
    row_transform: F,
}

impl<T: Float, F: Fn(&Matrix<T>) -> Matrix<T>> Solver<T, F> {
    /// Runs the transformation function on every row of the inputs.
    fn transform_inputs(row_transform: &F, inputs: &Matrix<T>) -> Matrix<T> {
        let test_row = row_transform(&inputs.get_row(0));

        let data: Vec<T> = (0..inputs.get_m())
            .map(|row_idx| inputs.get_row(row_idx))
            .map(|row_data| row_transform(&row_data).iter().cloned().collect::<Vec<T>>())
            .flatten()
            .collect();

        Matrix::new(inputs.get_m(), test_row.get_n(), data)
    }

    /// Creates a solver with an empty configuration and a transformation function.
    /// The transformation function takes a row of inputs and returns a row of transformed inputs.
    /// The size of the returned row always needs to be the same.
    pub fn new(row_transform: F) -> Solver<T, F> {
        Solver {
            solver: None,
            row_transform: row_transform,
        }
    }

    /// Returns the configuration of the linear regression solver.
    /// Panics if the configuration has not been set.
    pub fn get_configuration(&self) -> &Matrix<T> {
        return self.solver.as_ref().unwrap().get_configuration();
    }

    /// Trains the solver with the given inputs and outputs.
    /// The solve method can either be GradientDescent or NormalEquation.
    /// Returns whether the training has succeeded.
    /// If the training has not succeeded, the configuration will be unset.
    pub fn train(
        &mut self,
        inputs: &Matrix<T>,
        outputs: &Matrix<T>,
        method: SolveMethod,
        regularize_param: T,
    ) -> bool {
        let mut solver = linear::Solver::<T>::new();
        let success = solver.train(
            &Solver::transform_inputs(&self.row_transform, inputs),
            outputs,
            method,
            regularize_param,
        );
        self.solver = if success { Some(solver) } else { None };
        success
    }

    /// Runs a prediction on the given inputs to form desired outputs.
    /// ```
    /// let inputs = aicourse::matrix::Matrix::new(3, 2, vec![3.0, 5.0,
    ///                                                       9.0, 2.0,
    ///                                                      -3.0, 4.0]);
    /// let outputs = aicourse::matrix::Matrix::new(3, 1, vec![28.0,
    ///                                                        13.0,
    ///                                                        13.0]);
    ///
    /// let mut solver = aicourse::regression::polynomial::Solver::new(
    ///     |row| aicourse::matrix::Matrix::new(1, 2, vec![row[(0, 0)], row[(0, 1)] * row[(0, 1)]]));
    /// solver.train(&inputs, &outputs, aicourse::regression::SolveMethod::GradientDescent, 0.0);
    ///
    /// let predicted_outputs = solver.run(&inputs);
    ///
    /// assert!(predicted_outputs.approx_eq(&outputs, 0.1));
    /// ```
    pub fn run(&self, inputs: &Matrix<T>) -> Matrix<T> {
        assert!(self.solver != None, "solver needs to be trained first");
        self.solver
            .as_ref()
            .unwrap()
            .run(&Solver::transform_inputs(&self.row_transform, inputs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata::polyreg::*;

    #[test]
    fn train_gradient_descent() {
        for i in 0..tests_inputs().len() {
            let mut solver =
                Solver::<f64, fn(&Matrix<f64>) -> Matrix<f64>>::new(tests_transformation()[i]);
            solver.train(
                &tests_inputs()[i],
                &tests_outputs()[i],
                SolveMethod::GradientDescent,
                0.0,
            );

            assert!(
                solver
                    .get_configuration()
                    .approx_eq(&tests_configuration()[i], 0.09),
                "test case {}\nconfiguration =\n{}\nexpected =\n{}",
                i,
                solver.get_configuration(),
                tests_configuration()[i]
            );
        }
    }

    #[test]
    fn train_normal_equation() {
        for i in 0..tests_inputs().len() {
            let mut solver =
                Solver::<f64, fn(&Matrix<f64>) -> Matrix<f64>>::new(tests_transformation()[i]);
            solver.train(
                &tests_inputs()[i],
                &tests_outputs()[i],
                SolveMethod::NormalEquation,
                0.0,
            );

            assert!(
                solver
                    .get_configuration()
                    .approx_eq(&tests_configuration()[i], 0.01),
                "test case {}\nconfiguration =\n{}\nexpected =\n{}",
                i,
                solver.get_configuration(),
                tests_configuration()[i]
            );
        }
    }

    #[test]
    fn run() {
        for i in 0..tests_inputs().len() {
            let solver = Solver {
                solver: Some(linear::Solver {
                    configuration: Some(tests_configuration()[i].clone()),
                }),
                row_transform: tests_transformation()[i],
            };

            assert_eq!(
                solver.run(&tests_inputs()[i]),
                tests_outputs()[i],
                "test case {}",
                i
            );
        }
    }

    #[test]
    #[should_panic]
    fn run_untrained() {
        let solver = Solver::<f64, _>::new(|row| row.clone());
        solver.run(&tests_inputs()[0]);
    }

    #[test]
    #[should_panic]
    fn get_configuration_untrained() {
        let solver = Solver::<f64, _>::new(|row| row.clone());
        solver.get_configuration();
    }
}
