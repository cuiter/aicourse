use crate::logreg;
use crate::matrix::{Float, Matrix};

/// A multiple classification (One-vs-all) logistic regression problem solver.
/// It needs to be trained before it can make predictions.
#[derive(Debug, PartialEq, Clone)]
pub struct Solver<T: Float> {
    solvers: Vec<logreg::Solver<T>>,
}

fn assert_valid_classes<T: Float>(outputs: &Matrix<T>) {
    for x in outputs.iter() {
        assert!(
            *x != T::zero() && *x == T::floor(*x),
            "invalid class: {} is zero or not an integer",
            x
        );
    }
}

fn match_class<T: Float>(matrix: &Matrix<T>, class: T) -> Matrix<T> {
    Matrix::new(
        matrix.get_m(),
        matrix.get_n(),
        matrix
            .iter()
            .map(|x| {
                if *x == class {
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
        Solver { solvers: vec![] }
    }

    /// Trains the solvers with the given inputs and outputs.
    /// Returns whether the training has succeeded.
    /// If the training has not succeeded, the solvers will be uninitialized.
    /// ```
    /// let inputs = aicourse::matrix::Matrix::new(4, 2, vec![3.0, 3.0,
    ///                                                       0.0, 3.0,
    ///                                                       3.0, 0.0,
    ///                                                       0.0, 0.0]);
    /// let outputs = aicourse::matrix::Matrix::new(4, 1, vec![1.0,
    ///                                                        1.0,
    ///                                                        1.0,
    ///                                                        2.0]);
    ///
    /// let mut solver = aicourse::logregm::Solver::new();
    /// solver.train(&inputs, &outputs);
    /// ```
    pub fn train(&mut self, inputs: &Matrix<T>, outputs: &Matrix<T>) -> bool {
        assert_eq!(inputs.get_m(), outputs.get_m());
        assert_eq!(outputs.get_n(), 1);
        assert_valid_classes(outputs);

        self.solvers.clear();

        let max_class = outputs
            .iter()
            .cloned()
            .fold(T::neg_infinity(), T::max)
            .to_u32()
            .unwrap();
        let mut success = true;

        for i in 0..max_class {
            let mut solver = logreg::Solver::new();
            success &= solver.train(inputs, &match_class(outputs, T::from_u32(i + 1).unwrap()));
            self.solvers.push(solver);
        }

        if !success {
            self.solvers.clear();
        }
        success
    }

    /// Runs a prediction on every single-classifier logistic regression solver and collects the results.
    pub fn run_extended(&self, inputs: &Matrix<T>) -> Vec<Matrix<T>> {
        assert!(self.solvers.len() != 0, "solvers need to be trained first");

        self.solvers
            .iter()
            .map(|solver| solver.run(inputs))
            .collect()
    }

    /// Runs a prediction on the given inputs to form desired outputs.
    /// In this case, the output is the most likely class that the input belongs to.
    /// ```
    /// let inputs = aicourse::matrix::Matrix::new(4, 2, vec![3.0, 3.0,
    ///                                                       0.0, 3.0,
    ///                                                       3.0, 0.0,
    ///                                                       0.0, 0.0]);
    /// let outputs = aicourse::matrix::Matrix::new(4, 1, vec![1.0,
    ///                                                        1.0,
    ///                                                        1.0,
    ///                                                        2.0]);
    ///
    /// let mut solver = aicourse::logregm::Solver::new();
    /// solver.train(&inputs, &outputs);
    ///
    /// let predicted_outputs = solver.run(&inputs);
    ///
    /// assert_eq!(outputs, predicted_outputs);
    /// ```
    pub fn run(&self, inputs: &Matrix<T>) -> Matrix<T> {
        let results = self.run_extended(inputs);

        Matrix::new(
            inputs.get_m(),
            1,
            (0..inputs.get_m())
                .map(|rowidx| {
                    results
                        .iter()
                        .enumerate()
                        .map(|(classidx, result)| (classidx as u32 + 1, result[(rowidx, 0)]))
                        .fold(
                            (0u32, T::neg_infinity()),
                            |(sumidx, sumval), (curidx, curval)| {
                                if sumval >= curval {
                                    (sumidx, sumval)
                                } else {
                                    (curidx, curval)
                                }
                            },
                        )
                })
                .map(|(sumidx, _)| T::from_u32(sumidx).unwrap())
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata::logregm::*;

    #[test]
    fn train_and_run() {
        for i in 0..tests_inputs().len() {
            let mut solver = Solver::<f64>::new();
            solver.train(&tests_inputs()[i], &tests_outputs()[i]);

            let outputs = solver.run(&tests_inputs()[i]);

            assert_eq!(outputs, tests_outputs()[i]);
        }
    }
}
