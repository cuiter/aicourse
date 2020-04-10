use crate::matrix::{Matrix, Float};
use crate::linreg;
use crate::linreg::SolveMethod;

#[derive(Debug, PartialEq, Clone)]
pub struct Solver<T : Float, F : Fn(&Matrix<T>) -> Matrix<T>> {
    solver: Option<linreg::Solver<T>>,
    row_transform: F
}

// TODO: Add documentation
// TODO: Add doctests
// TODO: Add more test cases

impl <T : Float, F : Fn(&Matrix<T>) -> Matrix<T>> Solver<T, F> {
    fn transform_inputs(row_transform: &F, inputs: &Matrix<T>) -> Matrix<T> {
        let test_row = row_transform(&inputs.get_row(0));

        let data: Vec<T> = (0..inputs.get_m())
            .map(|row_idx| inputs.get_row(row_idx))
            .map(|row_data| row_transform(&row_data).iter().cloned().collect::<Vec<T>>())
            .flatten()
            .collect();

        Matrix::new(inputs.get_m(), test_row.get_n(),
            data)
    }

    pub fn new(row_transform: F) -> Solver<T, F> {
        Solver { solver: None, row_transform: row_transform }
    }

    pub fn get_configuration(&self) -> &Matrix<T> {
        return self.solver.as_ref().unwrap().get_configuration();
    }

    pub fn train(&mut self, inputs: &Matrix<T>, outputs: &Matrix<T>, method: SolveMethod) -> bool {
        let mut solver = linreg::Solver::<T>::new();
        let success = solver.train(&Solver::transform_inputs(&self.row_transform, inputs), outputs, method);
        self.solver = if success { Some(solver) } else { None };
        success
    }

    pub fn run(&self, inputs: &Matrix<T>) -> Matrix<T> {
        assert!(self.solver != None, "solver needs to be trained first");
        self.solver.as_ref().unwrap().run(&Solver::transform_inputs(&self.row_transform, inputs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tests_inputs() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(5, 1, vec![0.0,
                                    1.0,
                                    2.0,
                                    3.0,
                                    4.0])]
    }

    fn tests_outputs() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(5, 1, vec![0.0,
                                    2.0,
                                    6.0,
                                    12.0,
                                    20.0])]
    }

    fn tests_configuration() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(3, 1, vec![0.0,
                                    1.0,
                                    1.0])]
    }

    fn tests_transformation() -> Vec<fn(&Matrix<f64>) -> Matrix<f64>> {
        vec![
            |row| Matrix::new(1, 2, vec![row[(0, 0)], row[(0, 0)] * row[(0, 0)]])
        ]
    }

    #[test]
    fn train_gradient_descent() {
        for i in 0..tests_inputs().len() {
            let mut solver = Solver::<f64, fn(&Matrix<f64>) -> Matrix<f64>>::new(tests_transformation()[i]);
            solver.train(&tests_inputs()[i], &tests_outputs()[i], SolveMethod::GradientDescent);

            assert!(solver.get_configuration().approx_eq(&tests_configuration()[i], 0.01), "test case {}\nconfiguration =\n{}\nexpected =\n{}", i, solver.get_configuration(), tests_configuration()[i]);
        }
    }

    #[test]
    fn train_normal_equation() {
        for i in 0..tests_inputs().len() {
            let mut solver = Solver::<f64, fn(&Matrix<f64>) -> Matrix<f64>>::new(tests_transformation()[i]);
            solver.train(&tests_inputs()[i], &tests_outputs()[i], SolveMethod::NormalEquation);

            assert!(solver.get_configuration().approx_eq(&tests_configuration()[i], 0.01), "test case {}\nconfiguration =\n{}\nexpected =\n{}", i, solver.get_configuration(), tests_configuration()[i]);
        }
    }

    #[test]
    fn run() {
        for i in 0..tests_inputs().len() {
            let solver = Solver { solver: Some(linreg::Solver { configuration: Some(tests_configuration()[i].clone()) }), row_transform: tests_transformation()[i] };

            assert_eq!(solver.run(&tests_inputs()[i]), tests_outputs()[i], "test case {}", i);
        }
    }
}
