use crate::matrix::{Matrix, Float};

pub enum SolveMethod {
    GradientDescent,
    NormalEquation
}

#[derive(Debug, PartialEq, Clone)]
pub struct Solver<T : Float> {
    configuration: Option<Matrix<T>>
}

fn add_zero_feature<T : Float>(inputs: &Matrix<T>) -> Matrix<T> {
    Matrix::one(inputs.get_m(), 1).h_concat(inputs)
}

impl<T : Float> Solver<T> {
    pub fn new() -> Solver<T> {
        Solver { configuration: None }
    }


    fn cost(configuration: &Matrix<T>, n_inputs: &Matrix<T>, correct_outputs: &Matrix<T>) -> T {
        let outputs = Solver::<T>::run_n(configuration, n_inputs);

        let diff = &outputs - correct_outputs;
        let diffsquared = Matrix::new(diff.get_m(), diff.get_n(), diff.iter().map(|x| *x * *x).collect());
        let costsum = diffsquared.sum();

        costsum / T::from_u32(n_inputs.get_m() * 2).unwrap()
    }

    fn train_gradient_descent(n_inputs: &Matrix<T>, outputs: &Matrix<T>) -> Option<Matrix<T>> {
        let n_inputs_trans = n_inputs.transpose();
        let mut current_configuration = Matrix::<T>::zero(n_inputs.get_n(), 1);

        let cost_epsilon = T::from_f32(0.001).unwrap();
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
                current_configuration = new_configuration;
                learning_rate = learning_rate * T::from_f32(1.1).unwrap();
            } else {
                learning_rate = learning_rate * T::from_f32(0.5).unwrap();
            }
        }

        Some(current_configuration)
    }

    fn train_normal_equation(n_inputs: &Matrix<T>, outputs: &Matrix<T>) -> Option<Matrix<T>> {
        let a = &n_inputs.transpose() * n_inputs;
        let ainv = &a.pinv()?; // TODO: Figure out SVD in Matrix::pinv
        let int = ainv * &n_inputs.transpose();
        Some(&int * outputs)
    }

    fn run_n(configuration: &Matrix<T>, n_inputs: &Matrix<T>) -> Matrix<T> {
        (&configuration.transpose() * &n_inputs.transpose()).transpose()
    }

    pub fn get_configuration(&self) -> &Matrix<T> {
        self.configuration.as_ref().unwrap()
    }

    pub fn train(&mut self, inputs: &Matrix<T>, outputs: &Matrix<T>, method: SolveMethod) -> bool {
        assert_eq!(inputs.get_m(), outputs.get_m());
        assert_eq!(outputs.get_n(), 1);

        let n_inputs = add_zero_feature(inputs);

        self.configuration = match method {
            SolveMethod::NormalEquation => Solver::<T>::train_normal_equation(&n_inputs, outputs),
            SolveMethod::GradientDescent => Solver::<T>::train_gradient_descent(&n_inputs, outputs),
        };

        self.configuration != None
    }

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
        vec![Matrix::new(9, 2, vec![50.0, 6.0,
                                    100.0, 5.0,
                                    200.0, 3.0,
                                    400.0, 0.0,
                                    500.0, 0.0,
                                    600.0, 0.0,
                                    700.0, 0.0,
                                    800.0, 0.0,
                                   -100.0, 0.0]),
             Matrix::new(3, 1, vec![0.0,
                                    1.0,
                                    2.0])]
    }

    fn tests_outputs() -> Vec<Matrix<f64>> {
        vec![Matrix::new(9, 1, vec![1512.0,
                                    2010.0,
                                    3006.0,
                                    5000.0,
                                    6000.0,
                                    7000.0,
                                    8000.0,
                                    9000.0,
                                       0.0]),
             Matrix::new(3, 1, vec![0.0,
                                    1.0,
                                    2.0])]
    }

    fn tests_configuration() -> Vec<Matrix<f64>> {
        vec![Matrix::new(3, 1, vec![1000.0,
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
