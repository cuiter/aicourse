use crate::matrix::{Matrix};

pub mod linreg {
    use super::*;
    pub fn tests_inputs() -> Vec<Matrix<f64>> {
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

    pub fn tests_outputs() -> Vec<Matrix<f64>> {
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

    pub fn tests_configuration() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(3, 1, vec![1000.0,
                                    10.0,
                                    2.0]),
             Matrix::new(2, 1, vec![0.0,
                                    1.0])]
    }
}

pub mod logreg {
    use super::*;

    pub fn tests_inputs() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(7, 2, vec![50.0, 0.0,
                                    100.0, 199.5,
                                    100.0, 200.0,
                                    150.0, 99.5,
                                    200.0, -0.5,
                                    200.0, 0.0,
                                    250.0, 0.0]),
             Matrix::new(9, 1, vec![50.0,
                                    100.0,
                                    200.0,
                                    490.0,
                                    500.0,
                                    600.0,
                                    700.0,
                                    800.0,
                                   -100.0]),
        ]
    }

    pub fn tests_outputs() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(7, 1, vec![0.0,
                                    0.0,
                                    1.0,
                                    0.0,
                                    0.0,
                                    1.0,
                                    1.0]),
             Matrix::new(9, 1, vec![0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    0.0]),
        ]
    }

    pub fn tests_configuration() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(3, 1, vec![-200.0,
                                    1.0,
                                    0.5]),
             Matrix::new(2, 1, vec![-500.0,
                                    1.0]),
        ]
    }
}

pub mod logregm {
    use super::*;

    pub fn tests_inputs() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(4, 2, vec![-5.0, -5.0,
                                    5.0, -5.0,
                                    -5.0, 5.0,
                                    5.0, 5.0]),
             Matrix::new(4, 2, vec![5.0, 0.0,
                                    0.0, 5.0,
                                    -5.0, 0.0,
                                    0.0, -5.0]),
        ]
    }

    pub fn tests_outputs() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(4, 1, vec![1.0,
                                    2.0,
                                    3.0,
                                    4.0]),
             Matrix::new(4, 1, vec![1.0,
                                    2.0,
                                    3.0,
                                    4.0]),
        ]
    }
}

pub mod polyreg {
    use super::*;

    pub fn tests_inputs() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(6, 1, vec![-1.0,
                                    0.0,
                                    1.0,
                                    2.0,
                                    3.0,
                                    4.0]),
             Matrix::new(8, 1, vec![0.0,
                                    1.0,
                                    4.0,
                                    25.0,
                                    36.0,
                                    49.0,
                                    81.0,
                                    144.0])]
    }

    pub fn tests_outputs() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(6, 1, vec![100.0,
                                    100.0,
                                    102.0,
                                    106.0,
                                    112.0,
                                    120.0]),
             Matrix::new(8, 1, vec![-100.0,
                                    -100.0,
                                    -98.0,
                                    -80.0,
                                    -70.0,
                                    -58.0,
                                    -28.0,
                                    32.0])]
    }

    pub fn tests_configuration() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(3, 1, vec![100.0,
                                    1.0,
                                    1.0]),
             Matrix::new(3, 1, vec![-100.0,
                                    1.0,
                                    -1.0])]
    }

    pub fn tests_transformation() -> Vec<fn(&Matrix<f64>) -> Matrix<f64>> {
        vec![
            |row| Matrix::new(1, 2, vec![row[(0, 0)], row[(0, 0)] * row[(0, 0)]]),
            |row| Matrix::new(1, 2, vec![row[(0, 0)], f64::sqrt(row[(0, 0)])])
        ]
    }
}
