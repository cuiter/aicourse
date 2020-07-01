use crate::matrix::{Matrix};

pub mod util {
    use super::*;
    pub fn classify_inputs() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(3, 4, vec![1.0, 0.5, 0.8, 0.0,
                                    0.0, 0.2, 0.1, 0.4,
                                    0.9, 0.2, 1.0, 1.0]),
             Matrix::new(4, 4, vec![0.0, 0.1, 0.2, 0.3,
                                    1.0, 1.0, 1.0, 1.0,
                                    0.0, 0.0, 0.0, 0.0,
                                    0.3, 0.5, 0.5, 0.3])]
    }
    pub fn classify_outputs() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(3, 1, vec![1.0,
                                    4.0,
                                    3.0]),
             Matrix::new(4, 1, vec![4.0,
                                    1.0,
                                    1.0,
                                    2.0])]
    }
    pub fn unclassify_inputs() -> Vec<Matrix<f64>> {
        classify_outputs()
    }
    pub fn unclassify_outputs() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(3, 4, vec![1.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 1.0,
                                    0.0, 0.0, 1.0, 0.0]),
             Matrix::new(4, 4, vec![0.0, 0.0, 0.0, 1.0,
                                    1.0, 0.0, 0.0, 0.0,
                                    1.0, 0.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0, 0.0])]
    }
    pub fn accuracy_inputs() -> Vec<(Matrix<f64>, Matrix<f64>)> {
        vec![
            (Matrix::new(10, 1, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
             Matrix::new(10, 1, vec![1.0, 2.0, 1.0, 3.0, 4.0, 1.0, 1.0, 5.0, 1.0, 7.0])),
            (Matrix::new(8, 1, vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
             Matrix::new(8, 1, vec![8.0, 2.0, 1.0, 3.0, 4.0, 1.0, 1.0, 5.0]))
        ]
    }
    pub fn accuracy_outputs() -> Vec<f32> {
        vec![
            0.5,
            0.25
        ]
    }
    pub fn batch_inputs() -> Vec<Matrix<f64>> {
        vec![
            Matrix::new(8, 2, vec![8.0, 7.0,
                                   6.0, 5.0,
                                   4.0, 3.0,
                                   2.0, 1.0,
                                   11.0, 12.0,
                                   13.0, 14.0,
                                   15.0, 16.0,
                                   17.0, 18.0]),
            Matrix::new(10, 3, vec![01.0, 02.0, 03.0,
                                    11.0, 12.0, 13.0,
                                    21.0, 22.0, 23.0,
                                    31.0, 32.0, 33.0,
                                    41.0, 42.0, 43.0,
                                    51.0, 52.0, 53.0,
                                    61.0, 62.0, 63.0,
                                    71.0, 72.0, 73.0,
                                    81.0, 82.0, 83.0,
                                    91.0, 92.0, 93.0])
        ]
    }
    pub fn batch_outputs() -> Vec<Vec<Matrix<f64>>> {
        vec![
            vec![
                Matrix::new(4, 2, vec![8.0, 7.0,
                                       6.0, 5.0,
                                       4.0, 3.0,
                                       2.0, 1.0]),
                Matrix::new(4, 2, vec![11.0, 12.0,
                                       13.0, 14.0,
                                       15.0, 16.0,
                                       17.0, 18.0])
            ],
            vec![
                Matrix::new(4, 3, vec![01.0, 02.0, 03.0,
                                       11.0, 12.0, 13.0,
                                       21.0, 22.0, 23.0,
                                       31.0, 32.0, 33.0]),
                Matrix::new(4, 3, vec![41.0, 42.0, 43.0,
                                       51.0, 52.0, 53.0,
                                       61.0, 62.0, 63.0,
                                       71.0, 72.0, 73.0]),
                Matrix::new(2, 3, vec![81.0, 82.0, 83.0,
                                       91.0, 92.0, 93.0]),
            ]
        ]
    }
    pub fn split_inputs() -> Vec<(Matrix<f64>, Vec<f32>)> {
        vec![
            (Matrix::new(10, 2, vec![1.0, 11.0,
                                     2.0, 12.0,
                                     3.0, 13.0,
                                     4.0, 14.0,
                                     5.0, 15.0,
                                     6.0, 16.0,
                                     7.0, 17.0,
                                     8.0, 18.0,
                                     9.0, 19.0,
                                     0.0, 10.0]),
             vec![0.6, 0.2, 0.2])
        ]
    }
    pub fn split_outputs() -> Vec<Vec<Matrix<f64>>> {
        vec![
            vec![
                Matrix::new(6, 2, vec![1.0, 11.0,
                                       2.0, 12.0,
                                       3.0, 13.0,
                                       4.0, 14.0,
                                       5.0, 15.0,
                                       6.0, 16.0]),
                Matrix::new(2, 2, vec![7.0, 17.0,
                                       8.0, 18.0]),
                Matrix::new(2, 2, vec![9.0, 19.0,
                                       0.0, 10.0]),
            ]
        ]
    }
}

pub mod idx {
    use super::*;
    pub fn data_inputs() -> Vec<Vec<u8>> {
        vec![
            vec![0, 0, 0x08, 1, 0, 0, 0, 1, 96],
            vec![0, 0, 0x09, 1, 0, 0, 0, 1, 160],
            vec![0, 0, 0x0B, 1, 0, 0, 0, 1, 255, 160],
            vec![0, 0, 0x0C, 1, 0, 0, 0, 1, 255, 255, 255, 160],
            vec![0, 0, 0x0D, 1, 0, 0, 0, 1, 0xC2, 0xC0, 0x00, 0x00],
            vec![0, 0, 0x0E, 1, 0, 0, 0, 1, 0xC0, 0x58, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            vec![0, 0, 0x09, 2, 0, 0, 0, 2, 0, 0, 0, 2, 1, 2, 3, 128],
            vec![0, 0, 0x0B, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 0, 2, 255, 255, 127, 255],
            vec![0, 0, 0x08, 3, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 1, 2, 255, 127, 5, 4, 3, 2],
        ]
    }
    pub fn matrix_outputs() -> Vec<Matrix<f64>> {
        vec![
            Matrix::new(1, 1, vec![96.0]),
            Matrix::new(1, 1, vec![-96.0]),
            Matrix::new(1, 1, vec![-96.0]),
            Matrix::new(1, 1, vec![-96.0]),
            Matrix::new(1, 1, vec![-96.0]),
            Matrix::new(1, 1, vec![-96.0]),
            Matrix::new(2, 2, vec![1.0, 2.0,
                                   3.0, -128.0]),
            Matrix::new(2, 2, vec![1.0, 2.0,
                                   -1.0, 32767.0]),
            Matrix::new(2, 4, vec![1.0, 2.0, 255.0, 127.0,
                                   5.0, 4.0, 3.0, 2.0]),
        ]
    }
    pub fn wrong_data_inputs() -> Vec<Vec<u8>> {
        vec![
            vec![],
            vec![1, 0, 0x08, 1, 0, 0, 0, 1, 69],
            vec![0, 25, 0x08, 1, 0, 0, 0, 1, 69],
            vec![0, 0, 0x08, 1, 0, 0, 0, 2, 69],
            vec![0, 0, 0x99, 2, 0, 0, 0, 2, 0, 0, 0, 2, 1, 2, 3, 128],
            vec![0, 0, 0x0B, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 2, 255, 255, 127, 255],
        ]
    }
}

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

pub mod dff_logistic {
    use super::*;
    pub fn tests_inputs() -> Vec<Matrix<f64>> {
        vec![
             Matrix::new(6, 2, vec![-5.0, -5.0,
                                    5.0, -5.0,
                                    5.0, -5.0,
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
             Matrix::new(6, 1, vec![1.0,
                                    2.0,
                                    2.0,
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
