use crate::matrix::{Matrix, Float};
use opencv::prelude::{Mat, Size, CV_8UC1};

pub fn cv_to_matrix<T: Float>(mat: Mat) -> Matrix<T> {

}

pub fn matrix_to_cv<T: Float>(matrix: Matrix<T>) -> Mat {
    let mut mat = Mat::new_size(Size::new(matrix.get_n(), matrix.get_m()), CV_8UC1);
}
