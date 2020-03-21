use num_traits::{float, cast};
use core::slice::Iter;
use std::fmt;
use std::ops;
use std::cmp;

/// Any floating point type that can be used by Matrix<T>.
pub trait Float: float::Float + cast::FromPrimitive + fmt::Debug {}
impl<T: float::Float + cast::FromPrimitive + fmt::Debug> Float for T {}

/// A linear algebra matrix consisting of real (floating point) numbers.
#[derive(Debug, PartialEq, Clone)]
pub struct Matrix<T : Float> {
    m: u32, // Number of rows
    n: u32, // Number of columns
    data: Vec<T> // Data
}

impl<T : Float> Matrix<T> {
    /// Constructs a matrix with the given size and data.
    /// The data is in row-major order.
    pub fn new(m: u32, n: u32, data: Vec<T>) -> Matrix<T> {
        assert!(m > 0);
        assert!(n > 0);
        assert!(data.len() == (m * n) as usize);

        Matrix { m: m, n: n, data: data }
    }

    /// Constructs a matrix with the given size with all its elements set to zero.
    pub fn zero(m: u32, n: u32) -> Matrix<T> {
        assert!(m > 0);
        assert!(n > 0);

        let mut data = Vec::with_capacity((m * n) as usize);
        data.resize((m * n) as usize, T::zero());
        Matrix { m: m, n: n, data: data }
    }

    /// Constructs a matrix with the given size with all its elements set to one.
    pub fn one(m: u32, n: u32) -> Matrix<T> {
        assert!(m > 0);
        assert!(n > 0);

        let mut data = Vec::with_capacity((m * n) as usize);
        data.resize((m * n) as usize, T::from_u8(1).unwrap());
        Matrix { m: m, n: n, data: data }
    }

    /// Constructs an identity matrix with the given size.
    /// The size must be square.
    pub fn new_identity(m: u32, n: u32) -> Matrix<T> {
        assert_eq!(m, n);
        let mut result = Matrix::<T>::zero(m, n);

        for i in 0..cmp::min(m, n) {
            result[(i, i)] = T::from_u8(1).unwrap();
        }

        result
    }

    /// Returns m (number of rows).
    pub fn get_m(&self) -> u32 {
        self.m
    }
    /// Returns m (number of columns).
    pub fn get_n(&self) -> u32 {
        self.n
    }
    /// Returns the number of elements in the matrix.
    pub fn get_size(&self) -> u32 {
        self.m * self.n
    }

    /// Returns an iterator that loops through all the elements of the
    /// matrix in row-major order.
    pub fn iter(&self) -> Iter<T> {
        self.data.iter()
    }

    /// Compares whether two matrices are equal within a given precision.
    pub fn approx_eq(&self, other: &Matrix<T>, precision: T) -> bool {
        if self.m != other.get_m() || self.n != other.get_n() {
            false
        } else {
            !self.iter()
                .zip(other.iter())
                .any(|(a, b)| {
                    a != a || b != b || a < &(*b - precision) || a > &(*b + precision)
                })
        }
    }

    /// Extracts a part of the matrix, starting at (start_m, start_n) with size (m, n).
    pub fn get_sub_matrix(&self, start_m: u32, start_n: u32, m: u32, n: u32) -> Matrix<T> {
        assert!(start_m + m <= self.m);
        assert!(start_n + n <= self.n);

        let mut result = Matrix::zero(m, n);

        for row in start_m..(start_m + m) {
            for col in start_n..(start_n + n) {
                result[(row - start_m, col - start_n)] = self[(row, col)];
            }
        }

        result
    }

    /// Extracts one row of the matrix.
    pub fn get_row(&self, m: u32) -> Matrix<T> {
        self.get_sub_matrix(m, 0, 1, self.get_n())
    }

    /// Extracts one column of the matrix.
    pub fn get_column(&self, n: u32) -> Matrix<T> {
        self.get_sub_matrix(0, n, self.get_m(), 1)
    }

    /// Concatenates two matrices horizontally. The other matrix
    /// is the matrix to the right.
    pub fn h_concat(&self, other: &Matrix<T>) -> Matrix<T> {
        assert!(self.m == other.get_m());

        let mut result = Matrix::zero(self.m, self.n + other.get_n());

        for m in 0..self.m {
            for n in 0..self.n {
                result[(m, n)] = self[(m, n)]
            }

            for other_n in 0..other.get_n() {
                result[(m, other_n + self.n)] = other[(m, other_n)]
            }
        }

        result
    }

    /// Concatenates two matrices vertically. The other matrix
    /// is the matrix to the bottom.
    pub fn v_concat(&self, other: &Matrix<T>) -> Matrix<T> {
        assert!(self.n == other.get_n());

        let mut result = Matrix::zero(self.m + other.get_m(), self.n);

        for m in 0..self.m {
            for n in 0..self.n {
                result[(m, n)] = self[(m, n)]
            }
        }

        for other_m in 0..other.get_m() {
            for n in 0..self.n {
                result[(other_m + self.m, n)] = other[(other_m, n)]
            }
        }

        result
    }

    /// Swaps two rows in-place.
    pub fn swap_rows(&mut self, m_1: u32, m_2: u32) {
        assert!(m_1 < self.m);
        assert!(m_2 < self.m);

        if m_1 == m_2 { return; }

        let range_1 = ((m_1 * self.n) as usize)..((m_1 * self.n + self.n) as usize);
        let range_2 = ((m_2 * self.n) as usize)..((m_2 * self.n + self.n) as usize);

        let temp_1: Vec<T> = self.data[range_1.clone()].to_vec();
        let temp_2: Vec<T> = self.data[range_2.clone()].to_vec();

        self.data.splice(range_1.clone(), temp_2);
        self.data.splice(range_2.clone(), temp_1);
    }

    /// Returns the sum of all elements in the matrix.
    pub fn sum(&self) -> T {
        self.data.iter().fold(T::zero(), |sum, &val| { sum + val })
    }

    /// Returns the associated identity matrix.
    pub fn identity(&self) -> Matrix<T> {
        Matrix::new_identity(self.n, self.n)
    }

    /// Returns the matrix transposed (row and column indices swapped).
    pub fn transpose(&self) -> Matrix<T> {
        let mut result = Matrix::zero(self.n, self.m);
        for m in 0..self.m {
            for n in 0..self.n {
                result[(n, m)] = self[(m, n)];
            }
        }
        result
    }

    /// Performs the Gauss-Jordan elimination on the matrix.
    pub fn gauss_jordan(&self) -> Matrix<T> {
        let mut result = self.clone();

        for row in 0..self.m {
            let mut max_row = row;

            // Move maximum pivot value to top.
            for sub_row in (max_row + 1)..self.m {
                if self[(sub_row, row)] > self[(max_row, row)] {
                    max_row = sub_row;
                }
            }
            result.swap_rows(row, max_row);

            let gauss_factor = result[(row, row)];
            for col in 0..self.n {
                result[(row, col)] = result[(row, col)] / gauss_factor;
            }

            for sub_row in 0..self.m {
                if row != sub_row {
                    let gauss_factor_2 = result[(sub_row, row)] / result[(row, row)];
                    for sub_col in 0..self.n {
                        result[(sub_row, sub_col)] = result[(sub_row, sub_col)]
                            - result[(row, sub_col)] * gauss_factor_2;
                    }
                    let jordan_factor = result[(row, sub_row)] / result[(sub_row, sub_row)];
                    for sub_col in 0..self.n {
                        result[(row, sub_col)] = result[(row, sub_col)]
                            - result[(sub_row, sub_col)] * jordan_factor;
                    }
                }
            }
        }

        result
    }

    /// Solves the linear equation in the matrix,
    /// with the right-most column being the output.
    pub fn solve(&self) -> Matrix<T> {
        let gj_result = self.gauss_jordan();

        gj_result.get_column(self.n - 1)
    }

    fn determinant_n(&self, n: u32) -> T {
        let mut det = T::zero();

        if n == 2 {
            self[(0, 0)] * self[(1, 1)] - self[(1, 0)] * self[(0, 1)]
        } else {
            let mut submatrix = Matrix::zero(self.m, self.n);

            for x in 0..n {
                let mut sub_i = 0;
                for i in 1..n {
                    let mut sub_j = 0;
                    for j in 0..n {
                        if j == x {
                            continue;
                        }
                        submatrix[(sub_i, sub_j)] = self[(i, j)];
                        sub_j += 1;
                    }
                    sub_i += 1;
                }
                det = det + T::powi(T::from_i8(-1).unwrap(), x as i32) * self[(0, x)] * submatrix.determinant_n(n - 1);
            }

            det
        }
    }

    /// Returns the determinant of the matrix.
    pub fn determinant(&self) -> T {
        assert_eq!(self.m, self.n);

        self.determinant_n(self.n)
    }

    /// Returns the inverse of the matrix, only if the determinant is nonzero.
    pub fn inv(&self) -> Option<Matrix<T>> {
        if self.m != self.n || self.determinant() == T::zero() {
            None
        } else {
            let interim = self.h_concat(&self.identity());
            let interim_2 = interim.gauss_jordan();
            Some(interim_2.get_sub_matrix(0, self.n, self.m, self.n))
        }
    }

    /// Returns the Moore-Penrose inverse, if it can be computed by
    /// left or right inverse. SVD as used in Matlab or GNU Octave `pinv`
    /// is not yet implemented.
    pub fn pinv(&self) -> Option<Matrix<T>> {
        let l_interim = (&self.transpose() * self).inv();
        let r_interim = (self * &self.transpose()).inv();

        match (l_interim, r_interim) {
            (Some(u_l_interim), _) => Some(&u_l_interim * &self.transpose()),
            (_, Some(u_r_interim)) => Some(&self.transpose() * &u_r_interim),
            _ => None // TODO: Implement SVD
        }
    }

    /// Returns the self.data vector element index given a 2D element index (row, column).
    fn item_index(&self, idx: (u32, u32)) -> usize {
        assert!(idx.0 < self.m && idx.1 < self.n, "index out of bounds: m={} n={} self.m={}, self.n={}", idx.0, idx.1, self.m, self.n);
        (idx.0 * self.n + idx.1) as usize
    }
}

impl<'a, T : Float> ops::Index<(u32, u32)> for Matrix<T> {
    type Output = T;

    /// Returns an element of the matrix indexed by (row, column).
    fn index(&self, idx: (u32, u32)) -> &T {
        &self.data[self.item_index(idx)]
    }
}

impl<'a, T : Float> ops::IndexMut<(u32, u32)> for Matrix<T> {
    /// Returns a mutable reference of an element inside
    /// the matrix indexed by (row, column).
    fn index_mut(&mut self, idx: (u32, u32)) -> &mut T {
        let index = self.item_index(idx);
        &mut self.data[index]
    }
}

impl<'a, T : Float> ops::Neg for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn neg(self) -> Matrix<T> {
        Matrix::new(self.m, self.n,
                    self.data.iter().map(|a| -*a).collect())
    }
}

impl<'a, 'b, T : Float> ops::Add<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, other: &'b Matrix<T>) -> Matrix<T> {
        assert!(self.m == other.get_m());
        assert!(self.n == other.get_n());

        let aiter = self.iter()
            .zip(other.iter())
            .map(|(x, y)| x.clone() + y.clone());
        let res_data : Vec<T> = aiter.collect();
        Matrix::new(self.m, self.n, res_data)
    }
}

impl<'a, 'b, T : Float> ops::Sub<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, other: &'b Matrix<T>) -> Matrix<T> {
        self + &(-other)
    }
}

impl<'a, 'b, T : Float> ops::Mul<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    /// Returns the matrix product of two matrices.
    fn mul(self, other: &'b Matrix<T>) -> Matrix<T> {
        assert_eq!(self.n, other.get_m(), "self.n == other.m");

        let mut result = Matrix::zero(self.m, other.get_n());

        for row in 0..self.m {
            for column in 0..other.get_n() {
                for codependent in 0..self.n {
                    result[(row, column)] = result[(row, column)]
                        + self[(row, codependent)]
                          * other[(codependent, column)];
                }
            }
        }

        result
    }
}

impl<'a, T : Float> ops::Mul<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    /// Multiplies every element of the matrix with the given scalar.
    fn mul(self, other: T) -> Matrix<T> {
        Matrix::new(self.m, self.n, self.data.iter().map(|a| *a * other).collect())
    }
}

impl<'a, T : Float> ops::Div<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    /// Divides every element of the matrix with the given scalar.
    fn div(self, other: T) -> Matrix<T> {
        Matrix::new(self.m, self.n, self.data.iter().map(|a| *a / other).collect())
    }
}


impl<'b, T : Float> ops::AddAssign<&'b Matrix<T>> for Matrix<T> {
    fn add_assign(&mut self, other: &'b Matrix<T>) {
        *self = &*self + other
    }
}

impl<'b, T : Float> ops::SubAssign<&'b Matrix<T>> for Matrix<T> {
    fn sub_assign(&mut self, other: &'b Matrix<T>) {
        *self = &*self - other
    }
}

impl<'b, T : Float> ops::MulAssign<&'b Matrix<T>> for Matrix<T> {
    fn mul_assign(&mut self, other: &'b Matrix<T>) {
        *self = &*self * other
    }
}

impl<T : Float> ops::MulAssign<T> for Matrix<T> {
    fn mul_assign(&mut self, other: T) {
        *self = &*self * other
    }
}

impl<T : Float> ops::DivAssign<T> for Matrix<T> {
    fn div_assign(&mut self, other: T) {
        *self = &*self / other
    }
}

impl<T : Float + fmt::Display> fmt::Display for Matrix<T> {
    /// Prints the contents of the matrix in a pretty format.
    /// Every row is printed on a new line.
    /// It is recommended to print a newline before printing the matrix,
    /// so that the lines are aligned.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for m in 0..self.m {
            if m == 0 {
                write!(f, "[")?;
            } else {
                write!(f, " ")?;
            }
            for n in 0..self.n {
                write!(f, " {}", self[(m, n)])?;
            }
            if m == self.m - 1 {
                write!(f, " ]")?;
            } else {
                write!(f, "\n")?;
            }
        }
        write!(f, "")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let matrix = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(matrix.get_m(), 3);
        assert_eq!(matrix.get_n(), 2);
        assert_eq!(matrix.get_size(), 6);
    }

    #[test]
    #[should_panic]
    fn new_wrong_1() {
        #[allow(unused_variables)]
        // Wrong number of elements given
        let matrix = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    #[should_panic]
    fn new_wrong_2() {
        let empty_vec : Vec<f64> = vec![];
        #[allow(unused_variables)]
        // Size is zero
        let matrix = Matrix::new(3, 0, empty_vec);
    }

    #[test]
    fn zero() {
        let matrix = Matrix::zero(2, 2);

        let expected_matrix = Matrix::new(2, 2, vec![0.0, 0.0, 0.0, 0.0]);

        assert_eq!(matrix, expected_matrix);
    }

    #[test]
    fn new_identity() {
        let matrix = Matrix::new_identity(2, 2);

        let expected_matrix = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);

        assert_eq!(matrix, expected_matrix);
    }

    #[test]
    fn index() {
        let matrix = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(matrix[(0, 0)], 1.0);
        assert_eq!(matrix[(2, 0)], 5.0);
        assert_eq!(matrix[(2, 1)], 6.0);
    }

    #[test]
    fn index_mut() {
        let mut matrix = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        matrix[(0, 0)] = 11.0;
        matrix[(1, 1)] = 14.0;

        let expected_matrix = Matrix::new(3, 2, vec![11.0, 2.0, 3.0, 14.0, 5.0, 6.0]);
        assert_eq!(matrix, expected_matrix);
    }

    #[test]
    #[should_panic]
    fn index_mut_wrong_1() {
        let mut matrix = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Out-of-bounds index
        matrix[(9, 1)] = 14.0;
    }

    #[test]
    #[should_panic]
    fn index_mut_wrong_2() {
        let mut matrix = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Out-of-bounds index
        matrix[(1, 9)] = 14.0;
    }

    #[test]
    fn sub_matrix() {
        let matrix_1 = Matrix::new(3, 5, vec![1.0, 2.0, 3.0, 1.0, 0.0,
                                              4.0, 5.0, 6.0, 0.0, 1.0,
                                              7.0, 8.0, 9.0, 0.0, 0.0]);

        let expected_matrix = Matrix::new(2, 3, vec![4.0, 5.0, 6.0,
                                                     7.0, 8.0, 9.0]);

        assert_eq!(matrix_1.get_sub_matrix(1, 0, 2, 3), expected_matrix);
    }

    #[test]
    fn h_concat() {
        let matrix_1 = Matrix::new(3, 3, vec![1.0, 2.0, 3.0,
                                              4.0, 5.0, 6.0,
                                              7.0, 8.0, 9.0]);
        let matrix_2 = Matrix::new(3, 2, vec![1.0, 0.0,
                                              0.0, 1.0,
                                              0.0, 0.0]);

        let expected_matrix = Matrix::new(3, 5, vec![1.0, 2.0, 3.0, 1.0, 0.0,
                                                     4.0, 5.0, 6.0, 0.0, 1.0,
                                                     7.0, 8.0, 9.0, 0.0, 0.0]);

        assert_eq!(matrix_1.h_concat(&matrix_2), expected_matrix);
    }

    #[test]
    fn v_concat() {
        let matrix_1 = Matrix::new(3, 3, vec![1.0, 2.0, 3.0,
                                              4.0, 5.0, 6.0,
                                              7.0, 8.0, 9.0]);
        let matrix_2 = Matrix::new(2, 3, vec![1.0, 0.0, 0.0,
                                              0.0, 1.0, 0.0]);

        let expected_matrix = Matrix::new(5, 3, vec![1.0, 2.0, 3.0,
                                                     4.0, 5.0, 6.0,
                                                     7.0, 8.0, 9.0,
                                                     1.0, 0.0, 0.0,
                                                     0.0, 1.0, 0.0]);

        assert_eq!(matrix_1.v_concat(&matrix_2), expected_matrix);
    }

    #[test]
    fn swap_rows() {
        let mut matrix = Matrix::new(3, 3, vec![1.0, 2.0, 3.0,
                                            4.0, 5.0, 6.0,
                                            7.0, 8.0, 9.0]);
        matrix.swap_rows(0, 2);

        let expected_matrix = Matrix::new(3, 3, vec![7.0, 8.0, 9.0,
                                                     4.0, 5.0, 6.0,
                                                     1.0, 2.0, 3.0]);

        assert_eq!(matrix, expected_matrix);
    }

    #[test]
    fn sum() {
        let matrix = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert_eq!(matrix.sum(), 21.0);
    }

    #[test]
    fn neg() {
        let matrix = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let expected_matrix = Matrix::new(3, 2, vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);

        assert_eq!(-&matrix, expected_matrix);
    }

    #[test]
    fn add() {
        let matrix_1 = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix_2 = Matrix::new(3, 2, vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);

        let mut matrix_3 = matrix_1.clone();
        matrix_3 += &matrix_2;

        let expected_matrix = Matrix::new(3, 2, vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);

        assert_eq!(&matrix_1 + &matrix_2, expected_matrix);
        assert_eq!(matrix_3, expected_matrix);
    }

    #[test]
    fn sub() {
        let matrix_1 = Matrix::new(3, 2, vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
        let matrix_2 = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let mut matrix_3 = matrix_1.clone();
        matrix_3 -= &matrix_2;

        let expected_matrix = Matrix::new(3, 2, vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);

        assert_eq!(&matrix_1 - &matrix_2, expected_matrix);
        assert_eq!(matrix_3, expected_matrix);
    }

    #[test]
    fn mul_1() {
        let matrix_1 = Matrix::new(3, 3, vec![1.0, 2.0, 3.0,
                                              4.0, 5.0, 6.0,
                                              7.0, 8.0, 9.0]);
        let matrix_2 = Matrix::new(3, 3, vec![1.0, 2.0, 3.0,
                                              4.0, 5.0, 6.0,
                                              7.0, 8.0, 9.0]);

        let mut matrix_3 = matrix_1.clone();
        matrix_3 *= &matrix_2;

        let expected_matrix = Matrix::new(3, 3, vec![30.0, 36.0, 42.0,
                                                     66.0, 81.0, 96.0,
                                                     102.0, 126.0, 150.0]);

        assert_eq!(&matrix_1 * &matrix_2, expected_matrix);
        assert_eq!(matrix_3, expected_matrix);
    }

    #[test]
    fn mul_2() {
        let matrix_1 = Matrix::new(3, 1, vec![1.0,
                                              2.0,
                                              3.0]);
        let matrix_2 = Matrix::new(1, 3, vec![1.0, 2.0, 3.0]);

        let mut matrix_3 = matrix_1.clone();
        matrix_3 *= &matrix_2;

        let expected_matrix = Matrix::new(3, 3, vec![1.0, 2.0, 3.0,
                                                     2.0, 4.0, 6.0,
                                                     3.0, 6.0, 9.0]);

        assert_eq!(&matrix_1 * &matrix_2, expected_matrix);
        assert_eq!(matrix_3, expected_matrix);
    }

    #[test]
    fn mul_3() {
        let matrix_1 = Matrix::new(1, 3, vec![1.0, 2.0, 3.0]);
        let matrix_2 = Matrix::new(3, 1, vec![1.0,
                                              2.0,
                                              3.0]);

        let mut matrix_3 = matrix_1.clone();
        matrix_3 *= &matrix_2;

        let expected_matrix = Matrix::new(1, 1, vec![14.0]);

        assert_eq!(&matrix_1 * &matrix_2, expected_matrix);
        assert_eq!(matrix_3, expected_matrix);
    }

    #[test]
    fn mul_4() {
        let matrix_1 = Matrix::new(3, 3, vec![1.0, 2.0, 3.0,
                                              4.0, 5.0, 6.0,
                                              7.0, 8.0, 9.0]);
        let matrix_2 = Matrix::new(3, 1, vec![1.0,
                                              2.0,
                                              3.0]);

        let mut matrix_3 = matrix_1.clone();
        matrix_3 *= &matrix_2;

        let expected_matrix = Matrix::new(3, 1, vec![14.0,
                                                     32.0,
                                                     50.0]);

        assert_eq!(&matrix_1 * &matrix_2, expected_matrix);
        assert_eq!(matrix_3, expected_matrix);
    }

    #[test]
    fn mul_scalar() {
        let matrix_1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let scalar = 2.0;

        let mut matrix_2 = matrix_1.clone();
        matrix_2 *= scalar;

        let expected_matrix = Matrix::new(2, 2, vec![2.0, 4.0, 6.0, 8.0]);

        assert_eq!(&matrix_1 * scalar, expected_matrix);
        assert_eq!(matrix_2, expected_matrix);
    }

    #[test]
    fn div_scalar() {
        let matrix_1 = Matrix::new(2, 2, vec![2.0, 4.0, 6.0, 8.0]);
        let scalar = 2.0;

        let mut matrix_2 = matrix_1.clone();
        matrix_2 /= scalar;

        let expected_matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

        assert_eq!(&matrix_1 / scalar, expected_matrix);
        assert_eq!(matrix_2, expected_matrix);
    }

    #[test]
    fn transpose() {
        let matrix = Matrix::new(3, 2, vec![1.0, 2.0,
                                            3.0, 4.0,
                                            5.0, 6.0]);

        let expected_matrix = Matrix::new(2, 3, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);

        assert_eq!(matrix.transpose(), expected_matrix);
    }

    #[test]
    fn gauss_jordan_1() {
        let matrix = Matrix::new(3, 4, vec![0.0, 1.0, 1.0, 5.0,
                                            3.0, 2.0, 2.0, 13.0,
                                            1.0, -1.0, 3.0, 8.0]);

        let expected_matrix = Matrix::new(3, 4, vec![1.0, 0.0, 0.0, 1.0,
                                                     0.0, 1.0, 0.0, 2.0,
                                                     0.0, 0.0, 1.0, 3.0]);

        assert_eq!(matrix.gauss_jordan(), expected_matrix);
    }

    #[test]
    fn gauss_jordan_2() {
        let matrix = Matrix::new(3, 6, vec![1.0, 2.0, 0.0, 1.0, 0.0, 0.0,
                                            1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                                            2.0, 2.0, 2.0, 0.0, 0.0, 1.0]);


        let expected_matrix = Matrix::new(3, 6, vec![1.0, 0.0, 0.0, 1.0, 2.0, -1.0,
                                            0.0, 1.0, 0.0, 0.0, -1.0, 0.5,
                                            0.0, 0.0, 1.0, -1.0, -1.0, 1.0]);

        assert_eq!(matrix.gauss_jordan(), expected_matrix);
    }

    #[test]
    fn solve() {
        let matrix = Matrix::new(3, 4, vec![0.0, 1.0, 1.0, 5.0,
                                            3.0, 2.0, 2.0, 13.0,
                                            1.0, -1.0, 3.0, 8.0]);

        let result = matrix.solve();

        let expected_matrix = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]);

        assert_eq!(result, expected_matrix);
    }

    #[test]
    fn determinant_1() {
        let matrix = Matrix::new(4, 4, vec![1.0, 2.0, 3.0, 4.0,
                                            5.0, 6.0, 7.0, 8.0,
                                            9.0, 1.0, 2.0, 3.0,
                                            4.0, 5.0, 6.0, 9.0]);

        assert_eq!(matrix.determinant(), -72.0);
    }

    #[test]
    fn determinant_2() {
        let matrix = Matrix::new(4, 4, vec![1.0, 2.0, 3.0, 4.0,
                                            5.0, 6.0, 7.0, 8.0,
                                            9.0, 1.0, 2.0, 3.0,
                                            4.0, 5.0, 6.0, 7.0]);

        assert_eq!(matrix.determinant(), 0.0);
    }

    #[test]
    fn inv_1() {
        let matrix = Matrix::new(3, 3, vec![1.0, 2.0, 0.0,
                                            1.0, 0.0, 1.0,
                                            2.0, 2.0, 2.0]);

        assert_eq!(&matrix * &matrix.inv().unwrap(), matrix.identity());
        assert_eq!(&matrix.inv().unwrap() * &matrix, matrix.identity());
    }

    #[test]
    fn inv_2() {
        let matrix = Matrix::new(3, 3, vec![1.0, 2.0, 3.0,
                                            0.0, 1.0, 5.0,
                                            5.0, 6.0, 0.0]);

        let result_1 = &matrix * &matrix.inv().unwrap();
        let result_2 = &matrix.inv().unwrap() * &matrix;

        assert!(result_1.approx_eq(&matrix.identity(), 0.001), "result_1 =\n{}", result_1);
        assert!(result_2.approx_eq(&matrix.identity(), 0.001), "result_2 =\n{}", result_1);
   }

    #[test]
    fn pinv() {
        let matrix = Matrix::new(3, 5, vec![-1.0, 8.0, 2.0, 8.0, 7.0,
                                            5.0, 6.0, -5.0, 7.0, 2.0,
                                            -9.0, 0.0, 1.0, 2.0, -3.0]);

        let result = &matrix * &matrix.pinv().unwrap();

        assert!(result.approx_eq(&result.identity(), 0.001), "result =\n{}", result);
    }
}
