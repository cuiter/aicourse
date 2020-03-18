extern crate num_traits;

mod matrix;

use matrix::Matrix;

fn main() {
    let m1 = Matrix::new(2, 2, vec![9.0, 1.0, 2.0, 3.0]);
    let m2 = Matrix::new(2, 2, vec![10.0, 10.0, 10.0, 10.0]);
    println!("Hello, world!\n{}\n+\n{}\n=\n{}", &m1, &m2, &m1 + &m2);
}
