#[macro_use]
extern crate bencher;

use aicourse::matrix::{Matrix, load_idx};
use bencher::{Bencher, black_box};
use std::fs;

fn matrix_new(bench: &mut Bencher, rows_cols: u32) {
    let mut sum_size = 0;
    bench.iter(|| {
        let matrix = Matrix::<f32>::zero(rows_cols, rows_cols);
        sum_size += matrix.get_size();
    });
    black_box(sum_size);
}

fn matrix_new_100_100(bench: &mut Bencher) {
    matrix_new(bench, 100);
}

fn matrix_new_1000_1000(bench: &mut Bencher) {
    matrix_new(bench, 1000);
}

fn load_idx_mnist_labels(bench: &mut Bencher) {
    let train_images_data = fs::read("datasets/mnist/train-labels-idx1-ubyte").unwrap();
    bench.iter(|| {
        let matrix = load_idx::<f32>(&train_images_data).unwrap();
        black_box(matrix.sum());
    });
}

benchmark_group!(benches, matrix_new_100_100, matrix_new_1000_1000, load_idx_mnist_labels);
benchmark_main!(benches);
