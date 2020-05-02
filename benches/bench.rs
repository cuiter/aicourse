#[macro_use]
extern crate bencher;

use aicourse::matrix::{Matrix, load_idx};
use bencher::{Bencher, black_box};
use std::fs;

fn matrix_zero(bench: &mut Bencher, rows_cols: u32) {
    let mut sum_size = 0;
    bench.iter(|| {
        let matrix = Matrix::<f32>::zero(rows_cols, rows_cols);
        sum_size += matrix.get_size();
    });
    black_box(sum_size);
}

fn matrix_zero_100_100(bench: &mut Bencher) {
    matrix_zero(bench, 100);
}

fn matrix_zero_1000_1000(bench: &mut Bencher) {
    matrix_zero(bench, 1000);
}

fn load_idx_mnist_train_labels(bench: &mut Bencher) {
    let data = fs::read("datasets/mnist/train-labels-idx1-ubyte").unwrap();
    bench.iter(|| {
        let matrix = load_idx::<f32>(&data).unwrap();
        black_box(matrix.sum());
    });
}

fn load_idx_mnist_test_images(bench: &mut Bencher) {
    let data = fs::read("datasets/mnist/t10k-images-idx3-ubyte").unwrap();
    bench.iter(|| {
        let matrix = load_idx::<f32>(&data).unwrap();
        black_box(matrix.sum());
    });
}

benchmark_group!(benches, matrix_zero_100_100, matrix_zero_1000_1000, load_idx_mnist_train_labels, load_idx_mnist_test_images);
benchmark_main!(benches);
