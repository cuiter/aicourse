#[macro_use]
extern crate bencher;

use aicourse::matrix::{load_idx, Matrix};
use aicourse::network::dff_logistic::{CostMethod, NeuralNetwork};
use aicourse::testdata;
use bencher::{black_box, Bencher};
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

fn dff_logistic_train(bench: &mut Bencher) {
    let inputs = &testdata::dff_logistic::tests_inputs()[0];
    let correct_outputs = &testdata::dff_logistic::tests_outputs()[0];

    bench.iter(|| {
        let mut network = NeuralNetwork::<f64>::new_seeded(vec![2, 5, 5, 4], 420);

        network.train(inputs, correct_outputs, 0.0005, CostMethod::Delta);

        black_box(network);
    });
}

benchmark_group!(
    benches,
    matrix_zero_100_100,
    matrix_zero_1000_1000,
    load_idx_mnist_train_labels,
    load_idx_mnist_test_images,
    dff_logistic_train
);
benchmark_main!(benches);
