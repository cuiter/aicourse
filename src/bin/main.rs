use aicourse::matrix::{load_idx, Float, Matrix};
use aicourse::network::dff_logistic::*;
use aicourse::util::accuracy;
use std::fs;

fn first_rows<T: Float>(matrix: &Matrix<T>, n_rows: u32) -> Matrix<T> {
    matrix.get_sub_matrix(0, 0, n_rows, matrix.get_n())
}

fn main() {
    let train_images = first_rows(
        &load_idx::<f32>(&fs::read("datasets/mnist/train-images-idx3-ubyte").unwrap()).unwrap(),
        1000,
    );
    let train_labels = first_rows(
        &load_idx::<f32>(&fs::read("datasets/mnist/train-labels-idx1-ubyte").unwrap())
            .unwrap()
            .map(|x| x + 1.0),
        1000,
    );
    let t10k_images =
        load_idx::<f32>(&fs::read("datasets/mnist/t10k-images-idx3-ubyte").unwrap()).unwrap();
    let t10k_labels = load_idx::<f32>(&fs::read("datasets/mnist/t10k-labels-idx1-ubyte").unwrap())
        .unwrap()
        .map(|x| x + 1.0);

    println!(
        "{:?} {:?} {:?} {:?}",
        train_images.get_dimensions(),
        train_labels.get_dimensions(),
        t10k_images.get_dimensions(),
        t10k_labels.get_dimensions(),
    );

    let mut network = NeuralNetwork::<f32>::new(vec![28 * 28, 256, 10]);
    let mut train_params = TrainParameters::defaults();
    train_params.show_progress = true;
    train_params.max_epochs = 20;
    network.train(&train_images, &train_labels, train_params);

    println!(
        "Train accuracy: {}",
        accuracy(&network.run(&train_images), &train_labels)
    );
    println!(
        "Test accuracy: {}",
        accuracy(&network.run(&t10k_images), &t10k_labels)
    );
}
