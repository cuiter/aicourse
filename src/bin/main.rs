use aicourse::matrix::load_idx;
use aicourse::network::dff_executor;
use aicourse::network::dff_logistic::*;
use aicourse::util::{accuracy, first_rows};
use std::fs;

fn main() {
    let train_images = first_rows(
        &load_idx::<f32>(&fs::read("datasets/mnist/train-images-idx3-ubyte").unwrap()).unwrap(),
        5000,
    );
    let train_labels = first_rows(
        &load_idx::<f32>(&fs::read("datasets/mnist/train-labels-idx1-ubyte").unwrap())
            .unwrap()
            .map(|x| x + 1.0),
        5000,
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

    let network = NeuralNetwork::<f32>::new(vec![28 * 28, 256, 10]);
    let mut seq_network = network.clone();
    let mut train_params = TrainParameters::defaults();
    train_params.show_progress = true;
    train_params.max_epochs = 100;
    train_params.batch_size = 16;
    seq_network.train(&train_images, &train_labels, train_params.clone());
    let par_network =
        dff_executor::train_parallel(&network, &train_images, &train_labels, train_params.clone());

    println!(
        "Train accuracy: {}\nTest accuracy: {}",
        accuracy(&seq_network.run(&train_images), &train_labels),
        accuracy(&seq_network.run(&t10k_images), &t10k_labels)
    );
    println!(
        "Parallel Train accuracy: {}\nParallel Test accuracy: {}",
        accuracy(&par_network.run(&train_images), &train_labels),
        accuracy(&par_network.run(&t10k_images), &t10k_labels)
    );
}
