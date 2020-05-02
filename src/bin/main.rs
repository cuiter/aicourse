use aicourse::matrix::load_idx;
use aicourse::network::dff_logistic::*;
use std::fs;

fn main() {
    let train_images =
        load_idx::<f32>(&fs::read("datasets/mnist/train-images-idx3-ubyte").unwrap()).unwrap();
    let train_labels =
        load_idx::<f32>(&fs::read("datasets/mnist/train-labels-idx1-ubyte").unwrap())
            .unwrap()
            .map(|x| x + 1.0);
    let t10k_images =
        load_idx::<f32>(&fs::read("datasets/mnist/t10k-images-idx3-ubyte").unwrap()).unwrap();
    let t10k_labels = load_idx::<f32>(&fs::read("datasets/mnist/t10k-labels-idx1-ubyte").unwrap())
        .unwrap()
        .map(|x| x + 1.0);

    println!(
        "{:?} {:?} {:?} {:?}",
        t10k_images.get_dimensions(),
        t10k_labels.get_dimensions(),
        train_images.get_dimensions(),
        train_labels.get_dimensions()
    );

    let mut network = NeuralNetwork::<f32>::new(vec![28 * 28, 256, 10]);
    network.train(&train_images, &train_labels, 0.005, CostMethod::Delta);

    dbg!(network.get_configuration());
}
