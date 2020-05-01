use aicourse::matrix::{Matrix, load_idx};
use aicourse::network::dff_logistic::*;
use std::io;
use std::io::prelude::*;
use std::fs::File;

fn read_file(path: &str) -> Vec<u8> {
    let mut buffer = Vec::new();
    let mut f = File::open(path).unwrap();
    f.read_to_end(&mut buffer).unwrap();
    buffer
}

fn main() {
    let train_images = load_idx::<f64>(&read_file("datasets/train-images-idx3-ubyte")).unwrap();
    let train_labels = load_idx::<f64>(&read_file("datasets/train-labels-idx1-ubyte")).unwrap().map(|x| x + 1.0);
    let t10k_images = load_idx::<f64>(&read_file("datasets/t10k-images-idx3-ubyte")).unwrap();
    let t10k_labels = load_idx::<f64>(&read_file("datasets/t10k-labels-idx1-ubyte")).unwrap().map(|x| x + 1.0);

    println!("{:?} {:?} {:?} {:?}", t10k_images.get_dimensions(), t10k_labels.get_dimensions(), train_images.get_dimensions(), train_labels.get_dimensions());

    let mut network = NeuralNetwork::<f64>::new(vec![28 * 28, 256, 10]);
    network.train(&train_images, &train_labels, 0.005, CostMethod::Delta);

    dbg!(network.get_configuration());

}
