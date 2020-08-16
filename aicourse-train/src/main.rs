use aicourse::matrix::*;
use aicourse::network::dff_executor;
use aicourse::network::dff_logistic::*;
use aicourse::util::{accuracy, first_rows};
use clap::{App, Arg};
use std::fs;

const TRAIN_IMAGES_PATH: &str = "train-images.idx3-ubyte";
const TRAIN_LABELS_PATH: &str = "train-labels.idx1-ubyte";
const TEST_IMAGES_PATH: &str = "t10k-images.idx3-ubyte";
const TEST_LABELS_PATH: &str = "t10k-labels.idx1-ubyte";

const DEFAULT_NETWORK_PATH: &str = "matrix.idx";
const DEFAULT_DATASET_PATH: &str = "aicourse-train/datasets/mnist";
const DEFAULT_SAMPLES_COUNT: u32 = 5000;
const DEFAULT_NODE_COUNT: u32 = 64;
const DEFAULT_BATCH_SIZE: u32 = 16;
const DEFAULT_MAX_EPOCHS: u32 = 100;

struct MNIST<T: Float> {
    pub train_images: Matrix<T>,
    pub train_labels: Matrix<T>,
    pub test_images: Matrix<T>,
    pub test_labels: Matrix<T>,
}

fn load_mnist(dataset_dir: &str, samples_count: u32) -> MNIST<f32> {
    let train_images = first_rows(
        &load_idx::<f32>(&fs::read(format!("{}/{}", dataset_dir, TRAIN_IMAGES_PATH)).unwrap())
            .unwrap(),
        samples_count,
    );
    let train_labels = first_rows(
        &load_idx::<f32>(&fs::read(format!("{}/{}", dataset_dir, TRAIN_LABELS_PATH)).unwrap())
            .unwrap()
            .map(|x| x + 1.0),
        samples_count,
    );
    let test_images =
        load_idx::<f32>(&fs::read(format!("{}/{}", dataset_dir, TEST_IMAGES_PATH)).unwrap())
            .unwrap();
    let test_labels =
        load_idx::<f32>(&fs::read(format!("{}/{}", dataset_dir, TEST_LABELS_PATH)).unwrap())
            .unwrap()
            .map(|x| x + 1.0);

    MNIST {
        train_images,
        train_labels,
        test_images,
        test_labels,
    }
}

fn train_network<T: Float>(mnist: &MNIST<T>, parallel: bool, node_count: u32, batch_size: u32, max_epochs: u32) -> NeuralNetwork<T> {
    let mut network = NeuralNetwork::<T>::new(vec![28 * 28, node_count, 10]);
    let mut train_params = TrainParameters::defaults();
    train_params.show_progress = true;
    train_params.max_epochs = max_epochs;
    train_params.batch_size = batch_size;
    if parallel {
        dff_executor::train_parallel(
            &network,
            &mnist.train_images,
            &mnist.train_labels,
            train_params,
        )
    } else {
        network.train(&mnist.train_images, &mnist.train_labels, train_params);
        network
    }
}

fn main() {
    let matches = App::new("aicourse training program")
                      .about("Trains a network using the MNIST dataset")
                      .arg(Arg::with_name("network")
                               .long("network")
                               .short("n")
                               .value_name("network-path")
                               .help(&format!("The file where to store the network (default: {})", DEFAULT_NETWORK_PATH))
                               .takes_value(true))
                      .arg(Arg::with_name("dataset")
                               .long("dataset")
                               .short("d")
                               .value_name("dataset-path")
                               .help(&format!("The directory where the MNIST dataset is stored (default: {})", DEFAULT_DATASET_PATH))
                               .takes_value(true))
                      .arg(Arg::with_name("samples-count")
                               .long("samples-count")
                               .help(&format!("Amount of training samples (default: {})", DEFAULT_SAMPLES_COUNT))
                               .takes_value(true))
                      .arg(Arg::with_name("node-count")
                               .long("node-count")
                               .help(&format!("Amount of nodes in the second layer of the network (default: {})", DEFAULT_NODE_COUNT))
                               .takes_value(true))
                      .arg(Arg::with_name("batch-size")
                               .long("batch-size")
                               .help(&format!("The batch size (default: {})", DEFAULT_BATCH_SIZE))
                               .takes_value(true))
                      .arg(Arg::with_name("max-epochs")
                               .long("max-epochs")
                               .help(&format!("Max amount of epochs (default: {})", DEFAULT_MAX_EPOCHS))
                               .takes_value(true))
                      .arg(Arg::with_name("test-only")
                               .long("test-only")
                               .short("t")
                               .help("Don't train the network, only test it"))
                      .arg(Arg::with_name("sequential")
                               .long("sequential")
                               .short("s")
                               .help("Train the network sequentially (default: parallel)"))
                      .get_matches();
    let network_file = matches.value_of("network").unwrap_or("network.idx");
    let dataset_dir = matches.value_of("dataset").unwrap_or("aicourse-train/datasets/mnist");

    let samples_count = matches.value_of("samples-count").map(|tc| tc.parse::<u32>().unwrap()).unwrap_or(DEFAULT_SAMPLES_COUNT);
    let node_count = matches.value_of("node-count").map(|nc| nc.parse::<u32>().unwrap()).unwrap_or(DEFAULT_NODE_COUNT);
    let batch_size = matches.value_of("batch-size").map(|bc| bc.parse::<u32>().unwrap()).unwrap_or(DEFAULT_BATCH_SIZE);
    let max_epochs = matches.value_of("max-epochs").map(|me| me.parse::<u32>().unwrap()).unwrap_or(DEFAULT_MAX_EPOCHS);

    let mnist = load_mnist(
        dataset_dir,
        samples_count
    );

    let network = if matches.is_present("test-only") {
        NeuralNetwork::<f32>::load(&load_idx(&fs::read(&network_file).unwrap()).unwrap())
    } else {
        train_network(&mnist, !matches.is_present("sequential"), node_count, batch_size, max_epochs)
    };

    println!(
        "Train accuracy: {}\nTest accuracy: {}",
        accuracy(&network.run(&mnist.train_images), &mnist.train_labels),
        accuracy(&network.run(&mnist.test_images), &mnist.test_labels)
    );

    fs::write(&network_file, save_idx(&network.save(), DataType::F32)).unwrap();
}
