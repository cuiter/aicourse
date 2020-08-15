use aicourse::matrix::*;
use aicourse::network::dff_executor;
use aicourse::network::dff_logistic::*;
use aicourse::util::{accuracy, first_rows};
use clap::{App, Arg};
use std::fs;

const TRAIN_SAMPLES_COUNT: u32 = 5000;
const TRAIN_IMAGES_PATH: &str = "train-images.idx3-ubyte";
const TRAIN_LABELS_PATH: &str = "train-labels.idx1-ubyte";
const TEST_IMAGES_PATH: &str = "t10k-images.idx3-ubyte";
const TEST_LABELS_PATH: &str = "t10k-labels.idx1-ubyte";

struct MNIST<T: Float> {
    pub train_images: Matrix<T>,
    pub train_labels: Matrix<T>,
    pub test_images: Matrix<T>,
    pub test_labels: Matrix<T>,
}

fn load_mnist(dataset_dir: &str) -> MNIST<f32> {
    let train_images = first_rows(
        &load_idx::<f32>(&fs::read(format!("{}/{}", dataset_dir, TRAIN_IMAGES_PATH)).unwrap())
            .unwrap(),
        TRAIN_SAMPLES_COUNT,
    );
    let train_labels = first_rows(
        &load_idx::<f32>(&fs::read(format!("{}/{}", dataset_dir, TRAIN_LABELS_PATH)).unwrap())
            .unwrap()
            .map(|x| x + 1.0),
        TRAIN_SAMPLES_COUNT,
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

fn train_network<T: Float>(mnist: &MNIST<T>, parallel: bool) -> NeuralNetwork<T> {
    let mut network = NeuralNetwork::<T>::new(vec![28 * 28, 64, 10]);
    let mut train_params = TrainParameters::defaults();
    train_params.show_progress = true;
    train_params.max_epochs = 100;
    train_params.batch_size = 16;
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
                               .value_name("NETWORK_FILE")
                               .help("The file where to store the network (default: network.idx)")
                               .takes_value(true))
                      .arg(Arg::with_name("dataset")
                               .long("dataset")
                               .short("d")
                               .value_name("DATASET_DIR")
                               .help("The directory where the MNIST dataset is stored (default: aicourse-train/datasets/mnist)")
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

    let mnist = load_mnist(
        matches
            .value_of("dataset")
            .unwrap_or("aicourse-train/datasets/mnist"),
    );

    let network = if matches.is_present("test-only") {
        NeuralNetwork::<f32>::load(&load_idx(&fs::read(&network_file).unwrap()).unwrap())
    } else {
        train_network(&mnist, !matches.is_present("sequential"))
    };

    println!(
        "Train accuracy: {}\nTest accuracy: {}",
        accuracy(&network.run(&mnist.train_images), &mnist.train_labels),
        accuracy(&network.run(&mnist.test_images), &mnist.test_labels)
    );

    fs::write(&network_file, save_idx(&network.save(), DataType::F32)).unwrap();
}
