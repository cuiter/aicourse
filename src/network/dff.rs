use crate::matrix::{Float, Matrix};

#[derive(Copy, Clone)]
pub enum CostMethod {
    CostGradient,
    Delta,
}

/// Configurable parameters for training the neural network.
#[derive(Clone)]
pub struct TrainParameters<T: Float> {
    pub regularization_factor: T, // Regularization factor (λ). Higher suppresses variance but also increases bias.
    pub max_epochs: u32,          // The maximum number of epochs before stopping.
    pub cost_epsilon: T,          // Minimum cost before stopping.
    pub cost_method: CostMethod,  // Method to determine the cost.
    pub batch_size: u32,          // For each epoch the inputs are split into batches of this size.
    pub shuffle_inputs: bool,     // Shuffle the inputs on each epoch. NOTE: Not yet implemented.
    pub show_progress: bool,      // Show the progress on each epoch.
}

impl<T: Float> TrainParameters<T> {
    /// Sane defaults for testing.
    pub fn defaults() -> TrainParameters<T> {
        TrainParameters {
            regularization_factor: T::from_f64(0.0005).unwrap(),
            max_epochs: std::u32::MAX - 1,
            cost_epsilon: T::from_f64(0.0001).unwrap(),
            cost_method: CostMethod::Delta,
            batch_size: std::u32::MAX,
            shuffle_inputs: false,
            show_progress: false,
        }
    }
}

pub trait DFFNetwork<T: Float>: Clone {
    /// Trains the neural network with the given input and output data (test dataset).
    fn train(
        &mut self,
        inputs: &Matrix<T>,
        expected_output_classes: &Matrix<T>,
        params: TrainParameters<T>,
    );

    /// Runs the neural network model on the inputs and returns
    /// the classifications with the highest probability.
    fn run(&self, inputs: &Matrix<T>) -> Matrix<T>;
}
