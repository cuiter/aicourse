use crate::matrix::{Float, Matrix};

#[derive(Copy, Clone)]
pub enum CostMethod {
    CostGradient,
    Delta,
}

/// Configurable parameters for training the neural network.
#[derive(Clone)]
pub struct TrainParameters<T: Float> {
    pub regularization_factor: T,
    pub max_epochs: u32,
    pub cost_epsilon: T,
    pub cost_method: CostMethod,
    pub show_progress: bool,
}

impl<T: Float> TrainParameters<T> {
    /// Sane defaults for testing.
    pub fn defaults() -> TrainParameters<T> {
        TrainParameters {
            regularization_factor: T::from_f64(0.0005).unwrap(),
            max_epochs: std::u32::MAX - 1,
            cost_epsilon: T::from_f64(0.0001).unwrap(),
            cost_method: CostMethod::Delta,
            show_progress: false,
        }
    }
}

pub trait DFFNetwork<T : Float>: Clone {
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
