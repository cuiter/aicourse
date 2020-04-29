use crate::matrix::{Float, Matrix};
use crate::regression::common::add_zero_feature;
use crate::util::{classify, sigmoid};

/// A deep feed-forward logistic neural network that acts as a classifier.
/// It can take arbitrary many inputs and produce arbitrary many different classifications.
/// It needs to be trained before it can produce useful results.
struct NeuralNetwork<T: Float> {
    configuration: Vec<Matrix<T>>, // One Matrix per layer, layer - 1 in total
}

impl<T: Float> NeuralNetwork<T> {
    fn new(model_size: Vec<u32>) -> NeuralNetwork<T> {
        let mut configuration = vec![];
        for i in 0..(model_size.len() - 1) {
            configuration.push(Matrix::<T>::zero(model_size[i + 1], model_size[i] + 1));
        }

        NeuralNetwork { configuration }
    }

    /// Returns the configuration (Θ) for all layers (1..L).
    fn get_configuration(&self) -> &Vec<Matrix<T>> {
        &self.configuration
    }

    /// Returns the configuration (Θl) for the layer (l) specified.
    /// The layer is 1-indexed, just like the literature.
    fn get_layer_configuration(&self, layer: u32) -> &Matrix<T> {
        &self.configuration[layer as usize - 1]
    }

    /// Returns the amount of units (sj) in the layer (l), not counting the bias unit.
    fn get_layer_n_units(&self, layer: u32) -> u32 {
        if layer == self.get_n_layers() {
            self.configuration.last().unwrap().get_m()
        } else {
            self.get_layer_configuration(layer).get_n() - 1
        }
    }

    /// Returns the amount of layers (L) in the network.
    fn get_n_layers(&self) -> u32 {
        (self.configuration.len() + 1) as u32
    }

    /// Returns the amount of outputs (K) in the network.
    fn get_n_outputs(&self) -> u32 {
        self.configuration.last().unwrap().get_m()
    }

    /// Calculates a(l) for layer l given the inputs.
    fn calculate_a(&self, inputs: &Matrix<T>, layer: u32) -> Matrix<T> {
        let a_input = if layer == 2 {
            inputs.transpose()
        } else {
            self.calculate_a(inputs, layer - 1)
        };
        let a_input_one = Matrix::one(1, inputs.get_m()).v_concat(&a_input);
        dbg!(&layer);
        dbg!(&a_input_one);
        let z = self.get_layer_configuration(layer - 1) * &a_input_one;
        let a = z.map(sigmoid);

        a
    }

    fn run_extended(&self, inputs: &Matrix<T>) -> Matrix<T> {
        self.calculate_a(inputs, self.get_n_layers())
    }

    fn run(&self, inputs: &Matrix<T>) -> Matrix<T> {
        classify(&self.run_extended(inputs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let network = NeuralNetwork::<f32>::new(vec![3, 5, 5, 4]);
        assert_eq!(network.get_configuration()[0], Matrix::<f32>::zero(5, 4));
        assert_eq!(network.get_configuration()[1], Matrix::<f32>::zero(5, 6));
        assert_eq!(network.get_configuration()[2], Matrix::<f32>::zero(4, 6));
    }

    #[test]
    #[should_panic]
    fn new_wrong_1() {
        let network = NeuralNetwork::<f32>::new(vec![3, 5, 5, 0]);
        assert_eq!(network.get_configuration()[0], Matrix::<f32>::zero(5, 4));
    }

    #[test]
    #[should_panic]
    fn new_wrong_2() {
        let network = NeuralNetwork::<f32>::new(vec![3, 5, 0, 4]);
        assert_eq!(network.get_configuration()[0], Matrix::<f32>::zero(5, 4));
    }

    #[test]
    fn get_layer_configuration() {
        let network = NeuralNetwork::<f32>::new(vec![3, 5, 5, 4]);
        assert_eq!(
            network.get_layer_configuration(1),
            &Matrix::<f32>::zero(5, 4)
        );
        assert_eq!(
            network.get_layer_configuration(2),
            &Matrix::<f32>::zero(5, 6)
        );
        assert_eq!(
            network.get_layer_configuration(3),
            &Matrix::<f32>::zero(4, 6)
        );
    }

    #[test]
    #[should_panic]
    fn get_layer_configuration_wrong() {
        let network = NeuralNetwork::<f32>::new(vec![3, 5, 5, 4]);
        assert_eq!(
            network.get_layer_configuration(4),
            &Matrix::<f32>::zero(4, 6)
        );
    }

    #[test]
    fn get_layer_n_units() {
        let network = NeuralNetwork::<f32>::new(vec![3, 5, 5, 4]);
        assert_eq!(network.get_layer_n_units(1), 3);
        assert_eq!(network.get_layer_n_units(2), 5);
        assert_eq!(network.get_layer_n_units(3), 5);
        assert_eq!(network.get_layer_n_units(4), 4);
    }

    #[test]
    #[should_panic]
    fn get_layer_n_units_wrong() {
        let network = NeuralNetwork::<f32>::new(vec![3, 5, 5, 4]);
        assert_eq!(network.get_layer_n_units(5), 4);
    }

    #[test]
    fn get_n_layers() {
        let network = NeuralNetwork::<f32>::new(vec![3, 5, 5, 4]);
        assert_eq!(network.get_n_layers(), 4);
    }

    #[test]
    fn get_n_outputs() {
        let network = NeuralNetwork::<f32>::new(vec![3, 5, 5, 2]);
        assert_eq!(network.get_n_outputs(), 2);
    }

    #[test]
    fn run_extended_empty() {
        let network = NeuralNetwork::<f32>::new(vec![3, 5, 5, 4]);
        let inputs = Matrix::new(2, 3, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            network.run_extended(&inputs),
            Matrix::new(4, 2, vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        );
    }
}
