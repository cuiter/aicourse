use crate::matrix::{Float, Matrix};
use crate::regression::common::add_zero_feature;
use crate::util::{classify, sigmoid, unclassify};

/// A deep feed-forward logistic neural network that acts as a classifier.
/// It can take arbitrary many inputs and produce arbitrary many different classifications.
/// It needs to be trained before it can produce useful results.
struct NeuralNetwork<T: Float> {
    configuration: Vec<Matrix<T>>, // One Matrix per layer, layer - 1 in total
}

impl<T: Float> NeuralNetwork<T> {
    /// Creates a new empty neural network model with the specified size.
    /// The sizes are specified per-layer in order.
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
        let z = self.get_layer_configuration(layer - 1) * &a_input_one;
        let a = z.map(sigmoid);

        a
    }

    /// See https://www.youtube.com/watch?v=0twSSFZN9Mc&t=3m44s
    fn cost(&self, inputs: &Matrix<T>, correct_outputs: &Matrix<T>, regularization_factor: T) -> T {
        let mut error_sum = T::zero();

        let hypothesis = self.run_extended(inputs);
        let one = T::from_u8(1).unwrap();

        for i in 0..inputs.get_m() {
            for k in 0..correct_outputs.get_n() {
                error_sum = error_sum
                    + correct_outputs[(i, k)] * T::ln(hypothesis[(i, k)])
                    + (one - correct_outputs[(i, k)]) * T::ln(one - hypothesis[(i, k)]);
            }
        }

        let normal_cost = (-one / T::from_u32(inputs.get_m()).unwrap()) * error_sum;

        if regularization_factor == T::zero() {
            normal_cost
        } else {
            let configuration_squared_sum: T = self
                .configuration
                .iter()
                .map(|configuration_layer| configuration_layer.iter())
                .flatten()
                .map(|&x| x * x)
                .fold(T::zero(), |sum, val| sum + val);

            normal_cost
                + (regularization_factor / T::from_u32(inputs.get_m() * 2).unwrap())
                    * configuration_squared_sum
        }
    }

    fn backprop_error(
        &self,
        inputs: &Matrix<T>,
        expected_outputs: &Matrix<T>,
        layer: u32,
    ) -> Matrix<T> {
        if layer == self.get_n_layers() {
            &self.calculate_a(inputs, layer) - expected_outputs
        } else {
            let first_half = &self.get_layer_configuration(layer).transpose()
                * &self.backprop_error(inputs, expected_outputs, layer + 1);
            let a = self.calculate_a(inputs, layer);
            let second_half = a.elem_mul(&(&Matrix::<T>::one(a.get_m(), a.get_n()) - &a));

            dbg!(
                first_half.get_m(),
                first_half.get_n(),
                second_half.get_m(),
                second_half.get_n()
            );
            first_half.elem_mul(&second_half)
        }
    }

    pub fn train(
        &mut self,
        inputs: &Matrix<T>,
        expected_output_classes: &Matrix<T>,
        regularization_factor: T,
    ) {
        let expected_outputs = unclassify(expected_output_classes);

        assert_eq!(
            inputs.get_m(),
            expected_outputs.get_m(),
            "number of inputs equals number of outputs"
        );
        assert_eq!(
            inputs.get_n(),
            self.get_layer_n_units(1),
            "number of input features equals number of units in first layer"
        );
        assert_eq!(
            expected_outputs.get_n(),
            self.get_n_outputs(),
            "number of output classifications equals number of units in last layer"
        );
    }

    /// Runs the neural network model on the inputs and returns
    /// the probabilities for all different classifications.
    fn run_extended(&self, inputs: &Matrix<T>) -> Matrix<T> {
        self.calculate_a(inputs, self.get_n_layers()).transpose()
    }

    /// Runs the neural network model on the inputs and returns
    /// the classifications with the highest probability.
    pub fn run(&self, inputs: &Matrix<T>) -> Matrix<T> {
        classify(&self.run_extended(inputs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata::dff_logistic::*;

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
            Matrix::new(2, 4, vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        );
    }

    #[test]
    fn train_and_run() {
        let mut network = NeuralNetwork::<f32>::new(vec![2, 5, 5, 4]);
        for i in 0..tests_inputs().len() {
            let inputs = &tests_inputs()[i];
            let correct_outputs = &tests_outputs()[i];
            network.train(inputs, correct_outputs, 0.0);

            let outputs = network.run(inputs);
            assert_eq!(&outputs, correct_outputs);
        }
    }

    /*
    #[test]
    fn backprop() {
        let mut network = NeuralNetwork::<f32>::new(vec![2, 5, 5, 4]);
        for i in 0..tests_inputs().len() {
            let inputs = &tests_inputs()[i];
            let correct_outputs = &tests_outputs()[i];
            network.train(inputs, correct_outputs, 0.0);

            let error = network.backprop_error(inputs, &unclassify(correct_outputs), 2);
            assert_eq!(error, Matrix::zero(inputs.get_m(), 1));
        }
    }
    */

    #[test]
    fn cost() {
        let mut network = NeuralNetwork::<f32>::new(vec![2, 5, 5, 4]);
        for i in 0..tests_inputs().len() {
            let inputs = &tests_inputs()[i];
            let correct_outputs = &tests_outputs()[i];
            network.train(inputs, correct_outputs, 0.0);

            let n_cost = network.cost(inputs, &unclassify(correct_outputs), 0.0);
            assert!(
                n_cost.is_finite() && n_cost >= 0.0,
                "cost is finite and positive"
            );
        }
    }
}
