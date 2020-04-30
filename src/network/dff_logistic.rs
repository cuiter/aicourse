#![allow(dead_code)]
use crate::matrix::{Float, Matrix};
use crate::util::{classify, sigmoid, unclassify};
use rand::Rng;

/// A deep feed-forward logistic neural network that acts as a classifier.
/// It can take arbitrary many inputs and produce arbitrary many different classifications.
/// It needs to be trained before it can produce useful results.
struct NeuralNetwork<T: Float> {
    configuration: Vec<Matrix<T>>, // One Matrix per layer, layer - 1 in total
}

impl<T: Float> NeuralNetwork<T> {
    /// Creates a new randomized neural network model with the specified size.
    /// The sizes are specified per-layer in order.
    pub fn new(model_size: Vec<u32>) -> NeuralNetwork<T> {
        let mut rng = rand::thread_rng();
        let mut configuration = vec![];
        for i in 0..(model_size.len() - 1) {
            let mut matrix = Matrix::<T>::zero(model_size[i + 1], model_size[i] + 1);
            for m in 0..matrix.get_m() {
                for n in 0..matrix.get_n() {
                    matrix[(m, n)] = T::from_f64(rng.gen::<f64>() * 0.24 - 0.12).unwrap();
                }
            }
            configuration.push(matrix);
        }

        NeuralNetwork { configuration }
    }

    pub fn new_empty(model_size: Vec<u32>) -> NeuralNetwork<T> {
        let mut configuration = vec![];
        for i in 0..(model_size.len() - 1) {
            configuration.push(Matrix::<T>::zero(model_size[i + 1], model_size[i] + 1));
        }

        NeuralNetwork { configuration }
    }

    /// Creates a new neural network from an existing configuration.
    fn from_configuration(configuration: Vec<Matrix<T>>) -> NeuralNetwork<T> {
        NeuralNetwork { configuration }
    }

    /// Clones the configuration structure of the neural network and sets it to zero.
    fn clone_empty(&self) -> Vec<Matrix<T>> {
        self.configuration
            .iter()
            .map(|matrix| Matrix::<T>::zero(matrix.get_m(), matrix.get_n()))
            .collect()
    }

    /// Returns the configuration (Θ) for all layers (1..L).
    pub fn get_configuration(&self) -> &Vec<Matrix<T>> {
        &self.configuration
    }

    /// Returns the configuration (Θl) for the layer (l) specified.
    /// The layer is 1-indexed, just like the literature.
    pub fn get_layer_configuration(&self, layer: u32) -> &Matrix<T> {
        &self.configuration[layer as usize - 1]
    }

    /// Returns the amount of units (sj) in the layer (l), not counting the bias unit.
    pub fn get_layer_n_units(&self, layer: u32) -> u32 {
        if layer == self.get_n_layers() {
            self.configuration.last().unwrap().get_m()
        } else {
            self.get_layer_configuration(layer).get_n() - 1
        }
    }

    /// Returns the amount of layers (L) in the network.
    pub fn get_n_layers(&self) -> u32 {
        (self.configuration.len() + 1) as u32
    }

    /// Returns the amount of outputs (K) in the network.
    pub fn get_n_outputs(&self) -> u32 {
        self.configuration.last().unwrap().get_m()
    }

    /// Calculates a(l) for layer l given the inputs.
    fn calculate_a(&self, inputs: &Matrix<T>, layer: u32) -> Matrix<T> {
        if layer == 1 {
            inputs.transpose()
        } else {
            let a_input = self.calculate_a(inputs, layer - 1);
            let a_input_one = Matrix::one(1, inputs.get_m()).v_concat(&a_input);
            let z = self.get_layer_configuration(layer - 1) * &a_input_one;
            let a = z.map(sigmoid);

            a
        }
    }

    /// Calculates a(l), including the bias unit, for layer l given the inputs.
    fn calculate_a_with_bias(&self, inputs: &Matrix<T>, layer: u32) -> Matrix<T> {
        let a = self.calculate_a(inputs, layer);
        Matrix::one(1, a.get_n()).v_concat(&a)
    }

    /// See https://www.youtube.com/watch?v=0twSSFZN9Mc&t=3m44s
    fn cost(
        &self,
        inputs: &Matrix<T>,
        expected_outputs: &Matrix<T>,
        regularization_factor: T,
    ) -> T {
        let mut error_sum = T::zero();

        let hypothesis = self.run_extended(inputs);
        let one = T::from_u8(1).unwrap();

        for i in 0..inputs.get_m() {
            for k in 0..expected_outputs.get_n() {
                error_sum = error_sum
                    + expected_outputs[(i, k)] * T::ln(hypothesis[(i, k)])
                    + (one - expected_outputs[(i, k)]) * T::ln(one - hypothesis[(i, k)]);
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

    fn cost_gradient(
        &self,
        inputs: &Matrix<T>,
        expected_outputs: &Matrix<T>,
        regularization_factor: T,
    ) -> Vec<Matrix<T>> {
        let cost_epsilon = T::from_f64(0.0001).unwrap();

        let mut cost_gradient = self.clone_empty();

        for l in 1..self.get_n_layers() {
            for i in 0..self.get_layer_n_units(l + 1) {
                for j in 0..self.get_layer_n_units(l) + 1 {
                    let mut left_configuration = self.configuration.clone();
                    left_configuration[(l - 1) as usize][(i, j)] =
                        left_configuration[(l - 1) as usize][(i, j)] - cost_epsilon;
                    let left_cost = NeuralNetwork::from_configuration(left_configuration).cost(
                        inputs,
                        expected_outputs,
                        regularization_factor,
                    );
                    let mut right_configuration = self.configuration.clone();
                    right_configuration[(l - 1) as usize][(i, j)] =
                        right_configuration[(l - 1) as usize][(i, j)] + cost_epsilon;
                    let right_cost = NeuralNetwork::from_configuration(right_configuration).cost(
                        inputs,
                        expected_outputs,
                        regularization_factor,
                    );

                    cost_gradient[(l - 1) as usize][(i, j)] =
                        (right_cost - left_cost) / (T::from_f64(2.0).unwrap() * cost_epsilon);
                }
            }
        }

        cost_gradient
    }

    fn backprop_error(
        &self,
        inputs: &Matrix<T>,
        expected_outputs: &Matrix<T>,
        layer: u32,
    ) -> Matrix<T> {
        if layer == self.get_n_layers() {
            &self.calculate_a(inputs, layer) - &expected_outputs.transpose()
        } else {
            let left = self.get_layer_configuration(layer).transpose();
            let right = self.backprop_error(inputs, expected_outputs, layer + 1);
            let first_half_with_bias = &left * &right;
            let a = self.calculate_a_with_bias(inputs, layer);
            let second_half = a.elem_mul(&(&Matrix::<T>::one(a.get_m(), a.get_n()) - &a));

            let error = first_half_with_bias.elem_mul(&second_half);
            // Remove error0
            error.get_sub_matrix(1, 0, error.get_m() - 1, error.get_n())
        }
    }

    fn delta(
        &self,
        inputs: &Matrix<T>,
        expected_outputs: &Matrix<T>,
        regularization_factor: T,
    ) -> Vec<Matrix<T>> {
        let mut d = vec![];

        for _i in 0..inputs.get_m() {
            for l in 1..self.get_n_layers() {
                let delta_add = &self.backprop_error(inputs, &expected_outputs, l + 1)
                    * &self.calculate_a_with_bias(inputs, l).transpose();

                if d.len() < l as usize {
                    d.push(delta_add);
                } else {
                    d[(l - 1) as usize] += &(&delta_add / T::from_u32(inputs.get_m()).unwrap());
                }
            }
        }

        let mut big_d: Vec<Matrix<T>> = d
            .iter()
            .map(|matrix| matrix / T::from_u32(inputs.get_m()).unwrap())
            .collect();
        if regularization_factor != T::zero() {
            for l in 1..self.get_n_layers() {
                let layer_configuration = self.get_layer_configuration(l);
                big_d[(l - 1) as usize] = &big_d[(l - 1) as usize]
                    // TODO: remove bias units
                    + &(layer_configuration * regularization_factor);
            }
        }

        big_d
    }

    fn descend(
        &self,
        inputs: &Matrix<T>,
        expected_outputs: &Matrix<T>,
        regularization_factor: T,
    ) -> NeuralNetwork<T> {
        let d = self.delta(inputs, expected_outputs, regularization_factor);
        let new_configuration = self
            .configuration
            .iter()
            .zip(d.iter())
            .map(|(m1, m2)| m1 + m2)
            .collect();;

        NeuralNetwork::from_configuration(new_configuration)
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

        let cost_epsilon = T::from_f64(0.0001).unwrap();

        loop {
            let cost = self.cost(inputs, &expected_outputs, regularization_factor);

            let new_network = self.descend(inputs, &expected_outputs, regularization_factor);
            let new_cost = new_network.cost(inputs, &expected_outputs, regularization_factor);

            self.configuration = new_network.get_configuration().clone();

            if T::abs(new_cost - cost) < cost_epsilon {
                break;
            }
        }
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
        let network = NeuralNetwork::<f64>::new_empty(vec![3, 5, 5, 4]);
        assert_eq!(network.get_configuration()[0], Matrix::<f64>::zero(5, 4));
        assert_eq!(network.get_configuration()[1], Matrix::<f64>::zero(5, 6));
        assert_eq!(network.get_configuration()[2], Matrix::<f64>::zero(4, 6));
    }

    #[test]
    #[should_panic]
    fn new_wrong_1() {
        let network = NeuralNetwork::<f64>::new_empty(vec![3, 5, 5, 0]);
        assert_eq!(network.get_configuration()[0], Matrix::<f64>::zero(5, 4));
    }

    #[test]
    #[should_panic]
    fn new_wrong_2() {
        let network = NeuralNetwork::<f64>::new_empty(vec![3, 5, 0, 4]);
        assert_eq!(network.get_configuration()[0], Matrix::<f64>::zero(5, 4));
    }

    #[test]
    fn get_layer_configuration() {
        let network = NeuralNetwork::<f64>::new_empty(vec![3, 5, 5, 4]);
        assert_eq!(
            network.get_layer_configuration(1),
            &Matrix::<f64>::zero(5, 4)
        );
        assert_eq!(
            network.get_layer_configuration(2),
            &Matrix::<f64>::zero(5, 6)
        );
        assert_eq!(
            network.get_layer_configuration(3),
            &Matrix::<f64>::zero(4, 6)
        );
    }

    #[test]
    #[should_panic]
    fn get_layer_configuration_wrong() {
        let network = NeuralNetwork::<f64>::new_empty(vec![3, 5, 5, 4]);
        assert_eq!(
            network.get_layer_configuration(4),
            &Matrix::<f64>::zero(4, 6)
        );
    }

    #[test]
    fn get_layer_n_units() {
        let network = NeuralNetwork::<f64>::new(vec![3, 5, 5, 4]);
        assert_eq!(network.get_layer_n_units(1), 3);
        assert_eq!(network.get_layer_n_units(2), 5);
        assert_eq!(network.get_layer_n_units(3), 5);
        assert_eq!(network.get_layer_n_units(4), 4);
    }

    #[test]
    #[should_panic]
    fn get_layer_n_units_wrong() {
        let network = NeuralNetwork::<f64>::new(vec![3, 5, 5, 4]);
        assert_eq!(network.get_layer_n_units(5), 4);
    }

    #[test]
    fn get_n_layers() {
        let network = NeuralNetwork::<f64>::new(vec![3, 5, 5, 4]);
        assert_eq!(network.get_n_layers(), 4);
    }

    #[test]
    fn get_n_outputs() {
        let network = NeuralNetwork::<f64>::new(vec![3, 5, 5, 2]);
        assert_eq!(network.get_n_outputs(), 2);
    }

    #[test]
    fn run_extended_empty() {
        let network = NeuralNetwork::<f64>::new_empty(vec![3, 5, 5, 4]);
        let inputs = Matrix::new(2, 3, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            network.run_extended(&inputs),
            Matrix::new(2, 4, vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        );
    }

    // #[test]
    // fn train_and_run() {
    //     let mut network = NeuralNetwork::<f64>::new(vec![2, 5, 5, 4]);
    //     for i in 0..tests_inputs().len() {
    //         let inputs = &tests_inputs()[i];
    //         let correct_outputs = &tests_outputs()[i];
    //         network.train(inputs, correct_outputs, 0.0);

    //         let outputs = network.run(inputs);
    //         assert_eq!(&outputs, correct_outputs);
    //     }
    // }
    //
    #[test]
    fn calculate_a() {
        let network = NeuralNetwork::<f64>::new(vec![2, 5, 5, 4]);
        for i in 0..tests_inputs().len() {
            let inputs = &tests_inputs()[i];

            for l in 2..=network.get_n_layers() {
                let a = network.calculate_a(inputs, l);
                assert_eq!(a.get_m(), network.get_layer_n_units(l));
                assert_eq!(a.get_n(), inputs.get_m());
                for val in a.iter() {
                    assert!(val.is_finite(), "a value is finite");
                }
            }
        }
    }

    #[test]
    fn backprop() {
        let network = NeuralNetwork::<f64>::new(vec![2, 5, 5, 4]);
        for i in 0..tests_inputs().len() {
            let inputs = &tests_inputs()[i];
            let correct_outputs = &tests_outputs()[i];

            for l in 2..=network.get_n_layers() {
                let error = network.backprop_error(inputs, &unclassify(correct_outputs), l);
                assert_eq!(error.get_m(), network.get_layer_n_units(l));
                assert_eq!(error.get_n(), inputs.get_m());
                for delta in error.iter() {
                    assert!(delta.is_finite(), "error is finite");
                }
            }
        }
    }

    #[test]
    fn cost() {
        let network = NeuralNetwork::<f64>::new(vec![2, 5, 5, 4]);
        for i in 0..tests_inputs().len() {
            let inputs = &tests_inputs()[i];
            let correct_outputs = &tests_outputs()[i];

            let n_cost = network.cost(inputs, &unclassify(correct_outputs), 0.0);
            assert!(
                n_cost.is_finite() && n_cost >= 0.0,
                "cost is finite and positive"
            );
        }
    }

    #[test]
    fn cost_gradient() {
        let network = NeuralNetwork::<f64>::new(vec![2, 5, 5, 4]);
        for i in 0..tests_inputs().len() {
            let inputs = &tests_inputs()[i];
            let correct_outputs = &tests_outputs()[i];

            let gradient = network.cost_gradient(inputs, &unclassify(correct_outputs), 0.0);
            assert_eq!(gradient.len(), network.get_n_layers() as usize - 1);

            for l in 1..network.get_n_layers() {
                for i in 0..network.get_layer_n_units(l + 1) {
                    for j in 0..network.get_layer_n_units(l) + 1 {
                        assert!(gradient[(l - 1) as usize][(i, j)].is_finite());
                    }
                }
            }
        }
    }

    #[test]
    fn delta() {
        let network = NeuralNetwork::<f64>::new(vec![2, 5, 5, 4]);
        for i in 0..tests_inputs().len() {
            let inputs = &tests_inputs()[i];
            let correct_outputs = &tests_outputs()[i];

            let gradient = network.delta(inputs, &unclassify(correct_outputs), 0.0);
            assert_eq!(gradient.len(), network.get_n_layers() as usize - 1);
            dbg!(&gradient);
            dbg!(&network.cost_gradient(inputs, &unclassify(correct_outputs), 0.0));
            for l in 1..network.get_n_layers() {
                for i in 0..network.get_layer_n_units(l + 1) {
                    for j in 0..network.get_layer_n_units(l) + 1 {
                        assert!(gradient[(l - 1) as usize][(i, j)].is_finite());
                    }
                }
            }
        }
    }

    #[test]
    fn delta_cost_gradient_equal() {
        // TODO: Fix incorrect calculation.
        // TODO: Fix incorrect regularization.
        // It seems to be exactly that delta == cost_gradient * (2 - 1 / m)

        let network = NeuralNetwork::<f64>::new(vec![2, 5, 5, 4]);
        for i in 0..tests_inputs().len() {
            let inputs = &tests_inputs()[i];
            let correct_outputs = &tests_outputs()[i];

            let delta = &network.delta(inputs, &unclassify(correct_outputs), 0.0);
            let cost_gradient = network.cost_gradient(inputs, &unclassify(correct_outputs), 0.0);
            for l in 1..network.get_n_layers() {
                assert!(
                    delta[(l - 1) as usize].approx_eq(&cost_gradient[(l - 1) as usize], 0.0002),
                    "(layer {}) delta == cost gradient\ndelta = {}\ncost gradient = {}",
                    l,
                    &delta[(l - 1) as usize],
                    &cost_gradient[(l - 1) as usize]
                );
            }
        }
    }
}
