use crate::matrix::{Float, Matrix};
use crate::network::dff::*;
use crate::util::accuracy;
use rayon::prelude::*;

/// Trains the network using different regularization values in parallel.
pub fn train_parallel<T: Float, D: DFFNetwork<T>>(
    network: &D,
    inputs: &Matrix<T>,
    expected_output_classes: &Matrix<T>,
    params: TrainParameters<T>,
) -> D {
    let regularization_factors: Vec<T> = (-1i32..11i32)
        .map(|x| {
            if x == -1 {
                T::zero()
            } else {
                T::powf(T::from_u32(2).unwrap(), T::from_i32(x).unwrap())
                    * T::from_f64(0.01).unwrap()
            }
        })
        .collect();
    let mut trained_networks: Vec<D> = regularization_factors
        .par_iter()
        .map(move |x| {
            let mut new_params = params.clone();
            new_params.regularization_factor = *x;
            let mut new_network = network.clone();
            new_network.train(inputs, expected_output_classes, new_params);
            new_network
        })
        .collect();

    // TODO: Split train, CV, test samples.
    trained_networks.sort_by(|a, b| {
        let a_accuracy = accuracy(&a.run(inputs), expected_output_classes);
        let b_accuracy = accuracy(&b.run(inputs), expected_output_classes);
        a_accuracy.partial_cmp(&b_accuracy).unwrap()
    });

    trained_networks.pop().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::dff_logistic::*;
    use crate::testdata::dff_logistic::*;

    #[test]
    fn train_and_run() {
        let network = NeuralNetwork::<f64>::new_seeded(vec![2, 5, 5, 4], 420);
        for i in 0..tests_inputs().len() {
            let inputs = &tests_inputs()[i];
            let correct_outputs = &tests_outputs()[i];
            let mut train_params = TrainParameters::defaults();
            // The datasets require extremely small weight adjustments, so run for 200 epochs.
            train_params.cost_epsilon = 0.0;
            train_params.max_epochs = 200;
            let par_network = train_parallel(&network, inputs, correct_outputs, train_params);

            let outputs = par_network.run(inputs);
            assert_eq!(&outputs, correct_outputs);
        }
    }
}
