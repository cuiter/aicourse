use crate::network::dff::*;
use crate::matrix::{Float, Matrix};

pub fn train<T: Float, D: DFFNetwork<T>>(network: &D, params: TrainParameters<T>) {
}
