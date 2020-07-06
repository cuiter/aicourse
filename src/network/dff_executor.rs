use crate::matrix::{Float, Matrix};
use crate::network::dff::*;

pub fn train<T: Float, D: DFFNetwork<T>>(network: &D, params: TrainParameters<T>) {}
