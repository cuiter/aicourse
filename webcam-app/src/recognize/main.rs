extern crate opencv;

use opencv::prelude::*;
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use opencv::highgui::{named_window, imshow, wait_key, WINDOW_KEEPRATIO};

fn main() {
    let image_name = "/home/casper/Pictures/Screenshots/20170807-001022.png";
    let image = imread(image_name, IMREAD_COLOR).unwrap();
    named_window("opencv", WINDOW_KEEPRATIO).unwrap();
    imshow("opencv", &image).unwrap();
    wait_key(0).unwrap();
}
