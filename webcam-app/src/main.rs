use aicourse::matrix::*;
use aicourse::network::dff::DFFNetwork;
use aicourse::network::dff_logistic::NeuralNetwork;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::{Color, PixelFormatEnum};
use sdl2::rect::Rect;
use sdl2::render::{Canvas, Texture};
use sdl2::surface::Surface;
use sdl2::video::Window;
use std::fs;
use std::time::Duration;

const IMG_WIDTH: u32 = 800;
const IMG_HEIGHT: u32 = 600;
const FOCUS_WIDTH: u32 = 28;
const FOCUS_HEIGHT: u32 = 28;

fn convert_frame(frame: &[u8], display_frame: &mut [u8]) {
    for idx in 0..(frame.len() / 3) {
        // Apparently SDL2 RGB888 is really BGRX8888
        display_frame[idx * 4 + 2] = frame[idx * 3 + 0];
        display_frame[idx * 4 + 1] = frame[idx * 3 + 1];
        display_frame[idx * 4 + 0] = frame[idx * 3 + 2];
    }
}

fn draw_fdisp(canvas: &mut Canvas<Window>, fdisp_texture: &Texture, focus_area: Rect) {
    canvas.copy(&fdisp_texture, None, focus_area).unwrap();
    canvas.set_draw_color(Color::RGB(255, 0, 0));
    canvas.draw_rect(focus_area).unwrap();
}

fn extract_focus<T: Float>(
    frame_texture: &Texture,
    focus_area: Rect,
    focus_canvas: &mut Canvas<Window>,
    focus_matrix: &mut Matrix<T>,
) {
    focus_canvas
        .copy(
            frame_texture,
            focus_area,
            Rect::new(0, 0, FOCUS_WIDTH, FOCUS_HEIGHT),
        )
        .unwrap();
    let pixels = focus_canvas
        .read_pixels(None, PixelFormatEnum::RGB888)
        .unwrap();
    for idx in 0..pixels.len() / 4 {
        focus_matrix[(0, idx as u32)] = (T::from_u8(pixels[idx * 4 + 0]).unwrap()
            + T::from_u8(pixels[idx * 4 + 1]).unwrap()
            + T::from_u8(pixels[idx * 4 + 2]).unwrap())
            / T::from_u8(3).unwrap();
    }
}

fn write_fdisp<T: Float>(fdisp_buffer: &mut [u8], focus_matrix: &Matrix<T>) {
    for y in 0..FOCUS_HEIGHT {
        for x in 0..FOCUS_WIDTH {
            let idx = (x * FOCUS_WIDTH + y) as usize;
            fdisp_buffer[idx * 4 + 0] = focus_matrix[(0, idx as u32)].to_u8().unwrap();
            fdisp_buffer[idx * 4 + 1] = focus_matrix[(0, idx as u32)].to_u8().unwrap();
            fdisp_buffer[idx * 4 + 2] = focus_matrix[(0, idx as u32)].to_u8().unwrap();
        }
    }
}

fn contrast_stretch<T: Float>(matrix: &Matrix<T>, bottom: T, top: T) -> Matrix<T> {
    let mut min = T::infinity();
    let mut max = T::zero();
    for value in matrix.iter() {
        if *value < min {
            min = *value;
        }
        if *value > max {
            max = *value;
        }
    }

    matrix.map(|value| bottom + ((value - min) / (max - min)) * (top - bottom))
}

fn preprocess_focus<T: Float>(focus_matrix: &Matrix<T>) -> Matrix<T> {
    contrast_stretch(
        &focus_matrix,
        T::from_u8(0).unwrap(),
        T::from_u8(255).unwrap(),
    )
}

pub fn main() {
    let network =
        NeuralNetwork::load(&load_idx::<f32>(&fs::read("../network.idx").unwrap()).unwrap());

    let mut camera = rscam::new("/dev/video0").unwrap();
    let focus_area = Rect::new(
        IMG_WIDTH as i32 / 2 - 100,
        IMG_HEIGHT as i32 / 2 - 100,
        200,
        200,
    );
    camera
        .start(&rscam::Config {
            interval: (1, 30),
            resolution: (IMG_WIDTH, IMG_HEIGHT),
            format: b"RGB3",
            ..Default::default()
        })
        .unwrap();

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window("rust-sdl2 demo", IMG_WIDTH, IMG_HEIGHT)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();
    let texture_creator = canvas.texture_creator();
    assert!(PixelFormatEnum::RGB888.byte_size_per_pixel() == 4);

    let frame_surface = Surface::new(IMG_WIDTH, IMG_HEIGHT, PixelFormatEnum::RGB888).unwrap();
    let mut frame_texture = Texture::from_surface(&frame_surface, &texture_creator).unwrap();
    let mut frame_buffer = [0u8; (IMG_WIDTH * IMG_HEIGHT * 4) as usize];
    let mut focus_texture = texture_creator
        .create_texture_target(PixelFormatEnum::RGB888, FOCUS_WIDTH, FOCUS_HEIGHT)
        .unwrap();
    let mut focus_matrix = Matrix::zero(1, FOCUS_WIDTH * FOCUS_HEIGHT);
    let fdisp_surface = Surface::new(FOCUS_WIDTH, FOCUS_HEIGHT, PixelFormatEnum::RGB888).unwrap();
    let mut fdisp_texture = Texture::from_surface(&fdisp_surface, &texture_creator).unwrap();
    let mut fdisp_buffer = [0u8; (FOCUS_WIDTH * FOCUS_HEIGHT * 4) as usize];

    let mut event_pump = sdl_context.event_pump().unwrap();
    'running: loop {
        let frame = camera.capture().unwrap();
        convert_frame(&frame, &mut frame_buffer);
        frame_texture
            .update(None, &frame_buffer, (IMG_WIDTH * 4) as usize)
            .unwrap();
        canvas
            .with_texture_canvas(&mut focus_texture, |mut focus_canvas| {
                extract_focus::<f32>(
                    &frame_texture,
                    focus_area,
                    &mut focus_canvas,
                    &mut focus_matrix,
                );
            })
            .unwrap();

        focus_matrix = preprocess_focus(&focus_matrix);
        write_fdisp(&mut fdisp_buffer, &focus_matrix);
        fdisp_texture
            .update(None, &fdisp_buffer, (FOCUS_WIDTH * 4) as usize)
            .unwrap();

        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.clear();
        canvas
            .copy(&frame_texture, None, Rect::new(0, 0, IMG_WIDTH, IMG_HEIGHT))
            .unwrap();
        draw_fdisp(&mut canvas, &fdisp_texture, focus_area);

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }

        println!("Prediction: {}", network.run(&focus_matrix)[(0, 0)] - 1.0);

        canvas.present();
        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
    }
}
