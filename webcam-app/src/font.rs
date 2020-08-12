use sdl2::pixels::Color;
use sdl2::rect::{Point, Rect};
use sdl2::render::Canvas;
use sdl2::video::Window;

const FONT_CH_WIDTH: u32 = 4;
const FONT_CH_HEIGHT: u32 = 5;
const FONT_CH_DATA: &'static [u8] = &[
    0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0,
    0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1,
    0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0,
    1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0,
    0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1,
    0, 0, 0, 1, 0, 1, 1, 0,
];

pub fn draw_digit(
    canvas: &mut Canvas<Window>,
    position: Point,
    scale: u32,
    color: Color,
    digit: u32,
) {
    canvas.set_draw_color(color);
    for y in 0..FONT_CH_HEIGHT {
        for x in 0..FONT_CH_WIDTH {
            if FONT_CH_DATA
                [(digit * FONT_CH_WIDTH * FONT_CH_HEIGHT + y * FONT_CH_WIDTH + x) as usize]
                == 1
            {
                canvas
                    .fill_rect(Rect::new(
                        position.x + (x * scale) as i32,
                        position.y + (y * scale) as i32,
                        scale,
                        scale,
                    ))
                    .unwrap();
            }
        }
    }
}
