use anyhow::{Context, Result};
use base64::Engine;
use minifb::{Key, Window, WindowOptions};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;

const WINDOW_TITLE: &str = "FaceUnlock View — press ESC to close";

const COLOR_GREEN: u32 = 0x00FF00;
const COLOR_RED: u32 = 0xFF0000;
const COLOR_WHITE: u32 = 0xFFFFFF;

struct BBox {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

pub fn run_view(socket_path: &str, user: &str) -> Result<()> {
    let mut stream = UnixStream::connect(socket_path)
        .context("Failed to connect to faceunlockd daemon")?;

    let req = serde_json::json!({"action": "view", "user": user});
    let mut json = serde_json::to_string(&req)?;
    json.push('\n');
    stream.write_all(json.as_bytes())?;

    let mut reader = BufReader::new(&stream);
    let mut first_line = String::new();
    reader.read_line(&mut first_line)?;
    let first: serde_json::Value = serde_json::from_str(first_line.trim())?;

    if first.get("type").and_then(|v| v.as_str()) == Some("error") {
        anyhow::bail!(
            "Daemon error: {}",
            first["message"].as_str().unwrap_or("unknown")
        );
    }

    let width = first["width"]
        .as_u64()
        .context("Missing width in first frame")? as usize;
    let height = first["height"]
        .as_u64()
        .context("Missing height in first frame")? as usize;

    let mut window = Window::new(
        WINDOW_TITLE,
        width,
        height,
        WindowOptions {
            resize: false,
            ..Default::default()
        },
    )
    .context("Failed to create window — is a display available?")?;

    window.limit_update_rate(Some(std::time::Duration::from_millis(33)));

    let mut line = first_line;
    let engine = base64::engine::general_purpose::STANDARD;

    loop {
        if !window.is_open() || window.is_key_down(Key::Escape) {
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            continue;
        }

        let msg: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => {
                line.clear();
                if reader.read_line(&mut line)? == 0 {
                    break;
                }
                continue;
            }
        };

        if msg.get("type").and_then(|v| v.as_str()) != Some("frame") {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            continue;
        }

        let image_b64 = msg["image"].as_str().unwrap_or("");
        let grey_data = engine.decode(image_b64)?;

        let mut buffer = grey_to_rgb_buffer(&grey_data);

        if let Some(face) = msg.get("face") {
            let bbox = BBox {
                x: face["bbox"]["x"].as_f64().unwrap_or(0.0) as f32,
                y: face["bbox"]["y"].as_f64().unwrap_or(0.0) as f32,
                w: face["bbox"]["w"].as_f64().unwrap_or(0.0) as f32,
                h: face["bbox"]["h"].as_f64().unwrap_or(0.0) as f32,
            };

            let authenticated = face["authenticated"].as_bool().unwrap_or(false);
            let score = face["score"].as_f64().unwrap_or(0.0);
            let bbox_color = if authenticated { COLOR_GREEN } else { COLOR_RED };

            draw_bbox(&mut buffer, width, height, &bbox, bbox_color);

            if let Some(landmarks) = face["landmarks"].as_array() {
                for lm in landmarks {
                    if let (Some(x), Some(y)) = (lm[0].as_f64(), lm[1].as_f64()) {
                        draw_circle(&mut buffer, width, height, x as i32, y as i32, 4, COLOR_RED);
                    }
                }
            }

            let conf = face["confidence"].as_f64().unwrap_or(0.0);
            let label = if authenticated {
                format!("PASS {:.0}%", score * 100.0)
            } else {
                format!("FAIL {:.0}%", score * 100.0)
            };
            draw_text_label(
                &mut buffer,
                width,
                height,
                bbox.x as i32,
                (bbox.y - 18.0).max(0.0) as i32,
                &label,
                bbox_color,
            );

            let det_label = format!("det:{:.0}%", conf * 100.0);
            draw_text_label(
                &mut buffer,
                width,
                height,
                bbox.x as i32,
                (bbox.y + bbox.h + 4.0) as i32,
                &det_label,
                COLOR_WHITE,
            );
        }

        window.update_with_buffer(&buffer, width, height)?;

        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
    }

    println!("View closed");
    Ok(())
}

fn grey_to_rgb_buffer(grey: &[u8]) -> Vec<u32> {
    grey.iter()
        .map(|&p| {
            let r = p as u32;
            let g = p as u32;
            let b = p as u32;
            (r << 16) | (g << 8) | b
        })
        .collect()
}

fn draw_bbox(buffer: &mut [u32], width: usize, height: usize, bbox: &BBox, color: u32) {
    let x0 = (bbox.x as i32).max(0) as usize;
    let y0 = (bbox.y as i32).max(0) as usize;
    let x1 = ((bbox.x + bbox.w) as i32).max(0).min(width as i32 - 1) as usize;
    let y1 = ((bbox.y + bbox.h) as i32).max(0).min(height as i32 - 1) as usize;
    let thickness = 2;

    for t in 0..thickness {
        if y0 + t < height {
            for x in x0..=x1.min(width - 1) {
                buffer[(y0 + t) * width + x] = color;
            }
        }
        if y1 >= t {
            for x in x0..=x1.min(width - 1) {
                buffer[(y1 - t) * width + x] = color;
            }
        }
    }

    for t in 0..thickness {
        if x0 + t < width {
            for y in y0..=y1.min(height - 1) {
                buffer[y * width + (x0 + t)] = color;
            }
        }
        if x1 >= t {
            for y in y0..=y1.min(height - 1) {
                buffer[y * width + (x1 - t)] = color;
            }
        }
    }
}

fn draw_circle(buffer: &mut [u32], width: usize, height: usize, cx: i32, cy: i32, radius: i32, color: u32) {
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= radius * radius {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    buffer[py as usize * width + px as usize] = color;
                }
            }
        }
    }
}

fn draw_text_label(buffer: &mut [u32], width: usize, height: usize, x: i32, y: i32, text: &str, color: u32) {
    let char_width = 6;
    let mut offset = 0;
    for ch in text.chars() {
        let glyph = get_glyph(ch);
        for gy in 0..7i32 {
            for gx in 0..5i32 {
                if glyph[gy as usize] & (1 << (4 - gx)) != 0 {
                    let px = x + offset + gx;
                    let py = y + gy;
                    if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                        buffer[py as usize * width + px as usize] = color;
                    }
                }
            }
        }
        offset += char_width;
    }
}

fn get_glyph(ch: char) -> [u8; 7] {
    match ch {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        '3' => [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b01110, 0b00001],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        'a' => [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111],
        'b' => [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110],
        'c' => [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110],
        'd' => [0b00001, 0b00001, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111],
        'e' => [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110],
        'f' => [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000],
        'g' => [0b00000, 0b00000, 0b01111, 0b10001, 0b01111, 0b00001, 0b01110],
        'h' => [0b10000, 0b10000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001],
        'i' => [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110],
        'j' => [0b00010, 0b00000, 0b00110, 0b00010, 0b00010, 0b10010, 0b01100],
        'k' => [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010],
        'l' => [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'm' => [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001],
        'n' => [0b00000, 0b00000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001],
        'o' => [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110],
        'p' => [0b00000, 0b00000, 0b11110, 0b10001, 0b11110, 0b10000, 0b10000],
        'q' => [0b00000, 0b00000, 0b01111, 0b10001, 0b01111, 0b00001, 0b00001],
        'r' => [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000],
        's' => [0b00000, 0b00000, 0b01110, 0b10000, 0b01110, 0b00001, 0b11110],
        't' => [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110],
        'u' => [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101],
        'v' => [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'w' => [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010],
        'x' => [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001],
        'y' => [0b00000, 0b00000, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],
        'z' => [0b00000, 0b00000, 0b11111, 0b00010, 0b00100, 0b01000, 0b11111],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100],
        '%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b00111],
        ' ' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
        ':' => [0b00000, 0b00100, 0b00000, 0b00000, 0b00000, 0b00100, 0b00000],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        _ => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    }
}
