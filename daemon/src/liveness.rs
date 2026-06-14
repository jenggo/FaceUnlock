use crate::detect::BBox;

#[derive(Debug)]
pub struct LivenessResult {
    pub is_live: bool,
    pub ir_mean: f32,
    pub ir_std: f32,
}

pub fn check_liveness(
    frame_data: &[u8],
    frame_width: u32,
    bbox: &BBox,
    liveness_threshold: f32,
) -> LivenessResult {
    let face_pixels = extract_face_pixels(frame_data, frame_width, bbox);

    if face_pixels.is_empty() {
        return LivenessResult {
            is_live: false,
            ir_mean: 0.0,
            ir_std: 0.0,
        };
    }

    let ir_mean = compute_mean(&face_pixels);
    let ir_std = compute_std(&face_pixels, ir_mean);

    // Normalize std to [0, 1] range: raw std for 8-bit pixels ranges [0, 127.5],
    // dividing by 255 maps it to [0, ~0.5] which makes the threshold (default 0.15)
    // intuitive as a fraction of the intensity range.
    let normalized_std = ir_std / 255.0;

    let is_live = ir_mean >= 10.0 && ir_mean <= 240.0 && normalized_std >= liveness_threshold;

    LivenessResult {
        is_live,
        ir_mean,
        ir_std,
    }
}

fn extract_face_pixels(frame_data: &[u8], frame_width: u32, bbox: &BBox) -> Vec<u8> {
    let x_start = (bbox.x.round() as u32).min(frame_width);
    let y_start = bbox.y.round() as u32;
    let x_end = ((bbox.x + bbox.w).round() as u32).min(frame_width);
    let y_end = (bbox.y + bbox.h).round() as u32;

    let mut pixels = Vec::new();

    for y in y_start..y_end {
        for x in x_start..x_end {
            let idx = (y * frame_width + x) as usize;
            if idx < frame_data.len() {
                pixels.push(frame_data[idx]);
            }
        }
    }

    pixels
}

fn compute_mean(data: &[u8]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let sum: u64 = data.iter().map(|&p| p as u64).sum();
    sum as f32 / data.len() as f32
}

fn compute_std(data: &[u8], mean: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let variance: f32 = data
        .iter()
        .map(|&p| {
            let diff = p as f32 - mean;
            diff * diff
        })
        .sum::<f32>()
        / data.len() as f32;
    variance.sqrt()
}
