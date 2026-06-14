#![allow(dead_code)]

use anyhow::Result;
use ndarray::Array4;
use crate::camera::RawFrame;

const CLAHE_CLIP_LIMIT: f32 = 2.0;
const CLAHE_TILE_SIZE: usize = 8;
const TARGET_SIZE: usize = 112;

pub fn preprocess(frame: &RawFrame) -> Result<Array4<f32>> {
    let grey = &frame.data;
    let w = frame.width as usize;
    let h = frame.height as usize;

    let enhanced = clahe(grey, w, h, CLAHE_TILE_SIZE, CLAHE_CLIP_LIMIT);

    let rgb = grey_to_rgb(&enhanced, w, h);

    let resized = resize(&rgb, w, h, TARGET_SIZE, TARGET_SIZE);

    let normalized = normalize(&resized);

    let chw = hwc_to_chw(&normalized, TARGET_SIZE as usize, TARGET_SIZE as usize);

    Ok(chw)
}

fn clahe(data: &[u8], width: usize, height: usize, tile_size: usize, clip_limit: f32) -> Vec<u8> {
    let mut result = vec![0u8; width * height];

    let tiles_x = (width + tile_size - 1) / tile_size;
    let tiles_y = (height + tile_size - 1) / tile_size;
    let num_bins = 256;
    let max_bin = (num_bins as f32 - 1.0) / (clip_limit * tile_size as f32);

    // Phase 1: Compute LUT for each tile
    let mut tile_luts = vec![[0u8; 256]; tiles_x * tiles_y];

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x_start = tx * tile_size;
            let y_start = ty * tile_size;
            let x_end = ((tx + 1) * tile_size).min(width);
            let y_end = ((ty + 1) * tile_size).min(height);

            let mut histogram = [0u32; 256];
            for y in y_start..y_end {
                for x in x_start..x_end {
                    histogram[data[y * width + x] as usize] += 1;
                }
            }

            let mut clipped = 0u32;
            for bin in 0..num_bins {
                if histogram[bin] > max_bin as u32 {
                    clipped += histogram[bin] - max_bin as u32;
                    histogram[bin] = max_bin as u32;
                }
            }

            let redist = clipped / num_bins as u32;
            for bin in 0..num_bins {
                histogram[bin] += redist;
            }

            let total_pixels = ((x_end - x_start) * (y_end - y_start)) as f32;
            let mut lut = [0u8; 256];
            let mut cumsum = 0.0f32;
            for bin in 0..num_bins {
                cumsum += histogram[bin] as f32;
                lut[bin] = (cumsum / total_pixels * 255.0).min(255.0) as u8;
            }

            tile_luts[ty * tiles_x + tx] = lut;
        }
    }

    // Phase 2: Apply LUTs with bilinear interpolation between tile centers.
    // Each pixel is mapped by blending the 4 surrounding tile LUTs weighted
    // by the pixel's position relative to the tile centers.
    for y in 0..height {
        for x in 0..width {
            // Tile coordinates of the pixel (float)
            let tx_f = (x as f32 + 0.5) / tile_size as f32 - 0.5;
            let ty_f = (y as f32 + 0.5) / tile_size as f32 - 0.5;

            // Tile indices of the 4 surrounding tiles (clamp to valid range)
            let tx0 = tx_f.floor().max(0.0) as usize;
            let ty0 = ty_f.floor().max(0.0) as usize;
            let tx1 = (tx0 + 1).min(tiles_x - 1);
            let ty1 = (ty0 + 1).min(tiles_y - 1);

            // Interpolation weights (fractional position within the tile grid)
            let fx = (tx_f - tx_f.floor()).clamp(0.0, 1.0);
            let fy = (ty_f - ty_f.floor()).clamp(0.0, 1.0);

            let val = data[y * width + x] as usize;

            let lut_tl = &tile_luts[ty0 * tiles_x + tx0];
            let lut_tr = &tile_luts[ty0 * tiles_x + tx1];
            let lut_bl = &tile_luts[ty1 * tiles_x + tx0];
            let lut_br = &tile_luts[ty1 * tiles_x + tx1];

            let blended = (1.0 - fx) * (1.0 - fy) * lut_tl[val] as f32
                + fx * (1.0 - fy) * lut_tr[val] as f32
                + (1.0 - fx) * fy * lut_bl[val] as f32
                + fx * fy * lut_br[val] as f32;

            result[y * width + x] = blended.min(255.0).max(0.0) as u8;
        }
    }

    result
}

fn grey_to_rgb(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for i in 0..width * height {
        let v = data[i];
        rgb[i * 3] = v;
        rgb[i * 3 + 1] = v;
        rgb[i * 3 + 2] = v;
    }
    rgb
}

fn resize(data: &[u8], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<u8> {
    let img = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(src_w as u32, src_h as u32, data.to_vec())
        .expect("Failed to create image buffer");
    let resized = image::imageops::resize(
        &img,
        dst_w as u32,
        dst_h as u32,
        image::imageops::FilterType::Lanczos3,
    );
    resized.into_raw()
}

fn normalize(data: &[u8]) -> Vec<f32> {
    data.iter()
        .map(|&p| (p as f32 / 255.0 - 0.5) / 0.5)
        .collect()
}

fn hwc_to_chw(data: &[f32], height: usize, width: usize) -> Array4<f32> {
    let mut arr = Array4::<f32>::zeros((1, 3, height, width));
    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                arr[[0, c, y, x]] = data[(y * width + x) * 3 + c];
            }
        }
    }
    arr
}
