use anyhow::{Context, Result};
use ndarray::Array4;
use ort::session::Session;

#[derive(Debug, Clone)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Debug, Clone)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone)]
pub struct DetectedFace {
    pub bbox: BBox,
    pub landmarks: [Point2D; 5],
    #[allow(dead_code)]
    pub confidence: f32,
}

pub struct Detector {
    session: Session,
    confidence_threshold: f32,
}

impl Detector {
    pub fn new(model_path: &str, confidence_threshold: f32) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load detector model: {}", model_path))?;

        Ok(Self {
            session,
            confidence_threshold,
        })
    }

    pub fn detect_with_letterbox(
        &mut self,
        frame_data: &[u8],
        frame_w: u32,
        frame_h: u32,
    ) -> Result<Option<DetectedFace>> {
        let model_size = 640u32;
        let (resized, pad_x, pad_y, scale) =
            letterbox_resize(frame_data, frame_w, frame_h, model_size);

        let grey: Vec<f32> = resized
            .iter()
            .map(|&p| (p as f32 - 127.5) / 128.0)
            .collect();

        let mut input = Array4::<f32>::zeros((1, 3, model_size as usize, model_size as usize));
        for y in 0..model_size as usize {
            for x in 0..model_size as usize {
                let val = grey[y * model_size as usize + x];
                for c in 0..3 {
                    input[[0, c, y, x]] = val;
                }
            }
        }

        let input_tensor = ort::value::Tensor::from_array((
            input.shape().to_vec(),
            input.as_slice().unwrap().to_vec(),
        ))?;

        let outputs = self.session.run(ort::inputs![input_tensor])?;

        let (faces, landmarks, confidences) =
            parse_scrfd_outputs(&outputs, self.confidence_threshold)?;

        let mut best: Option<DetectedFace> = None;
        let mut best_area = 0.0f32;

        for (i, &conf) in confidences.iter().enumerate() {
            if conf < self.confidence_threshold {
                continue;
            }

            let area = faces[i].w * faces[i].h;
            if area > best_area {
                best_area = area;
                best = Some(DetectedFace {
                    bbox: faces[i].clone(),
                    landmarks: landmarks[i].clone(),
                    confidence: conf,
                });
            }
        }

        if let Some(ref face) = best {
            tracing::debug!(
                "Face detected: bbox=({:.1},{:.1},{:.1},{:.1}) conf={:.3}",
                face.bbox.x,
                face.bbox.y,
                face.bbox.w,
                face.bbox.h,
                face.confidence
            );
        }

        if let Some(mut face) = best {
            face.bbox.x = (face.bbox.x - pad_x) / scale;
            face.bbox.y = (face.bbox.y - pad_y) / scale;
            face.bbox.w /= scale;
            face.bbox.h /= scale;

            for lm in &mut face.landmarks {
                lm.x = (lm.x - pad_x) / scale;
                lm.y = (lm.y - pad_y) / scale;
            }

            Ok(Some(face))
        } else {
            Ok(None)
        }
    }
}

fn letterbox_resize(data: &[u8], src_w: u32, src_h: u32, target: u32) -> (Vec<u8>, f32, f32, f32) {
    let scale = target as f32 / src_w.max(src_h) as f32;
    let new_w = (src_w as f32 * scale) as u32;
    let new_h = (src_h as f32 * scale) as u32;
    let pad_x = ((target - new_w) / 2) as f32;
    let pad_y = ((target - new_h) / 2) as f32;

    let mut output = vec![128u8; (target * target) as usize];

    let img = image::ImageBuffer::<image::Luma<u8>, _>::from_raw(src_w, src_h, data)
        .expect("Failed to create image buffer");
    let resized =
        image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Lanczos3);

    for y in 0..new_h {
        for x in 0..new_w {
            let ox = (x as f32 + pad_x) as u32;
            let oy = (y as f32 + pad_y) as u32;
            if ox < target && oy < target {
                output[(oy * target + ox) as usize] = resized.get_pixel(x, y)[0];
            }
        }
    }

    (output, pad_x, pad_y, scale)
}

fn parse_scrfd_outputs(
    outputs: &ort::session::SessionOutputs,
    confidence_threshold: f32,
) -> Result<(Vec<BBox>, Vec<[Point2D; 5]>, Vec<f32>)> {
    let mut boxes = Vec::new();
    let mut landmarks = Vec::new();
    let mut confidences = Vec::new();

    let mut score_tensors = Vec::new();
    let mut bbox_tensors = Vec::new();
    let mut kps_tensors = Vec::new();

    let output_vec: Vec<_> = outputs.iter().collect();
    for (_name, val) in &output_vec {
        if let Ok((shape, data)) = val.try_extract_tensor::<f32>() {
            let n = shape[0] as usize;
            let d = data.to_vec();
            match shape[1] as usize {
                1 => score_tensors.push((n, d)),
                4 => bbox_tensors.push((n, d)),
                10 => kps_tensors.push((n, d)),
                _ => {}
            }
        }
    }

    let anchor_counts: Vec<usize> = vec![12800, 3200, 800];
    let strides: Vec<f32> = vec![8.0, 16.0, 32.0];

    for (level, &num_anchors) in anchor_counts.iter().enumerate() {
        let score_entry = score_tensors.iter().find(|(n, _)| *n == num_anchors);
        let bbox_entry = bbox_tensors.iter().find(|(n, _)| *n == num_anchors);
        let kps_entry = kps_tensors.iter().find(|(n, _)| *n == num_anchors);

        let (_score_n, score_data) = match score_entry {
            Some(v) => v,
            None => continue,
        };
        let (_, bbox_data) = match bbox_entry {
            Some(v) => v,
            None => continue,
        };
        let (_, kps_data) = match kps_entry {
            Some(v) => v,
            None => continue,
        };

        let stride = strides[level];
        let feat_size = (num_anchors as f32 / 2.0).sqrt() as usize;
        let num_anchors_per_loc = 2;

        for anchor_idx in 0..num_anchors {
            let score = score_data[anchor_idx];
            if score < confidence_threshold {
                continue;
            }

            let spatial_idx = anchor_idx / num_anchors_per_loc;
            let h_idx = spatial_idx / feat_size;
            let w_idx = spatial_idx % feat_size;

            let cx = (w_idx as f32 + 0.5) * stride;
            let cy = (h_idx as f32 + 0.5) * stride;

            let base = anchor_idx * 4;
            let l = bbox_data[base] * stride;
            let t = bbox_data[base + 1] * stride;
            let r = bbox_data[base + 2] * stride;
            let b = bbox_data[base + 3] * stride;

            boxes.push(BBox {
                x: cx - l,
                y: cy - t,
                w: l + r,
                h: t + b,
            });

            let kps_base = anchor_idx * 10;
            landmarks.push([
                Point2D {
                    x: cx + kps_data[kps_base] * stride,
                    y: cy + kps_data[kps_base + 1] * stride,
                },
                Point2D {
                    x: cx + kps_data[kps_base + 2] * stride,
                    y: cy + kps_data[kps_base + 3] * stride,
                },
                Point2D {
                    x: cx + kps_data[kps_base + 4] * stride,
                    y: cy + kps_data[kps_base + 5] * stride,
                },
                Point2D {
                    x: cx + kps_data[kps_base + 6] * stride,
                    y: cy + kps_data[kps_base + 7] * stride,
                },
                Point2D {
                    x: cx + kps_data[kps_base + 8] * stride,
                    y: cy + kps_data[kps_base + 9] * stride,
                },
            ]);

            confidences.push(score);
        }
    }

    Ok((boxes, landmarks, confidences))
}
