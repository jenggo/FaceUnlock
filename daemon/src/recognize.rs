use anyhow::{Context, Result};
use ndarray::Array4;
use ort::session::Session;

use crate::detect::Point2D;

const FACE_SIZE: usize = 112;

const CANONICAL_LANDMARKS: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

fn scaled_canonical_landmarks(face_size: usize) -> [[f32; 2]; 5] {
    if face_size == FACE_SIZE {
        return CANONICAL_LANDMARKS;
    }
    let scale = face_size as f32 / FACE_SIZE as f32;
    [
        [CANONICAL_LANDMARKS[0][0] * scale, CANONICAL_LANDMARKS[0][1] * scale],
        [CANONICAL_LANDMARKS[1][0] * scale, CANONICAL_LANDMARKS[1][1] * scale],
        [CANONICAL_LANDMARKS[2][0] * scale, CANONICAL_LANDMARKS[2][1] * scale],
        [CANONICAL_LANDMARKS[3][0] * scale, CANONICAL_LANDMARKS[3][1] * scale],
        [CANONICAL_LANDMARKS[4][0] * scale, CANONICAL_LANDMARKS[4][1] * scale],
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    ArcFace,
    AdaFace,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::ArcFace => write!(f, "ArcFace"),
            ModelType::AdaFace => write!(f, "AdaFace"),
        }
    }
}

pub struct Recognizer {
    session: Session,
    model_type: ModelType,
    face_size: usize,
}

impl Recognizer {
    pub fn new(model_path: &str) -> Result<Self> {
        let mut session = Session::builder()?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load recognizer model: {}", model_path))?;

        let model_type = detect_model_type(&session)?;

        let embedding_dim = session.outputs()[0]
            .dtype()
            .tensor_shape()
            .and_then(|shape| shape.last().copied())
            .map(|d| d as usize)
            .unwrap_or(512);

        let face_size = session.inputs()[0]
            .dtype()
            .tensor_shape()
            .and_then(|shape| {
                if shape.len() >= 4 {
                    Some(shape[2] as usize)
                } else {
                    None
                }
            })
            .unwrap_or(FACE_SIZE);

        tracing::info!(
            "Detected recognizer model: {} (embedding_dim={}, face_size={}, input='{}', output='{}')",
            model_type,
            embedding_dim,
            face_size,
            session.inputs()[0].name(),
            session.outputs()[0].name(),
        );

        let dummy = ndarray::Array4::<f32>::zeros((1, 3, face_size, face_size));
        let test_tensor = ort::value::Tensor::from_array((
            dummy.shape().to_vec(),
            dummy.as_slice().unwrap().to_vec(),
        ));
        match test_tensor {
            Ok(t) => match session.run(ort::inputs![t]) {
                Ok(out) => {
                    for (name, val) in out.iter() {
                        if let Ok((shape, _)) = val.try_extract_tensor::<f32>() {
                            tracing::info!("Recognizer output '{}': shape {:?}", name, shape);
                        }
                    }
                    tracing::info!("Recognizer test OK with [1,3,{},{}]", face_size, face_size);
                }
                Err(e) => tracing::warn!("Recognizer test run failed: {}", e),
            },
            Err(e) => tracing::warn!("Recognizer test tensor failed: {}", e),
        }

        Ok(Self {
            session,
            model_type,
            face_size,
        })
    }

    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    pub fn embed(&mut self, aligned_face: &Array4<f32>) -> Result<Vec<f32>> {
        let input_tensor = ort::value::Tensor::from_array((
            aligned_face.shape().to_vec(),
            aligned_face.as_slice().unwrap().to_vec(),
        ))?;

        let outputs = self.session.run(ort::inputs![input_tensor])?;

        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        Ok(data.to_vec())
    }

    pub fn align_face(
        &self,
        grey_data: &[u8],
        width: u32,
        height: u32,
        landmarks: &[Point2D; 5],
    ) -> Result<Array4<f32>> {
        let canonical = scaled_canonical_landmarks(self.face_size);
        let transform = estimate_affine(landmarks, &canonical)?;

        let img =
            image::ImageBuffer::<image::Luma<u8>, _>::from_raw(width, height, grey_data.to_vec())
                .context("Failed to create image buffer for alignment")?;

        let aligned = apply_affine(&img, &transform, self.face_size, self.face_size)?;

        let mut arr = Array4::<f32>::zeros((1, 1, self.face_size, self.face_size));
        for y in 0..self.face_size {
            for x in 0..self.face_size {
                let pixel = aligned[y * self.face_size + x] as f32;
                arr[[0, 0, y, x]] = match self.model_type {
                    ModelType::ArcFace => pixel / 127.5 - 1.0,
                    ModelType::AdaFace => (pixel / 255.0 - 0.5) / 0.5,
                };
            }
        }

        let rgb_arr = replicate_channels(&arr, self.face_size);

        Ok(rgb_arr)
    }
}

fn detect_model_type(session: &Session) -> Result<ModelType> {
    let input_name = session.inputs()[0].name().to_lowercase();
    let output_name = session.outputs()[0].name().to_lowercase();

    if input_name == "input" || output_name == "output" {
        return Ok(ModelType::AdaFace);
    }

    if input_name == "input.1"
        || input_name == "data"
        || input_name.starts_with("input.")
        || output_name.starts_with("fc")
        || output_name
            .chars()
            .next()
            .map_or(false, |c| c.is_ascii_digit())
    {
        return Ok(ModelType::ArcFace);
    }

    if let Ok(meta) = session.metadata() {
        if let Some(name) = meta.name() {
            let name_lower = name.to_lowercase();
            if name_lower.contains("adaface") {
                return Ok(ModelType::AdaFace);
            }
            if name_lower.contains("arcface") || name_lower.contains("insightface") {
                return Ok(ModelType::ArcFace);
            }
        }
        if let Some(producer) = meta.producer() {
            let producer_lower = producer.to_lowercase();
            if producer_lower.contains("adaface") {
                return Ok(ModelType::AdaFace);
            }
        }
        for key in meta.custom_keys().unwrap_or_default() {
            if let Some(val) = meta.custom(&key) {
                let val_lower = val.to_lowercase();
                if val_lower.contains("adaface") {
                    return Ok(ModelType::AdaFace);
                }
                if val_lower.contains("arcface") {
                    return Ok(ModelType::ArcFace);
                }
            }
        }
    }

    tracing::warn!(
        "Could not auto-detect recognizer model type from input='{}' output='{}', defaulting to ArcFace",
        input_name,
        output_name
    );
    Ok(ModelType::ArcFace)
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

fn replicate_channels(gray: &Array4<f32>, face_size: usize) -> Array4<f32> {
    let mut rgb = Array4::<f32>::zeros((1, 3, face_size, face_size));
    for c in 0..3 {
        for y in 0..face_size {
            for x in 0..face_size {
                rgb[[0, c, y, x]] = gray[[0, 0, y, x]];
            }
        }
    }
    rgb
}

fn estimate_affine(src: &[Point2D; 5], dst: &[[f32; 2]; 5]) -> Result<AffineTransform> {
    let n = src.len() as f32;

    let src_mean_x: f32 = src.iter().map(|p| p.x).sum::<f32>() / n;
    let src_mean_y: f32 = src.iter().map(|p| p.y).sum::<f32>() / n;
    let dst_mean_x: f32 = dst.iter().map(|p| p[0]).sum::<f32>() / n;
    let dst_mean_y: f32 = dst.iter().map(|p| p[1]).sum::<f32>() / n;

    let src_norm: Vec<(f32, f32)> = src
        .iter()
        .map(|p| (p.x - src_mean_x, p.y - src_mean_y))
        .collect();
    let dst_norm: Vec<(f32, f32)> = dst
        .iter()
        .map(|p| (p[0] - dst_mean_x, p[1] - dst_mean_y))
        .collect();

    let mut src_var = [[0.0f32; 2]; 2];
    let mut cross = [[0.0f32; 2]; 2];

    for i in 0..src.len() {
        src_var[0][0] += src_norm[i].0 * src_norm[i].0;
        src_var[0][1] += src_norm[i].0 * src_norm[i].1;
        src_var[1][0] += src_norm[i].1 * src_norm[i].0;
        src_var[1][1] += src_norm[i].1 * src_norm[i].1;

        cross[0][0] += dst_norm[i].0 * src_norm[i].0;
        cross[0][1] += dst_norm[i].0 * src_norm[i].1;
        cross[1][0] += dst_norm[i].1 * src_norm[i].0;
        cross[1][1] += dst_norm[i].1 * src_norm[i].1;
    }

    let det = src_var[0][0] * src_var[1][1] - src_var[0][1] * src_var[1][0];
    if det.abs() < 1e-10 {
        anyhow::bail!("Singular matrix in affine estimation");
    }

    let inv_var = [
        [src_var[1][1] / det, -src_var[0][1] / det],
        [-src_var[1][0] / det, src_var[0][0] / det],
    ];

    let a = cross[0][0] * inv_var[0][0] + cross[0][1] * inv_var[1][0];
    let b = cross[0][0] * inv_var[0][1] + cross[0][1] * inv_var[1][1];
    let c = cross[1][0] * inv_var[0][0] + cross[1][1] * inv_var[1][0];
    let d = cross[1][0] * inv_var[0][1] + cross[1][1] * inv_var[1][1];

    let tx = dst_mean_x - a * src_mean_x - b * src_mean_y;
    let ty = dst_mean_y - c * src_mean_x - d * src_mean_y;

    Ok(AffineTransform { a, b, c, d, tx, ty })
}

#[derive(Debug)]
struct AffineTransform {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    tx: f32,
    ty: f32,
}

fn invert_affine(t: &AffineTransform) -> AffineTransform {
    let det = t.a * t.d - t.b * t.c;
    AffineTransform {
        a: t.d / det,
        b: -t.b / det,
        c: -t.c / det,
        d: t.a / det,
        tx: (t.b * t.ty - t.d * t.tx) / det,
        ty: (t.c * t.tx - t.a * t.ty) / det,
    }
}

fn apply_affine(
    img: &image::ImageBuffer<image::Luma<u8>, Vec<u8>>,
    transform: &AffineTransform,
    out_w: usize,
    out_h: usize,
) -> Result<Vec<u8>> {
    let inv = invert_affine(transform);
    let mut output = vec![0u8; out_w * out_h];
    let src_w = img.width() as f32;
    let src_h = img.height() as f32;

    for y in 0..out_h {
        for x in 0..out_w {
            let src_x = inv.a * x as f32 + inv.b * y as f32 + inv.tx;
            let src_y = inv.c * x as f32 + inv.d * y as f32 + inv.ty;

            if src_x >= 0.0 && src_x < src_w && src_y >= 0.0 && src_y < src_h {
                let x0 = src_x.floor() as u32;
                let y0 = src_y.floor() as u32;
                let x1 = (x0 + 1).min(img.width() - 1);
                let y1 = (y0 + 1).min(img.height() - 1);

                let fx = src_x - x0 as f32;
                let fy = src_y - y0 as f32;

                let v00 = img.get_pixel(x0, y0)[0] as f32;
                let v10 = img.get_pixel(x1, y0)[0] as f32;
                let v01 = img.get_pixel(x0, y1)[0] as f32;
                let v11 = img.get_pixel(x1, y1)[0] as f32;

                let val = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                output[y * out_w + x] = val.min(255.0).max(0.0) as u8;
            }
        }
    }

    Ok(output)
}
