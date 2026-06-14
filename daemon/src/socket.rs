use anyhow::{Context, Result};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use crate::camera::Camera;
use crate::config::Config;
use crate::detect::Detector;
use crate::enroll::EnrollmentStore;
use crate::liveness;
use crate::recognize::Recognizer;

const SOCKET_PATH: &str = "/run/faceunlockd/auth.sock";

#[derive(Debug, Deserialize)]
pub struct AuthRequest {
    pub action: String,
    pub user: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AuthResponse {
    pub result: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ir_mean: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ir_std: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub liveness: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub users: Option<Vec<UserInfo>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<Timing>,
}

#[derive(Debug, Serialize)]
pub struct Timing {
    pub detect_ms: u64,
    pub recognize_ms: u64,
    pub total_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct UserInfo {
    pub username: String,
    pub embedding_count: usize,
}

pub struct SocketServer {
    config: Config,
}

impl SocketServer {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    pub async fn bind(&self) -> Result<UnixListener> {
        let socket_path = Path::new(SOCKET_PATH);

        if socket_path.exists() {
            std::fs::remove_file(socket_path)?;
        }

        if let Some(parent) = socket_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let listener = UnixListener::bind(socket_path)
            .with_context(|| format!("Failed to bind socket: {}", SOCKET_PATH))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o660);
            std::fs::set_permissions(socket_path, perms)?;

            let _ = std::process::Command::new("chown")
                .args(["root:faceunlock", SOCKET_PATH])
                .output();
        }

        tracing::info!("Socket bound at {}", SOCKET_PATH);
        Ok(listener)
    }

    pub async fn run(&self, listener: UnixListener) -> Result<()> {
        tracing::info!("Listening for connections");

        loop {
            let (stream, _addr) = listener.accept().await?;
            let config = self.config.clone();
            tokio::spawn(async move {
                if let Err(e) = handle_connection(stream, &config).await {
                    tracing::error!("Connection error: {}", e);
                }
            });
        }
    }
}

async fn handle_connection(stream: UnixStream, config: &Config) -> Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    let Some(line) = lines.next_line().await? else {
        let resp = serde_json::to_string(&AuthResponse {
            result: "fail".to_string(),
            reason: Some("Empty request".to_string()),
            score: None, ir_mean: None, ir_std: None, liveness: None, users: None, timing: None,
        })?;
        writer.write_all(resp.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        return Ok(());
    };

    let req: AuthRequest = match serde_json::from_str(&line) {
        Ok(r) => r,
        Err(e) => {
            let resp = serde_json::to_string(&AuthResponse {
                result: "fail".to_string(),
                reason: Some(format!("Invalid request: {}", e)),
                score: None, ir_mean: None, ir_std: None, liveness: None, users: None, timing: None,
            })?;
            writer.write_all(resp.as_bytes()).await?;
            writer.write_all(b"\n").await?;
            return Ok(());
        }
    };

    if req.action == "view" {
        return handle_view(&mut writer, config, req.user.as_deref()).await;
    }

    let response = handle_request(req, config).await;
    let json = serde_json::to_string(&response)?;
    writer.write_all(json.as_bytes()).await?;
    writer.write_all(b"\n").await?;

    Ok(())
}

async fn handle_request(req: AuthRequest, config: &Config) -> AuthResponse {
    match req.action.as_str() {
        "authenticate" => handle_authenticate(req, config).await,
        "enroll" => handle_enroll(req, config).await,
        "enroll_clear" => handle_enroll_clear(req, config).await,
        "status" => handle_status(config).await,
        _ => AuthResponse {
            result: "fail".to_string(),
            reason: Some(format!("Unknown action: {}", req.action)),
            score: None,
            ir_mean: None,
            ir_std: None,
            liveness: None,
            users: None,
                timing: None,
        },
    }
}

async fn handle_authenticate(req: AuthRequest, config: &Config) -> AuthResponse {
    let username = match req.user {
        Some(u) => u,
        None => {
            return AuthResponse {
                result: "fail".to_string(),
                reason: Some("Missing user field".to_string()),
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
            };
        }
    };

    let store = EnrollmentStore::new(&config.enrollment.store_path);
    let embeddings = match store.load_embeddings(&username) {
        Ok(e) => e,
        Err(_) => {
            return AuthResponse {
                result: "fail".to_string(),
                reason: Some("User not enrolled".to_string()),
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
            };
        }
    };

    if embeddings.is_empty() {
        return AuthResponse {
            result: "fail".to_string(),
            reason: Some("User not enrolled".to_string()),
            score: None,
            ir_mean: None,
            ir_std: None,
            liveness: None,
            users: None,
                timing: None,
        };
    }

    let mut camera = match Camera::new(
        &config.camera.device,
        config.camera.width,
        config.camera.height,
        config.camera.pixel_format.as_deref(),
    ) {
        Ok(c) => c,
        Err(e) => {
            return AuthResponse {
                result: "fail".to_string(),
                reason: Some(format!("Failed to open camera: {}", e)),
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
            };
        }
    };

    camera.warmup(config.auth.camera_warmup);

    let mut detector = match Detector::new(&config.models.detector, config.auth.detection_threshold) {
        Ok(d) => d,
        Err(e) => {
            return AuthResponse {
                result: "fail".to_string(),
                reason: Some(format!("Failed to load detector: {}", e)),
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
            };
        }
    };

    let mut recognizer = match Recognizer::new(&config.models.recognizer) {
        Ok(r) => r,
        Err(e) => {
            return AuthResponse {
                result: "fail".to_string(),
                reason: Some(format!("Failed to load recognizer: {}", e)),
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
            };
        }
    };

    let mut best_score = 0.0f32;
    let mut last_liveness = None;
    let mut last_ir_mean = None;
    let mut last_ir_std = None;
    let mut total_detect_ms = 0u64;
    let mut total_recognize_ms = 0u64;
    let total_start = std::time::Instant::now();

    for _ in 0..config.auth.max_frames {
        let frame = match camera.capture_frame() {
            Ok(f) => f,
            Err(_) => continue,
        };

        let detect_start = std::time::Instant::now();
        let face = match detector.detect_with_letterbox(
            &frame.data,
            frame.width,
            frame.height,
        ) {
            Ok(Some(f)) => f,
            _ => continue,
        };

        let liveness_result = liveness::check_liveness(
            &frame.data,
            frame.width,
            &face.bbox,
            config.auth.liveness_threshold,
        );
        total_detect_ms += detect_start.elapsed().as_millis() as u64;

        tracing::debug!(
            "Liveness: mean={:.1} std={:.1} live={}",
            liveness_result.ir_mean, liveness_result.ir_std, liveness_result.is_live
        );

        last_liveness = Some(liveness_result.is_live);
        last_ir_mean = Some(liveness_result.ir_mean);
        last_ir_std = Some(liveness_result.ir_std);

        if !liveness_result.is_live {
            continue;
        }

        let recog_start = std::time::Instant::now();
        let aligned = match recognizer.align_face(
            &frame.data,
            frame.width,
            frame.height,
            &face.landmarks,
        ) {
            Ok(a) => a,
            Err(_) => continue,
        };

        let embedding = match recognizer.embed(&aligned) {
            Ok(e) => e,
            Err(_) => continue,
        };
        total_recognize_ms += recog_start.elapsed().as_millis() as u64;

        for enrolled in &embeddings {
            let score = crate::recognize::cosine_similarity(&embedding, enrolled);
            if score > best_score {
                best_score = score;
            }
        }
    }

    let total_ms = total_start.elapsed().as_millis() as u64;

    let result = if best_score >= config.auth.similarity_threshold {
        "ok"
    } else {
        "fail"
    };

    AuthResponse {
        result: result.to_string(),
        reason: if result == "fail" {
            Some(format!(
                "Score {:.4} below threshold {:.4}",
                best_score, config.auth.similarity_threshold
            ))
        } else {
            None
        },
        score: Some(best_score),
        ir_mean: last_ir_mean,
        ir_std: last_ir_std,
        liveness: last_liveness,
        users: None,
        timing: Some(Timing {
            detect_ms: total_detect_ms,
            recognize_ms: total_recognize_ms,
            total_ms,
        }),
    }
}

async fn handle_enroll(req: AuthRequest, config: &Config) -> AuthResponse {
    let username = match req.user {
        Some(u) => u,
        None => {
            return AuthResponse {
                result: "fail".to_string(),
                reason: Some("Missing user field".to_string()),
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
            };
        }
    };

    let mut camera = match Camera::new(
        &config.camera.device,
        config.camera.width,
        config.camera.height,
        config.camera.pixel_format.as_deref(),
    ) {
        Ok(c) => c,
        Err(e) => {
            return AuthResponse {
                result: "fail".to_string(),
                reason: Some(format!("Failed to open camera: {}", e)),
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
            };
        }
    };

    camera.warmup(config.auth.camera_warmup);

    let mut detector = match Detector::new(&config.models.detector, config.auth.detection_threshold) {
        Ok(d) => d,
        Err(e) => {
            return AuthResponse {
                result: "fail".to_string(),
                reason: Some(format!("Failed to load detector: {}", e)),
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
            };
        }
    };

    let mut recognizer = match Recognizer::new(&config.models.recognizer) {
        Ok(r) => r,
        Err(e) => {
            return AuthResponse {
                result: "fail".to_string(),
                reason: Some(format!("Failed to load recognizer: {}", e)),
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
            };
        }
    };

    let store = EnrollmentStore::new(&config.enrollment.store_path);
    let mut enrolled_count = 0;

    for attempt in 0..3 {
        let mut captured = false;
        for _ in 0..config.auth.max_frames {
            let frame = match camera.capture_frame() {
                Ok(f) => f,
                Err(e) => {
                    tracing::warn!("Camera capture failed: {}", e);
                    continue;
                }
            };

            let pixel_stats = frame.data.iter().fold((u32::MAX, u32::MIN, 0u64), |(min, max, sum), &p| {
                (min.min(p as u32), max.max(p as u32), sum + p as u64)
            });
            let mean = pixel_stats.2 as f64 / frame.data.len() as f64;
            tracing::info!(
                "Frame captured: {}x{}, pixels: min={} max={} mean={:.1}",
                frame.width, frame.height, pixel_stats.0, pixel_stats.1, mean
            );

            let face = match detector.detect_with_letterbox(
                &frame.data,
                frame.width,
                frame.height,
            ) {
                Ok(Some(f)) => f,
                Ok(None) => {
                    tracing::info!("No face detected in frame");
                    continue;
                }
                Err(e) => {
                    tracing::warn!("Face detection error: {}", e);
                    continue;
                }
            };

            let aligned = match recognizer.align_face(
                &frame.data,
                frame.width,
                frame.height,
                &face.landmarks,
            ) {
                Ok(a) => a,
                Err(e) => {
                    tracing::warn!("Face alignment failed: {}", e);
                    continue;
                }
            };

            let embedding = match recognizer.embed(&aligned) {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("Embedding failed: {} (aligned shape: {:?})", e, aligned.shape());
                    continue;
                }
            };

            if let Err(e) = store.save_embedding(&username, &embedding) {
                tracing::error!("Failed to save embedding: {}", e);
                continue;
            }

            enrolled_count += 1;
            captured = true;
            break;
        }

        if !captured {
            tracing::warn!("Failed to capture face for enrollment attempt {}", attempt + 1);
        }
    }

    if enrolled_count > 0 {
        AuthResponse {
            result: "ok".to_string(),
            reason: Some(format!("Enrolled {} embeddings", enrolled_count)),
            score: None,
            ir_mean: None,
            ir_std: None,
            liveness: None,
            users: None,
                timing: None,
        }
    } else {
        AuthResponse {
            result: "fail".to_string(),
            reason: Some("Failed to capture any face frames".to_string()),
            score: None,
            ir_mean: None,
            ir_std: None,
            liveness: None,
            users: None,
                timing: None,
        }
    }
}

async fn handle_enroll_clear(req: AuthRequest, config: &Config) -> AuthResponse {
    let username = match req.user {
        Some(u) => u,
        None => {
            return AuthResponse {
                result: "fail".to_string(),
                reason: Some("Missing user field".to_string()),
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
            };
        }
    };

    let store = EnrollmentStore::new(&config.enrollment.store_path);
    match store.clear_embeddings(&username) {
        Ok(()) => AuthResponse {
            result: "ok".to_string(),
            reason: None,
            score: None,
            ir_mean: None,
            ir_std: None,
            liveness: None,
            users: None,
                timing: None,
        },
        Err(e) => AuthResponse {
            result: "fail".to_string(),
            reason: Some(format!("Failed to clear embeddings: {}", e)),
            score: None,
            ir_mean: None,
            ir_std: None,
            liveness: None,
            users: None,
                timing: None,
        },
    }
}

async fn handle_status(config: &Config) -> AuthResponse {
    let store = EnrollmentStore::new(&config.enrollment.store_path);
    let users = store.list_users().unwrap_or_default();

    let user_infos: Vec<UserInfo> = users
        .into_iter()
        .map(|(username, count)| UserInfo {
            username,
            embedding_count: count,
        })
        .collect();

    AuthResponse {
        result: "ok".to_string(),
        reason: None,
        score: None,
        ir_mean: None,
        ir_std: None,
        liveness: None,
        users: Some(user_infos),
        timing: None,
    }
}

async fn handle_view(writer: &mut tokio::net::unix::OwnedWriteHalf, config: &Config, user: Option<&str>) -> Result<()> {
    tracing::info!("View stream requested");

    let user = match user {
        Some(u) => u.to_string(),
        None => {
            let err = serde_json::json!({"type": "error", "message": "Missing user field"});
            let mut msg = serde_json::to_string(&err)?;
            msg.push('\n');
            writer.write_all(msg.as_bytes()).await?;
            return Ok(());
        }
    };

    let store = EnrollmentStore::new(&config.enrollment.store_path);

    let mut camera = match Camera::new(
        &config.camera.device,
        config.camera.width,
        config.camera.height,
        config.camera.pixel_format.as_deref(),
    ) {
        Ok(c) => c,
        Err(e) => {
            let err = serde_json::json!({"type": "error", "message": format!("Camera error: {}", e)});
            let mut msg = serde_json::to_string(&err)?;
            msg.push('\n');
            writer.write_all(msg.as_bytes()).await?;
            return Ok(());
        }
    };

    camera.warmup(config.auth.camera_warmup);

    let mut detector = match Detector::new(&config.models.detector, config.auth.detection_threshold) {
        Ok(d) => d,
        Err(e) => {
            let err = serde_json::json!({"type": "error", "message": format!("Detector error: {}", e)});
            let mut msg = serde_json::to_string(&err)?;
            msg.push('\n');
            writer.write_all(msg.as_bytes()).await?;
            return Ok(());
        }
    };

    let mut recognizer = match Recognizer::new(&config.models.recognizer) {
        Ok(r) => r,
        Err(e) => {
            let err = serde_json::json!({"type": "error", "message": format!("Recognizer error: {}", e)});
            let mut msg = serde_json::to_string(&err)?;
            msg.push('\n');
            writer.write_all(msg.as_bytes()).await?;
            return Ok(());
        }
    };

    let engine = base64::engine::general_purpose::STANDARD;
    let threshold = config.auth.similarity_threshold;

    loop {
        let frame = match camera.capture_frame() {
            Ok(f) => f,
            Err(_) => {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                continue;
            }
        };

        let detect_start = Instant::now();
        let face = detector.detect_with_letterbox(&frame.data, frame.width, frame.height).ok().flatten();
        let detect_ms = detect_start.elapsed().as_millis() as u64;

        let image_b64 = engine.encode(&frame.data);

        let mut msg = serde_json::json!({
            "type": "frame",
            "image": image_b64,
            "width": frame.width,
            "height": frame.height,
            "detect_ms": detect_ms,
        });

        if let Some(f) = face {
            let mut best_score: f32 = 0.0;
            let mut authenticated = false;

            let embeddings = store.load_embeddings(&user).unwrap_or_default();
            if !embeddings.is_empty() {
                if let Ok(aligned) = recognizer.align_face(
                    &frame.data, frame.width, frame.height, &f.landmarks,
                ) {
                    if let Ok(embedding) = recognizer.embed(&aligned) {
                        for enrolled in &embeddings {
                            let score = crate::recognize::cosine_similarity(&embedding, enrolled);
                            if score > best_score {
                                best_score = score;
                            }
                        }
                        authenticated = best_score >= threshold;
                    }
                }
            }

            msg["face"] = serde_json::json!({
                "bbox": {"x": f.bbox.x, "y": f.bbox.y, "w": f.bbox.w, "h": f.bbox.h},
                "confidence": f.confidence,
                "landmarks": f.landmarks.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>(),
                "authenticated": authenticated,
                "score": best_score,
            });
        }

        let mut json = serde_json::to_string(&msg)?;
        json.push('\n');

        if writer.write_all(json.as_bytes()).await.is_err() {
            break;
        }
    }

    tracing::info!("View stream ended");
    Ok(())
}
