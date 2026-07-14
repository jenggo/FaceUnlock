use anyhow::{Context, Result};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use crate::camera::{
    classify_frame, Camera, DeltaTracker, FrameClass, PixelStats, ReadinessConfig, SettleResult,
    BLACK_PATIENCE,
};
use crate::config::Config;
use crate::detect::Detector;
use crate::enroll::EnrollmentStore;
use crate::liveness;
use crate::recognize::{ModelType, Recognizer};

const SOCKET_PATH: &str = "/run/faceunlockd/auth.sock";

/// Build the camera readiness configuration from the daemon config.
fn readiness_config(config: &Config) -> ReadinessConfig {
    ReadinessConfig {
        max_frames: config.auth.camera_warmup,
        delay_ms: config.auth.camera_warmup_delay_ms,
        mean_threshold: config.auth.camera_warmup_mean_threshold,
        delta_threshold: config.auth.camera_warmup_delta_threshold,
        std_threshold: config.auth.camera_warmup_std_threshold,
        stable_frames_required: config.auth.camera_warmup_stable_frames,
    }
}

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
    pub reason_code: Option<String>,
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
    pub arcface_count: usize,
    pub adaface_count: usize,
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
            reason_code: None,
            score: None,
            ir_mean: None,
            ir_std: None,
            liveness: None,
            users: None,
            timing: None,
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
                reason_code: None,
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
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
            reason_code: None,
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
                reason_code: None,
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
                reason_code: None,
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
            };
        }
    };

    let mut detector = match Detector::new(&config.models.detector, config.auth.detection_threshold)
    {
        Ok(d) => d,
        Err(e) => {
            return AuthResponse {
                result: "fail".to_string(),
                reason: Some(format!("Failed to load detector: {}", e)),
                reason_code: None,
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
                reason_code: None,
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
            };
        }
    };

    let model_type = recognizer.model_type();
    let store = EnrollmentStore::new(
        &config.enrollment.store_path,
        config.enrollment.max_embeddings,
    );
    let embeddings = match store.load_embeddings(&username, model_type) {
        Ok(e) => e,
        Err(_) => {
            return AuthResponse {
                result: "fail".to_string(),
                reason: Some("User not enrolled".to_string()),
                reason_code: None,
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
            reason: Some(format!("User not enrolled for {} model", model_type)),
            reason_code: None,
            score: None,
            ir_mean: None,
            ir_std: None,
            liveness: None,
            users: None,
            timing: None,
        };
    }

    let total_start = std::time::Instant::now();

    // --- Phase 1: Capture first frame ---
    // No readiness pre-check on cold start. The face-acquire loop retries up to
    // max_frames times, so it naturally waits for the sensor to settle.
    // A separate readiness gate would just waste time rejecting frames that the
    // acquire loop would skip anyway.
    let mut acquire_frame = match camera.capture_frame() {
        Ok(f) => f,
        Err(e) => {
            let total_ms = total_start.elapsed().as_millis() as u64;
            tracing::warn!("Failed to capture first frame: {}", e);
            return AuthResponse {
                result: "fail".to_string(),
                reason: Some(format!("Camera capture failed: {}", e)),
                reason_code: Some("camera_error".to_string()),
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: Some(Timing {
                    detect_ms: 0,
                    recognize_ms: 0,
                    total_ms,
                }),
            };
        }
    };

    // --- Phase 2: Face-acquire with reactive settling ---
    // Try detection on each frame. If detection fails on the first frame,
    // enter a settling phase: sample frames at ~500ms intervals and wait for
    // frame-to-frame deltas to drop (sensor converged), then resume detection.
    let mut face_acquired = false;
    let mut total_detect_ms = 0u64;
    let mut settle_triggered = false;

    // Quick first attempt — if the camera is already warm, this succeeds immediately.
    let detect_start = std::time::Instant::now();
    match detector.detect_with_letterbox(
        &acquire_frame.data,
        acquire_frame.width,
        acquire_frame.height,
    ) {
        Ok(Some(_)) => {
            total_detect_ms += detect_start.elapsed().as_millis() as u64;
            face_acquired = true;
        }
        _ => {
            total_detect_ms += detect_start.elapsed().as_millis() as u64;
            // First attempt failed — camera may still be adjusting.
            // Enter settling phase: wait for the stream to stabilize.
            settle_triggered = true;
            let stats = PixelStats::compute(&acquire_frame);
            tracing::info!(
                "No face on first frame (mean={:.1}, std={:.1}), entering settling phase",
                stats.mean, stats.std
            );

            let settle = camera.wait_for_settle(
                stats.mean,
                stats.std,
                config.auth.camera_settle_max_frames,
                config.auth.camera_settle_delay_ms,
                config.auth.camera_warmup_delta_threshold,
                config.auth.camera_warmup_std_threshold,
            );

            // Use the settled frame for the remaining acquire attempts.
            match settle {
                SettleResult::Settled { frame } => {
                    acquire_frame = frame;
                }
                SettleResult::Timeout { frame } => {
                    // Timeout — try with whatever we have.
                    if !frame.data.is_empty() {
                        acquire_frame = frame;
                    }
                    // else: keep the original acquire_frame
                }
            }
        }
    }

    // Resume acquire loop if face not yet found.
    // Budget: if settling was triggered, use remaining frames; otherwise full budget.
    let remaining = if face_acquired {
        0
    } else if settle_triggered {
        // Settling already consumed some budget; give a few more detection attempts.
        config.auth.max_frames.saturating_sub(1).min(5)
    } else {
        config.auth.max_frames
    };

    for _ in 0..remaining {
        let detect_start = std::time::Instant::now();
        match detector.detect_with_letterbox(
            &acquire_frame.data,
            acquire_frame.width,
            acquire_frame.height,
        ) {
            Ok(Some(_)) => {
                total_detect_ms += detect_start.elapsed().as_millis() as u64;
                face_acquired = true;
                break;
            }
            _ => {
                total_detect_ms += detect_start.elapsed().as_millis() as u64;
                acquire_frame = match camera.capture_frame() {
                    Ok(f) => f,
                    Err(_) => continue,
                };
            }
        }
    }

    if !face_acquired {
        let total_ms = total_start.elapsed().as_millis() as u64;
        return AuthResponse {
            result: "fail".to_string(),
            reason: Some("No face detected within acquire budget".to_string()),
            reason_code: Some("no_face".to_string()),
            score: None,
            ir_mean: None,
            ir_std: None,
            liveness: None,
            users: None,
            timing: Some(Timing {
                detect_ms: total_detect_ms,
                recognize_ms: 0,
                total_ms,
            }),
        };
    }

    // --- Phase 3: Recognize ---
    let mut best_score = 0.0f32;
    let mut last_liveness = None;
    let mut last_ir_mean = None;
    let mut last_ir_std = None;
    let mut total_recognize_ms = 0u64;

    // Use the frame where face was acquired as the first recognize attempt
    let mut recog_frame = acquire_frame;

    for _ in 0..config.auth.max_frames {
        let detect_start = std::time::Instant::now();
        let face = match detector.detect_with_letterbox(
            &recog_frame.data,
            recog_frame.width,
            recog_frame.height,
        ) {
            Ok(Some(f)) => f,
            _ => {
                recog_frame = match camera.capture_frame() {
                    Ok(f) => f,
                    Err(_) => continue,
                };
                total_detect_ms += detect_start.elapsed().as_millis() as u64;
                continue;
            }
        };

        let liveness_result = liveness::check_liveness(
            &recog_frame.data,
            recog_frame.width,
            &face.bbox,
            config.auth.liveness_threshold,
        );
        total_detect_ms += detect_start.elapsed().as_millis() as u64;

        tracing::debug!(
            "Liveness: mean={:.1} std={:.1} live={}",
            liveness_result.ir_mean,
            liveness_result.ir_std,
            liveness_result.is_live
        );

        last_liveness = Some(liveness_result.is_live);
        last_ir_mean = Some(liveness_result.ir_mean);
        last_ir_std = Some(liveness_result.ir_std);

        if !liveness_result.is_live {
            recog_frame = match camera.capture_frame() {
                Ok(f) => f,
                Err(_) => continue,
            };
            continue;
        }

        let recog_start = std::time::Instant::now();
        let aligned = match recognizer.align_face(
            &recog_frame.data,
            recog_frame.width,
            recog_frame.height,
            &face.landmarks,
        ) {
            Ok(a) => a,
            Err(_) => {
                recog_frame = match camera.capture_frame() {
                    Ok(f) => f,
                    Err(_) => continue,
                };
                continue;
            }
        };

        let embedding = match recognizer.embed(&aligned) {
            Ok(e) => e,
            Err(_) => {
                recog_frame = match camera.capture_frame() {
                    Ok(f) => f,
                    Err(_) => continue,
                };
                continue;
            }
        };
        total_recognize_ms += recog_start.elapsed().as_millis() as u64;

        for enrolled in &embeddings {
            let score = crate::recognize::cosine_similarity(&embedding, enrolled);
            if score > best_score {
                best_score = score;
            }
        }

        recog_frame = match camera.capture_frame() {
            Ok(f) => f,
            Err(_) => continue,
        };
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
        reason_code: if result == "fail" {
            Some("score_below_threshold".to_string())
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
                reason_code: None,
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
            };
        }
    };

    // Phase 1: Capture first frame — no readiness pre-check (see authenticate flow).
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
                reason_code: None,
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
            };
        }
    };

    let _first_frame = match camera.capture_frame() {
        Ok(f) => f,
        Err(e) => {
            return AuthResponse {
                result: "fail".to_string(),
                reason: Some(format!("Camera capture failed: {}", e)),
                reason_code: None,
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
            };
        }
    };

    let mut detector = match Detector::new(&config.models.detector, config.auth.detection_threshold)
    {
        Ok(d) => d,
        Err(e) => {
            return AuthResponse {
                result: "fail".to_string(),
                reason: Some(format!("Failed to load detector: {}", e)),
                reason_code: None,
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
                reason_code: None,
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
            };
        }
    };

    let store = EnrollmentStore::new(
        &config.enrollment.store_path,
        config.enrollment.max_embeddings,
    );
    let mut enrolled_count = 0;
    let model_type = recognizer.model_type();

    for attempt in 0..3 {
        let mut captured = false;

        // Phase 2: Face-acquire for this attempt (with reactive settling)
        let mut face_frame = match camera.capture_frame() {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(
                    "Camera capture failed at start of attempt {}: {}",
                    attempt + 1,
                    e
                );
                continue;
            }
        };
        let mut face_found = false;

        // Quick first detection attempt.
        match detector.detect_with_letterbox(
            &face_frame.data,
            face_frame.width,
            face_frame.height,
        ) {
            Ok(Some(_)) => {
                face_found = true;
            }
            _ => {
                // First attempt failed — enter settling phase.
                let stats = PixelStats::compute(&face_frame);
                tracing::info!(
                    "No face on first frame (mean={:.1}, std={:.1}), entering settling phase",
                    stats.mean, stats.std
                );
                let settle = camera.wait_for_settle(
                    stats.mean,
                    stats.std,
                    config.auth.camera_settle_max_frames,
                    config.auth.camera_settle_delay_ms,
                    config.auth.camera_warmup_delta_threshold,
                    config.auth.camera_warmup_std_threshold,
                );
                match settle {
                    SettleResult::Settled { frame } => {
                        face_frame = frame;
                    }
                    SettleResult::Timeout { frame } => {
                        if !frame.data.is_empty() {
                            face_frame = frame;
                        }
                    }
                }
            }
        }

        // Resume acquire loop if face not yet found.
        if !face_found {
            let remaining = config.auth.max_frames.min(5);
            for _ in 0..remaining {
                match detector.detect_with_letterbox(
                    &face_frame.data,
                    face_frame.width,
                    face_frame.height,
                ) {
                    Ok(Some(_)) => {
                        face_found = true;
                        break;
                    }
                    _ => {
                        face_frame = match camera.capture_frame() {
                            Ok(f) => f,
                            Err(_) => continue,
                        };
                    }
                }
            }
        }

        if !face_found {
            tracing::warn!(
                "Failed to capture face for enrollment attempt {}",
                attempt + 1
            );
            continue;
        }

        // Phase 3: Capture embedding
        for _ in 0..config.auth.max_frames {
            let frame = match camera.capture_frame() {
                Ok(f) => f,
                Err(e) => {
                    tracing::warn!("Camera capture failed: {}", e);
                    continue;
                }
            };

            let stats = crate::camera::PixelStats::compute(&frame);
            tracing::info!(
                "Frame captured: {}x{}, pixels: min={} max={} mean={:.1}",
                frame.width,
                frame.height,
                stats.min,
                stats.max,
                stats.mean
            );

            let face = match detector.detect_with_letterbox(&frame.data, frame.width, frame.height)
            {
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

            let liveness_result = liveness::check_liveness(
                &frame.data,
                frame.width,
                &face.bbox,
                config.auth.liveness_threshold,
            );
            tracing::debug!(
                "Enroll liveness: mean={:.1} std={:.1} live={}",
                liveness_result.ir_mean,
                liveness_result.ir_std,
                liveness_result.is_live
            );
            if !liveness_result.is_live {
                tracing::info!(
                    "Liveness check failed: mean={:.1} std={:.1} (threshold={})",
                    liveness_result.ir_mean,
                    liveness_result.ir_std,
                    config.auth.liveness_threshold
                );
                continue;
            }

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
                    tracing::warn!(
                        "Embedding failed: {} (aligned shape: {:?})",
                        e,
                        aligned.shape()
                    );
                    continue;
                }
            };

            if let Err(e) = store.save_embedding(&username, &embedding, model_type) {
                tracing::error!("Failed to save embedding: {}", e);
                continue;
            }

            enrolled_count += 1;
            captured = true;
            break;
        }

        if !captured {
            tracing::warn!(
                "Failed to capture face for enrollment attempt {}",
                attempt + 1
            );
        }
    }

    if enrolled_count > 0 {
        AuthResponse {
            result: "ok".to_string(),
            reason: Some(format!("Enrolled {} embeddings", enrolled_count)),
            reason_code: None,
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
            reason_code: None,
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
                reason_code: None,
                score: None,
                ir_mean: None,
                ir_std: None,
                liveness: None,
                users: None,
                timing: None,
            };
        }
    };

    let store = EnrollmentStore::new(
        &config.enrollment.store_path,
        config.enrollment.max_embeddings,
    );
    match store.clear_embeddings(&username) {
        Ok(()) => AuthResponse {
            result: "ok".to_string(),
            reason: None,
            reason_code: None,
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
            reason_code: None,
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
    let store = EnrollmentStore::new(
        &config.enrollment.store_path,
        config.enrollment.max_embeddings,
    );
    let users = store.list_users().unwrap_or_default();

    let user_infos: Vec<UserInfo> = users
        .into_iter()
        .map(|(username, model_counts)| {
            let mut arcface_count = 0;
            let mut adaface_count = 0;
            for (model, count) in model_counts {
                match model {
                    ModelType::ArcFace => arcface_count = count,
                    ModelType::AdaFace => adaface_count = count,
                }
            }
            UserInfo {
                username,
                arcface_count,
                adaface_count,
            }
        })
        .collect();

    AuthResponse {
        result: "ok".to_string(),
        reason: None,
        reason_code: None,
        score: None,
        ir_mean: None,
        ir_std: None,
        liveness: None,
        users: Some(user_infos),
        timing: None,
    }
}

async fn handle_view(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    config: &Config,
    user: Option<&str>,
) -> Result<()> {
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

    let store = EnrollmentStore::new(
        &config.enrollment.store_path,
        config.enrollment.max_embeddings,
    );

    let mut camera = match Camera::new(
        &config.camera.device,
        config.camera.width,
        config.camera.height,
        config.camera.pixel_format.as_deref(),
    ) {
        Ok(c) => c,
        Err(e) => {
            let err =
                serde_json::json!({"type": "error", "message": format!("Camera error: {}", e)});
            let mut msg = serde_json::to_string(&err)?;
            msg.push('\n');
            writer.write_all(msg.as_bytes()).await?;
            return Ok(());
        }
    };

    // Capture first frame — no readiness pre-check (see authenticate flow).
    let _first_frame = match camera.capture_frame() {
        Ok(f) => f,
        Err(e) => {
            let err =
                serde_json::json!({"type": "error", "message": format!("Camera capture failed: {}", e)});
            let mut msg = serde_json::to_string(&err)?;
            msg.push('\n');
            writer.write_all(msg.as_bytes()).await?;
            return Ok(());
        }
    };

    // Live-readiness tracker for the streaming loop — status overlay only,
    // does not block detection.
    let mut delta_tracker = DeltaTracker::new();
    let mut black_consecutive: u32 = 0;

    let mut detector = match Detector::new(&config.models.detector, config.auth.detection_threshold)
    {
        Ok(d) => d,
        Err(e) => {
            let err =
                serde_json::json!({"type": "error", "message": format!("Detector error: {}", e)});
            let mut msg = serde_json::to_string(&err)?;
            msg.push('\n');
            writer.write_all(msg.as_bytes()).await?;
            return Ok(());
        }
    };

    let mut recognizer = match Recognizer::new(&config.models.recognizer) {
        Ok(r) => r,
        Err(e) => {
            let err =
                serde_json::json!({"type": "error", "message": format!("Recognizer error: {}", e)});
            let mut msg = serde_json::to_string(&err)?;
            msg.push('\n');
            writer.write_all(msg.as_bytes()).await?;
            return Ok(());
        }
    };

    let threshold = config.auth.similarity_threshold;
    let engine = base64::engine::general_purpose::STANDARD;
    let readiness_cfg = readiness_config(&config);

    loop {
        let frame = match camera.capture_frame() {
            Ok(f) => f,
            Err(_) => {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                continue;
            }
        };

        let image_b64 = engine.encode(&frame.data);

        let mut msg = serde_json::json!({
            "type": "frame",
            "image": image_b64,
            "width": frame.width,
            "height": frame.height,
        });

        // Run readiness state machine on every streamed frame.
        let prev_mean = delta_tracker.previous_mean();
        let stats = PixelStats::compute(&frame);
        let delta = delta_tracker.update(stats.mean);
        let class = classify_frame(stats.mean, delta, stats.std, &readiness_cfg);

        let (status_text, status_kind, blocked) = match class {
            FrameClass::Black => {
                let rising = prev_mean.is_some() && stats.mean > prev_mean.unwrap_or(0.0);
                if rising {
                    black_consecutive = 0;
                } else {
                    black_consecutive += 1;
                }
                let is_blocked = black_consecutive >= BLACK_PATIENCE;
                (
                    if is_blocked {
                        "CAMERA BLOCKED - open the shutter".to_string()
                    } else {
                        format!("camera: black | mean={:.1} (waiting)", stats.mean)
                    },
                    if is_blocked { "blocked" } else { "black" },
                    is_blocked,
                )
            }
            FrameClass::Unstable => {
                black_consecutive = 0;
                let label = if prev_mean.is_some() {
                    format!("camera: unstable | dmean={:.1} (waiting)", delta)
                } else {
                    "camera: warming up (first frame)".to_string()
                };
                (label, "unstable", false)
            }
            FrameClass::Stable => {
                black_consecutive = 0;
                ("camera: ready".to_string(), "ready", false)
            }
        };

        msg["status"] = serde_json::json!(status_text);
        msg["status_kind"] = serde_json::json!(status_kind);
        msg["mean"] = serde_json::json!(stats.mean);
        msg["delta"] = serde_json::json!(delta);

        // Short-circuit: when permanently black, throttle + skip detection.
        if blocked {
            let mut json = serde_json::to_string(&msg)?;
            json.push('\n');
            if writer.write_all(json.as_bytes()).await.is_err() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            continue;
        }

        // Always run detection — the acquire loop naturally skips bad frames.
        // The readiness overlay is informational only, not a gate.
        {
            let detect_start = Instant::now();
            let face = detector
                .detect_with_letterbox(&frame.data, frame.width, frame.height)
                .ok()
                .flatten();
            let detect_ms = detect_start.elapsed().as_millis() as u64;
            msg["detect_ms"] = serde_json::json!(detect_ms);

            if let Some(f) = face {
                let liveness_result = liveness::check_liveness(
                    &frame.data,
                    frame.width,
                    &f.bbox,
                    config.auth.liveness_threshold,
                );

                msg["liveness"] = serde_json::json!({
                    "is_live": liveness_result.is_live,
                    "ir_mean": liveness_result.ir_mean,
                    "ir_std": liveness_result.ir_std,
                    "threshold": config.auth.liveness_threshold,
                });

                let mut best_score: f32 = 0.0;
                let mut authenticated = false;
                let mut reason: Option<String> = None;

                if !liveness_result.is_live {
                    reason = Some(format!(
                        "liveness: mean={:.1} std={:.1} (need std/{:.2} >= {:.2})",
                        liveness_result.ir_mean,
                        liveness_result.ir_std,
                        255.0,
                        config.auth.liveness_threshold,
                    ));
                } else {
                    let model_type = recognizer.model_type();
                    let embeddings = store.load_embeddings(&user, model_type).unwrap_or_default();
                    if embeddings.is_empty() {
                        reason = Some("no enrolled embeddings".to_string());
                    } else if let Ok(aligned) = recognizer.align_face(
                        &frame.data,
                        frame.width,
                        frame.height,
                        &f.landmarks,
                    ) {
                        if let Ok(embedding) = recognizer.embed(&aligned) {
                            for enrolled in &embeddings {
                                let score =
                                    crate::recognize::cosine_similarity(&embedding, enrolled);
                                if score > best_score {
                                    best_score = score;
                                }
                            }
                            authenticated = best_score >= threshold;
                            if !authenticated {
                                reason = Some(format!(
                                    "score {:.4} < threshold {:.4}",
                                    best_score, threshold
                                ));
                            }
                        } else {
                            reason = Some("embedding failed".to_string());
                        }
                    } else {
                        reason = Some("alignment failed".to_string());
                    }
                }

                msg["face"] = serde_json::json!({
                    "bbox": {"x": f.bbox.x, "y": f.bbox.y, "w": f.bbox.w, "h": f.bbox.h},
                    "confidence": f.confidence,
                    "landmarks": f.landmarks.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>(),
                    "authenticated": authenticated,
                    "score": best_score,
                });

                if let Some(r) = &reason {
                    msg["reason"] = serde_json::json!(r);
                }

                let status_suffix = if authenticated {
                    format!("score: {:.2} PASS", best_score)
                } else if let Some(r) = &reason {
                    format!("FAIL | {}", r)
                } else {
                    format!("score: {:.2} FAIL", best_score)
                };
                msg["status"] = serde_json::json!(format!("{} | {}", status_text, status_suffix));
            } else {
                msg["status"] = serde_json::json!(format!("{} | no face", status_text));
            }
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
