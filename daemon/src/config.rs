use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

const PRIMARY_CONFIG: &str = "/etc/faceunlock/faceunlock.toml";
const FALLBACK_CONFIG: &str = "/etc/faceunlock.toml";

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub camera: CameraConfig,
    pub models: ModelsConfig,
    #[serde(default)]
    pub auth: AuthConfig,
    #[serde(default)]
    pub enrollment: EnrollmentConfig,
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CameraConfig {
    pub device: String,
    #[serde(default = "default_width")]
    pub width: u32,
    #[serde(default = "default_height")]
    pub height: u32,
    pub pixel_format: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelsConfig {
    pub detector: String,
    pub recognizer: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AuthConfig {
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,
    #[serde(default = "default_max_frames")]
    pub max_frames: u32,
    #[serde(default = "default_liveness_threshold")]
    pub liveness_threshold: f32,
    #[serde(default = "default_detection_threshold")]
    pub detection_threshold: f32,
    #[serde(default = "default_camera_warmup")]
    pub camera_warmup: u32,
    #[serde(default = "default_camera_warmup_delay_ms")]
    pub camera_warmup_delay_ms: u64,
    #[serde(default = "default_camera_warmup_mean_threshold")]
    pub camera_warmup_mean_threshold: f32,
    #[serde(default = "default_camera_warmup_delta_threshold")]
    pub camera_warmup_delta_threshold: f32,
    #[serde(default = "default_camera_warmup_std_threshold")]
    pub camera_warmup_std_threshold: f32,
    #[serde(default = "default_camera_warmup_stable_frames")]
    pub camera_warmup_stable_frames: u32,
    #[serde(default = "default_camera_settle_delay_ms")]
    pub camera_settle_delay_ms: u64,
    #[serde(default = "default_camera_settle_max_frames")]
    pub camera_settle_max_frames: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct EnrollmentConfig {
    #[serde(default = "default_store_path")]
    pub store_path: String,
    #[serde(default = "default_max_embeddings")]
    pub max_embeddings: usize,
}

fn default_width() -> u32 {
    640
}
fn default_height() -> u32 {
    480
}
fn default_similarity_threshold() -> f32 {
    0.40
}
fn default_max_frames() -> u32 {
    10
}
fn default_liveness_threshold() -> f32 {
    0.15
}
fn default_detection_threshold() -> f32 {
    0.5
}
/// Maximum frames to wait for stable exposure in the camera-ready phase.
fn default_camera_warmup() -> u32 {
    15
}
fn default_camera_warmup_delay_ms() -> u64 {
    100
}
/// Mean luma below this is classified as black/blocked.
fn default_camera_warmup_mean_threshold() -> f32 {
    10.0
}
/// |Δmean| below this is considered exposure-stable.
fn default_camera_warmup_delta_threshold() -> f32 {
    8.0
}
/// Minimum luma standard deviation for a frame to be considered textured.
/// Below this the frame is treated as black/blocked or not yet converged.
fn default_camera_warmup_std_threshold() -> f32 {
    2.0
}
/// Number of consecutive stable frames required before the camera is considered ready.
fn default_camera_warmup_stable_frames() -> u32 {
    2
}
/// Delay between settling-check frames (ms) after a failed detection.
fn default_camera_settle_delay_ms() -> u64 {
    500
}
/// Maximum settling-check frames before giving up and resuming detection anyway.
fn default_camera_settle_max_frames() -> u32 {
    10
}
fn default_store_path() -> String {
    "/var/lib/faceunlock/embeddings".to_string()
}
fn default_max_embeddings() -> usize {
    5
}
fn default_log_level() -> String {
    "info".to_string()
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: default_similarity_threshold(),
            max_frames: default_max_frames(),
            liveness_threshold: default_liveness_threshold(),
            detection_threshold: default_detection_threshold(),
            camera_warmup: default_camera_warmup(),
            camera_warmup_delay_ms: default_camera_warmup_delay_ms(),
            camera_warmup_mean_threshold: default_camera_warmup_mean_threshold(),
            camera_warmup_delta_threshold: default_camera_warmup_delta_threshold(),
            camera_warmup_std_threshold: default_camera_warmup_std_threshold(),
            camera_warmup_stable_frames: default_camera_warmup_stable_frames(),
            camera_settle_delay_ms: default_camera_settle_delay_ms(),
            camera_settle_max_frames: default_camera_settle_max_frames(),
        }
    }
}

impl Default for EnrollmentConfig {
    fn default() -> Self {
        Self {
            store_path: default_store_path(),
            max_embeddings: default_max_embeddings(),
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let path = if Path::new(PRIMARY_CONFIG).exists() {
            PRIMARY_CONFIG
        } else if Path::new(FALLBACK_CONFIG).exists() {
            FALLBACK_CONFIG
        } else {
            anyhow::bail!(
                "Config file not found at {} or {}",
                PRIMARY_CONFIG,
                FALLBACK_CONFIG
            );
        };

        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path))?;

        let config: Config = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config: {}", path))?;

        config.validate()?;

        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.camera.device.is_empty() {
            anyhow::bail!("camera.device is required and must not be empty");
        }

        if !(0.0..=1.0).contains(&self.auth.similarity_threshold) {
            anyhow::bail!(
                "auth.similarity_threshold must be between 0.0 and 1.0, got {}",
                self.auth.similarity_threshold
            );
        }

        if !(0.0..=1.0).contains(&self.auth.liveness_threshold) {
            anyhow::bail!(
                "auth.liveness_threshold must be between 0.0 and 1.0, got {}",
                self.auth.liveness_threshold
            );
        }

        if !(0.0..=1.0).contains(&self.auth.detection_threshold) {
            anyhow::bail!(
                "auth.detection_threshold must be between 0.0 and 1.0, got {}",
                self.auth.detection_threshold
            );
        }

        if self.auth.camera_warmup_mean_threshold < 0.0 {
            anyhow::bail!(
                "auth.camera_warmup_mean_threshold must be non-negative, got {}",
                self.auth.camera_warmup_mean_threshold
            );
        }

        if self.auth.camera_warmup_delta_threshold < 0.0 {
            anyhow::bail!(
                "auth.camera_warmup_delta_threshold must be non-negative, got {}",
                self.auth.camera_warmup_delta_threshold
            );
        }

        if self.auth.camera_warmup_std_threshold < 0.0 {
            anyhow::bail!(
                "auth.camera_warmup_std_threshold must be non-negative, got {}",
                self.auth.camera_warmup_std_threshold
            );
        }

        if self.auth.camera_warmup_stable_frames == 0 {
            anyhow::bail!("auth.camera_warmup_stable_frames must be at least 1");
        }

        if self.auth.camera_settle_max_frames == 0 {
            anyhow::bail!("auth.camera_settle_max_frames must be at least 1");
        }

        if self.models.detector.is_empty() {
            anyhow::bail!("models.detector is required");
        }

        if self.models.recognizer.is_empty() {
            anyhow::bail!("models.recognizer is required");
        }

        Ok(())
    }
}
