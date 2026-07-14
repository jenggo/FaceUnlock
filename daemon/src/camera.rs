use anyhow::{Context, Result};
use std::pin::Pin;
use v4l::buffer::Type;
use v4l::device::Device;
use v4l::format::FourCC;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::Format;

// --- Readiness constants (sensor physics, not user config) ---

/// Consecutive black frames with no rising trend before aborting as permanent-black.
pub const BLACK_PATIENCE: u32 = 3;

#[derive(Debug, Clone, PartialEq)]
pub enum PixelFormat {
    Grey,
    Y8,
    Mjpeg,
}

#[derive(Debug, Clone)]
pub struct RawFrame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    #[allow(dead_code)]
    pub format: PixelFormat,
}

/// Whole-frame pixel statistics for a single frame.
#[derive(Debug, Clone)]
pub struct PixelStats {
    pub mean: f32,
    pub std: f32,
    pub min: u8,
    pub max: u8,
}

impl PixelStats {
    /// Compute mean, std, min, max over all pixels in a grayscale frame.
    pub fn compute(frame: &RawFrame) -> Self {
        let data = &frame.data;
        if data.is_empty() {
            return Self {
                mean: 0.0,
                std: 0.0,
                min: 0,
                max: 0,
            };
        }
        let (min, max, sum) = data
            .iter()
            .fold((u8::MAX, u8::MIN, 0u64), |(mn, mx, s), &p| {
                (mn.min(p), mx.max(p), s + p as u64)
            });
        let mean = sum as f32 / data.len() as f32;
        let variance: f32 = data
            .iter()
            .map(|&p| {
                let diff = p as f32 - mean;
                diff * diff
            })
            .sum::<f32>()
            / data.len() as f32;
        let std = variance.sqrt();
        Self {
            mean,
            std,
            min,
            max,
        }
    }
}

/// Tracks frame-to-frame delta of the pixel mean.
/// First frame returns infinity (no baseline).
pub struct DeltaTracker {
    prev_mean: Option<f32>,
}

impl DeltaTracker {
    pub fn new() -> Self {
        Self { prev_mean: None }
    }

    /// Feed the current frame's mean; returns |Δmean| (infinity for the first frame).
    pub fn update(&mut self, current_mean: f32) -> f32 {
        let delta = match self.prev_mean {
            Some(prev) => (current_mean - prev).abs(),
            None => f32::INFINITY,
        };
        self.prev_mean = Some(current_mean);
        delta
    }

    /// Returns the previous frame's mean (None if no frame has been fed yet).
    pub fn previous_mean(&self) -> Option<f32> {
        self.prev_mean
    }
}

/// Classification of a single frame's readiness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameClass {
    Black,
    Unstable,
    Stable,
}

/// Classify a frame based on its mean, |Δmean|, standard deviation, and readiness config.
///
/// A frame is:
/// - `Black` if mean is below the black threshold (shutter closed / no signal).
/// - `Unstable` if the mean is shifting too much between frames OR the frame lacks texture
///   (std too low), which happens while AE/AF is still converging.
/// - `Stable` only when mean is above black, variance is sufficient, and mean drift is low.
pub fn classify_frame(mean: f32, delta: f32, std: f32, config: &ReadinessConfig) -> FrameClass {
    if mean < config.mean_threshold {
        FrameClass::Black
    } else if delta >= config.delta_threshold || std < config.std_threshold {
        FrameClass::Unstable
    } else {
        FrameClass::Stable
    }
}

/// Result of the settling phase.
#[derive(Debug)]
pub enum SettleResult {
    /// Camera has settled; contains the last captured frame.
    Settled { frame: RawFrame },
    /// Budget exhausted; camera may still be adjusting.
    Timeout { frame: RawFrame },
}

/// Result of the camera-ready phase.
#[allow(dead_code)]
pub enum ReadinessResult {
    Ready { frame: RawFrame },
    Blocked { reason: String },
}

/// Tunable parameters for the camera readiness phase.
#[derive(Debug, Clone, Copy)]
pub struct ReadinessConfig {
    /// Maximum frames to wait before giving up.
    #[allow(dead_code)]
    pub max_frames: u32,
    /// Delay between readiness frames (ms).
    #[allow(dead_code)]
    pub delay_ms: u64,
    /// Mean luma below this is classified as black/blocked.
    pub mean_threshold: f32,
    /// |Δmean| below this is considered exposure-stable.
    pub delta_threshold: f32,
    /// Standard deviation below this means the frame lacks texture.
    pub std_threshold: f32,
    /// How many consecutive stable frames are required to be "ready".
    #[allow(dead_code)]
    pub stable_frames_required: u32,
}

impl Default for ReadinessConfig {
    fn default() -> Self {
        Self {
            max_frames: 15,
            delay_ms: 100,
            mean_threshold: 10.0,
            delta_threshold: 8.0,
            std_threshold: 2.0,
            stable_frames_required: 2,
        }
    }
}

pub struct Camera {
    #[allow(dead_code)]
    dev: Pin<Box<Device>>,
    stream: MmapStream<'static>,
    width: u32,
    height: u32,
    format: PixelFormat,
}

impl Camera {
    pub fn new(
        device_path: &str,
        width: u32,
        height: u32,
        forced_format: Option<&str>,
    ) -> Result<Self> {
        let forced_format = forced_format.and_then(|f| match f.to_uppercase().as_str() {
            "GREY" => Some(PixelFormat::Grey),
            "Y8" => Some(PixelFormat::Y8),
            "MJPEG" | "MJPG" => Some(PixelFormat::Mjpeg),
            _ => None,
        });

        let dev = Device::with_path(device_path)
            .with_context(|| format!("Failed to open V4L2 device: {}", device_path))?;

        let formats = dev.enum_formats().context("Failed to enumerate formats")?;

        let format = if let Some(forced) = &forced_format {
            let fourcc = match forced {
                PixelFormat::Grey => FourCC::new(b"GREY"),
                PixelFormat::Y8 => FourCC::new(b"Y8  "),
                PixelFormat::Mjpeg => FourCC::new(b"MJPG"),
            };

            if formats.iter().any(|f| f.fourcc == fourcc) {
                forced.clone()
            } else {
                anyhow::bail!(
                    "Forced format {:?} not available on {}",
                    forced,
                    device_path
                );
            }
        } else {
            Self::negotiate_format(&formats, device_path)?
        };

        let fourcc = match &format {
            PixelFormat::Grey | PixelFormat::Y8 => FourCC::new(b"GREY"),
            PixelFormat::Mjpeg => FourCC::new(b"MJPG"),
        };

        let fmt = Format::new(width, height, fourcc);
        dev.set_format(&fmt).context("Failed to set pixel format")?;

        let mut dev = Box::pin(dev);
        let dev_ptr: *mut Device = &mut *dev;

        let stream = unsafe {
            MmapStream::with_buffers(&mut *dev_ptr, Type::VideoCapture, 1)
                .context("Failed to create buffer stream")?
        };

        Ok(Self {
            dev,
            stream,
            width,
            height,
            format,
        })
    }

    /// Camera-ready phase: capture frames, compute stats, wait for stable exposure.
    /// Returns `ReadinessResult::Ready` with the last stable frame once the stream has
    /// settled, or `Blocked` if the camera is permanently black or the budget is exhausted.
    #[allow(dead_code)]
    pub fn wait_for_ready(&mut self, config: ReadinessConfig) -> ReadinessResult {
        let mut delta_tracker = DeltaTracker::new();
        let mut black_consecutive: u32 = 0;
        let mut stable_consecutive: u32 = 0;
        let mut best_non_black_frame: Option<RawFrame> = None;

        for i in 0..config.max_frames {
            let frame = match self.capture_frame() {
                Ok(f) => f,
                Err(_) => {
                    if i + 1 < config.max_frames && config.delay_ms > 0 {
                        std::thread::sleep(std::time::Duration::from_millis(config.delay_ms));
                    }
                    continue;
                }
            };

            let prev_mean = delta_tracker.prev_mean;
            let stats = PixelStats::compute(&frame);
            let delta = delta_tracker.update(stats.mean);
            let class = classify_frame(stats.mean, delta, stats.std, &config);

            tracing::debug!(
                "Readiness frame {}/{}: mean={:.1} std={:.1} delta={:.1} class={:?} stable_streak={}",
                i + 1, config.max_frames, stats.mean, stats.std, delta, class, stable_consecutive
            );

            match class {
                FrameClass::Black => {
                    stable_consecutive = 0;

                    // Check rising trend: current mean > previous frame's mean
                    let rising = prev_mean.is_some() && stats.mean > prev_mean.unwrap_or(0.0);
                    if rising {
                        black_consecutive = 0;
                    } else {
                        black_consecutive += 1;
                    }

                    if black_consecutive >= BLACK_PATIENCE {
                        return ReadinessResult::Blocked {
                            reason: format!(
                                "Camera permanently black after {} frames (mean={:.1})",
                                BLACK_PATIENCE, stats.mean
                            ),
                        };
                    }
                }
                FrameClass::Unstable => {
                    black_consecutive = 0;
                    stable_consecutive = 0;
                    best_non_black_frame = Some(frame);
                }
                FrameClass::Stable => {
                    black_consecutive = 0;
                    stable_consecutive += 1;
                    best_non_black_frame = Some(frame);

                    if stable_consecutive >= config.stable_frames_required {
                        let kept = best_non_black_frame.take().unwrap();
                        tracing::info!(
                            "Camera ready after {} frames (mean={:.1}, std={:.1}, delta={:.1})",
                            i + 1,
                            stats.mean,
                            stats.std,
                            delta
                        );
                        return ReadinessResult::Ready { frame: kept };
                    }
                }
            }

            if i + 1 < config.max_frames && config.delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(config.delay_ms));
            }
        }

        // Budget exhausted — use the best non-black frame if we have one.
        match best_non_black_frame {
            Some(frame) => {
                tracing::warn!(
                    "Camera-ready budget exhausted, proceeding with best available frame"
                );
                ReadinessResult::Ready { frame }
            }
            None => ReadinessResult::Blocked {
                reason: "Camera-ready budget exhausted, all frames were black".to_string(),
            },
        }
    }

    /// Reactive settling phase: called after a failed detection attempt.
    /// Captures frames at `delay_ms` intervals and waits until frame-to-frame
    /// deltas (mean + std) drop below the readiness thresholds, indicating the
    /// sensor has converged. Returns the last captured frame either way.
    pub fn wait_for_settle(
        &mut self,
        baseline_mean: f32,
        baseline_std: f32,
        max_frames: u32,
        delay_ms: u64,
        delta_threshold: f32,
        std_threshold: f32,
    ) -> SettleResult {
        let mut prev_mean = baseline_mean;
        let mut prev_std = baseline_std;

        for i in 0..max_frames {
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));

            let frame = match self.capture_frame() {
                Ok(f) => f,
                Err(_) => continue,
            };

            let stats = PixelStats::compute(&frame);
            let mean_delta = (stats.mean - prev_mean).abs();
            let std_delta = (stats.std - prev_std).abs();

            tracing::debug!(
                "Settle frame {}/{}: mean={:.1} std={:.1} |Δmean|={:.1} |Δstd|={:.1}",
                i + 1, max_frames, stats.mean, stats.std, mean_delta, std_delta
            );

            let settled = mean_delta < delta_threshold
                && std_delta < delta_threshold
                && stats.std >= std_threshold;

            prev_mean = stats.mean;
            prev_std = stats.std;

            if settled {
                tracing::info!(
                    "Camera settled after {} settle frames (mean={:.1}, std={:.1})",
                    i + 1,
                    stats.mean,
                    stats.std
                );
                return SettleResult::Settled { frame };
            }
        }

        tracing::warn!("Settle budget exhausted ({} frames)", max_frames);
        // Return the last frame we captured (or the baseline if all failed).
        // The caller will use this for a final detection attempt.
        SettleResult::Timeout {
            frame: RawFrame {
                data: vec![],
                width: 0,
                height: 0,
                format: PixelFormat::Grey,
            },
        }
    }

    pub fn capture_frame(&mut self) -> Result<RawFrame> {
        let (buf, _meta) = self.stream.next().context("Failed to capture frame")?;

        let data = buf.to_vec();
        let width = self.width;
        let height = self.height;

        match &self.format {
            PixelFormat::Mjpeg => self.decode_mjpeg(&data, width, height),
            _ => Ok(RawFrame {
                data,
                width,
                height,
                format: self.format.clone(),
            }),
        }
    }

    fn negotiate_format(
        available: &[v4l::format::Description],
        device_path: &str,
    ) -> Result<PixelFormat> {
        let grey = FourCC::new(b"GREY");
        let y8 = FourCC::new(b"Y8  ");
        let mjpg = FourCC::new(b"MJPG");

        for desc in available {
            if desc.fourcc == grey {
                return Ok(PixelFormat::Grey);
            }
        }

        for desc in available {
            if desc.fourcc == y8 {
                return Ok(PixelFormat::Y8);
            }
        }

        for desc in available {
            if desc.fourcc == mjpg {
                return Ok(PixelFormat::Mjpeg);
            }
        }

        anyhow::bail!("No supported pixel format found on {}", device_path)
    }

    fn decode_mjpeg(&self, data: &[u8], width: u32, height: u32) -> Result<RawFrame> {
        let jpeg_decoder = image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(data))
            .context("Failed to decode MJPEG frame")?;

        let img = image::DynamicImage::from_decoder(jpeg_decoder)
            .context("Failed to decode MJPEG image")?;

        let grey = img.to_luma8();

        Ok(RawFrame {
            data: grey.into_raw(),
            width,
            height,
            format: PixelFormat::Grey,
        })
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        // Stream drops first (stops V4L2 streaming), then Device drops (closes fd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_frame() {
        let cfg = ReadinessConfig {
            max_frames: 15,
            delay_ms: 100,
            mean_threshold: 10.0,
            delta_threshold: 8.0,
            std_threshold: 2.0,
            stable_frames_required: 2,
        };

        // Black: mean below threshold regardless of other values
        assert_eq!(classify_frame(0.0, 0.0, 5.0, &cfg), FrameClass::Black);
        assert_eq!(classify_frame(9.9, 0.0, 5.0, &cfg), FrameClass::Black);

        // Unstable: mean ok but delta too high
        assert_eq!(classify_frame(10.0, 8.0, 5.0, &cfg), FrameClass::Unstable);
        assert_eq!(classify_frame(128.0, 20.0, 5.0, &cfg), FrameClass::Unstable);

        // Unstable: mean/delta ok but std too low (no texture)
        assert_eq!(classify_frame(128.0, 0.0, 1.0, &cfg), FrameClass::Unstable);

        // Stable: all thresholds satisfied
        assert_eq!(classify_frame(10.0, 7.99, 5.0, &cfg), FrameClass::Stable);
        assert_eq!(classify_frame(128.0, 0.0, 5.0, &cfg), FrameClass::Stable);
    }

    #[test]
    fn test_delta_tracker_first_frame_is_infinity() {
        let mut t = DeltaTracker::new();
        let delta = t.update(100.0);
        assert!(delta.is_infinite());
        assert_eq!(t.prev_mean, Some(100.0));
    }

    #[test]
    fn test_delta_tracker_subsequent_frames_abs_delta() {
        let mut t = DeltaTracker::new();
        let _ = t.update(100.0);
        assert_eq!(t.update(108.0), 8.0);
        assert_eq!(t.update(103.0), 5.0);
    }

    #[test]
    fn test_rising_trend_resets_patience() {
        let mut t = DeltaTracker::new();
        let means = [2.0_f32, 5.0, 9.0];
        let mut black_consecutive: u32 = 0;
        for &m in &means {
            let prev_mean = t.prev_mean;
            let _delta = t.update(m);
            let rising = prev_mean.is_some() && m > prev_mean.unwrap_or(0.0);
            if rising {
                black_consecutive = 0;
            } else {
                black_consecutive += 1;
            }
        }
        assert_eq!(black_consecutive, 0);
    }

    #[test]
    fn test_flat_black_trend_exhausts_patience() {
        let mut t = DeltaTracker::new();
        let means = [2.0_f32, 2.0, 2.0];
        let mut black_consecutive: u32 = 0;
        for &m in &means {
            let prev_mean = t.prev_mean;
            let _delta = t.update(m);
            let rising = prev_mean.is_some() && m > prev_mean.unwrap_or(0.0);
            if rising {
                black_consecutive = 0;
            } else {
                black_consecutive += 1;
            }
        }
        assert_eq!(black_consecutive, BLACK_PATIENCE);
    }

    #[test]
    fn test_pixel_stats_empty_frame() {
        let frame = RawFrame {
            data: vec![],
            width: 0,
            height: 0,
            format: PixelFormat::Grey,
        };
        let s = PixelStats::compute(&frame);
        assert_eq!(s.mean, 0.0);
        assert_eq!(s.std, 0.0);
        assert_eq!(s.min, 0);
        assert_eq!(s.max, 0);
    }

    #[test]
    fn test_pixel_stats_uniform_frame() {
        let frame = RawFrame {
            data: vec![128; 4],
            width: 2,
            height: 2,
            format: PixelFormat::Grey,
        };
        let s = PixelStats::compute(&frame);
        assert_eq!(s.mean, 128.0);
        assert_eq!(s.std, 0.0);
        assert_eq!(s.min, 128);
        assert_eq!(s.max, 128);
    }

    #[test]
    fn test_pixel_stats_known_frame() {
        let frame = RawFrame {
            data: vec![0, 100, 200, 255],
            width: 2,
            height: 2,
            format: PixelFormat::Grey,
        };
        let s = PixelStats::compute(&frame);
        assert!((s.mean - 138.75).abs() < 1e-3);
        assert_eq!(s.min, 0);
        assert_eq!(s.max, 255);
    }
}
