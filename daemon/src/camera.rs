use anyhow::{Context, Result};
use std::pin::Pin;
use v4l::buffer::Type;
use v4l::device::Device;
use v4l::format::FourCC;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::Format;

#[derive(Debug, Clone, PartialEq)]
pub enum PixelFormat {
    Grey,
    Y8,
    Mjpeg,
}

#[derive(Debug)]
pub struct RawFrame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    #[allow(dead_code)]
    pub format: PixelFormat,
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
    pub fn new(device_path: &str, width: u32, height: u32, forced_format: Option<&str>) -> Result<Self> {
        let forced_format = forced_format.and_then(|f| match f.to_uppercase().as_str() {
            "GREY" => Some(PixelFormat::Grey),
            "Y8" => Some(PixelFormat::Y8),
            "MJPEG" | "MJPG" => Some(PixelFormat::Mjpeg),
            _ => None,
        });

        let dev = Device::with_path(device_path)
            .with_context(|| format!("Failed to open V4L2 device: {}", device_path))?;

        let formats = dev.enum_formats()
            .context("Failed to enumerate formats")?;

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
        dev.set_format(&fmt)
            .context("Failed to set pixel format")?;

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

    pub fn warmup(&mut self, frames: u32) {
        for _ in 0..frames {
            let _ = self.capture_frame();
        }
    }

    pub fn capture_frame(&mut self) -> Result<RawFrame> {
        let (buf, _meta) = self.stream.next()
            .context("Failed to capture frame")?;

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

    fn negotiate_format(available: &[v4l::format::Description], device_path: &str) -> Result<PixelFormat> {
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

        anyhow::bail!(
            "No supported pixel format found on {}",
            device_path
        )
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
