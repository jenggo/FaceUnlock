mod camera;
mod config;
mod detect;
mod enroll;
mod liveness;
mod recognize;
mod socket;

use anyhow::Result;
use config::Config;
use std::fs::{File, OpenOptions};
use std::io;
use std::sync::Mutex;
use tracing_appender::non_blocking;
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

struct LocalTimer;

impl FormatTime for LocalTimer {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> std::fmt::Result {
        let now = chrono::Local::now();
        write!(w, "{}", now.format("%Y-%m-%dT%H:%M:%S%.3f%:z"))
    }
}

const LOG_PATH: &str = "/var/log/faceunlock.log";
const MAX_LOG_BYTES: u64 = 1_048_576; // 1 MB

struct MaxSizeWriter {
    file: Mutex<File>,
    max_bytes: u64,
}

impl MaxSizeWriter {
    fn new(path: &str, max_bytes: u64) -> Self {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("Failed to open log file");
        Self { file: Mutex::new(file), max_bytes }
    }
}

impl io::Write for MaxSizeWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let file = self.file.get_mut().unwrap();
        let len = file.metadata()?.len();
        if len + buf.len() as u64 > self.max_bytes {
            file.set_len(0)?;
        }
        file.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.get_mut().unwrap().flush()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::load()?;

    let writer = MaxSizeWriter::new(LOG_PATH, MAX_LOG_BYTES);
    let (non_blocking, _guard) = non_blocking(writer);

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.log_level)))
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false)
                .with_timer(LocalTimer),
        )
        .init();

    tracing::info!("Starting faceunlockd");
    tracing::info!("Config loaded from disk");

    let _detector = detect::Detector::new(&config.models.detector, config.auth.detection_threshold)?;
    tracing::info!("Detector model loaded: {}", config.models.detector);

    let _recognizer = recognize::Recognizer::new(&config.models.recognizer)?;
    tracing::info!("Recognizer model loaded: {}", config.models.recognizer);

    drop(_detector);
    drop(_recognizer);

    let server = socket::SocketServer::new(config);

    let listener = server.bind().await?;

    tokio::select!(
        result = server.run(listener) => {
            if let Err(e) = result {
                tracing::error!("Server error: {}", e);
            }
        }
        _ = shutdown_signal() => {
            tracing::info!("Shutdown signal received");
        }
    );

    tracing::info!("faceunlockd stopped");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    let mut sigterm =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();

    tokio::select! {
        _ = ctrl_c => {},
        _ = sigterm.recv() => {},
    }
}
