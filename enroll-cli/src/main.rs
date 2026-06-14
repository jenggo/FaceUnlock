mod view;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;

const SOCKET_PATH: &str = "/run/faceunlockd/auth.sock";

#[derive(Parser)]
#[command(name = "faceunlock", about = "Face authentication CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Enroll a user's face
    Add {
        /// Username to enroll
        user: String,
    },
    /// List enrolled users
    List {
        /// Show specific user only
        user: Option<String>,
    },
    /// Remove a user's enrollment
    Remove {
        /// Username to remove
        user: String,
    },
    /// Test face authentication
    Test {
        /// Username to test
        user: String,
        /// Show detailed output
        #[arg(short, long)]
        verbose: bool,
        /// Open GUI window with live camera feed and face detection overlay
        #[arg(long)]
        view: bool,
    },
    /// Discover camera devices
    Discover,
}

fn send_request(action: &str, user: Option<&str>) -> Result<String> {
    let mut stream =
        UnixStream::connect(SOCKET_PATH).context("Failed to connect to faceunlockd daemon")?;

    let mut request = serde_json::json!({
        "action": action,
    });

    if let Some(u) = user {
        request["user"] = serde_json::json!(u);
    }

    let json_str = serde_json::to_string(&request)?;
    stream.write_all(json_str.as_bytes())?;
    stream.write_all(b"\n")?;

    let mut response = String::new();
    let reader = BufReader::new(&stream);
    for line in reader.lines() {
        let line = line?;
        if !line.is_empty() {
            response = line;
            break;
        }
    }

    Ok(response)
}

fn cmd_add(user: &str) -> Result<()> {
    println!("Enrolling user: {}", user);
    println!("Please look at the IR camera...");

    let response = send_request("enroll", Some(user))?;
    let resp: serde_json::Value = serde_json::from_str(&response)?;

    if resp["result"] == "ok" {
        println!(
            "Enrollment successful: {}",
            resp["reason"].as_str().unwrap_or("done")
        );
        Ok(())
    } else {
        anyhow::bail!(
            "Enrollment failed: {}",
            resp["reason"].as_str().unwrap_or("unknown error")
        );
    }
}

fn cmd_list(user: Option<&str>) -> Result<()> {
    let response = send_request("status", user)?;
    let resp: serde_json::Value = serde_json::from_str(&response)?;

    if resp["result"] != "ok" {
        anyhow::bail!(
            "Failed to get status: {}",
            resp["reason"].as_str().unwrap_or("unknown error")
        );
    }

    let users = resp["users"].as_array().context("Invalid response format")?;

    if users.is_empty() {
        println!("No enrolled users.");
        return Ok(());
    }

    println!("{:<20} {}", "Username", "Embeddings");
    println!("{}", "-".repeat(32));

    for u in users {
        let username = u["username"].as_str().unwrap_or("unknown");
        let count = u["embedding_count"].as_u64().unwrap_or(0);
        println!("{:<20} {}", username, count);
    }

    Ok(())
}

fn cmd_remove(user: &str) -> Result<()> {
    let response = send_request("enroll_clear", Some(user))?;
    let resp: serde_json::Value = serde_json::from_str(&response)?;

    if resp["result"] == "ok" {
        println!("Removed enrollment for user: {}", user);
        Ok(())
    } else {
        anyhow::bail!(
            "Failed to remove enrollment: {}",
            resp["reason"].as_str().unwrap_or("unknown error")
        );
    }
}

fn cmd_test(user: &str, verbose: bool) -> Result<()> {
    println!("Testing authentication for user: {}", user);

    let response = send_request("authenticate", Some(user))?;
    let resp: serde_json::Value = serde_json::from_str(&response)?;

    if verbose {
        println!("Response: {}", serde_json::to_string_pretty(&resp)?);
    }

    if resp["result"] == "ok" {
        println!("Authentication: PASS");
        if let Some(score) = resp["score"].as_f64() {
            println!("Similarity score: {:.4}", score);
        }
        if let Some(timing) = resp["timing"].as_object() {
            let detect = timing.get("detect_ms").and_then(|v| v.as_u64()).unwrap_or(0);
            let recognize = timing.get("recognize_ms").and_then(|v| v.as_u64()).unwrap_or(0);
            let total = timing.get("total_ms").and_then(|v| v.as_u64()).unwrap_or(0);
            println!("Timing: detect={}ms recognize={}ms total={}ms", detect, recognize, total);
        }
        Ok(())
    } else {
        println!("Authentication: FAIL");
        if let Some(reason) = resp["reason"].as_str() {
            println!("Reason: {}", reason);
        }
        if let Some(score) = resp["score"].as_f64() {
            println!("Similarity score: {:.4}", score);
        }
        if let Some(timing) = resp["timing"].as_object() {
            let detect = timing.get("detect_ms").and_then(|v| v.as_u64()).unwrap_or(0);
            let recognize = timing.get("recognize_ms").and_then(|v| v.as_u64()).unwrap_or(0);
            let total = timing.get("total_ms").and_then(|v| v.as_u64()).unwrap_or(0);
            println!("Timing: detect={}ms recognize={}ms total={}ms", detect, recognize, total);
        }
        anyhow::bail!("Authentication failed")
    }
}

fn cmd_discover() -> Result<()> {
    println!("Discovering camera devices...\n");

    let entries: Vec<_> = std::fs::read_dir("/dev")
        .context("Failed to read /dev")?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|n| n.starts_with("video"))
                .unwrap_or(false)
        })
        .collect();

    if entries.is_empty() {
        println!("No video devices found.");
        return Ok(());
    }

    for entry in &entries {
        let path = entry.path();
        let device = path.to_str().unwrap_or("unknown");
        println!("Device: {}", device);

        let output = std::process::Command::new("v4l2-ctl")
            .args(["--device", device, "--list-formats"])
            .output();

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                for line in stdout.lines() {
                    if line.contains("['") || line.contains("[") {
                        println!("  {}", line.trim());
                    }
                }
            }
            Err(_) => println!("  (could not query formats)"),
        }
        println!();
    }

    Ok(())
}

fn cmd_view(user: &str) -> Result<()> {
    view::run_view(SOCKET_PATH, user)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Add { user } => cmd_add(&user),
        Commands::List { user } => cmd_list(user.as_deref()),
        Commands::Remove { user } => cmd_remove(&user),
        Commands::Test { user, verbose, view } => {
            if view {
                cmd_view(&user)
            } else {
                cmd_test(&user, verbose)
            }
        }
        Commands::Discover => cmd_discover(),
    }
}
