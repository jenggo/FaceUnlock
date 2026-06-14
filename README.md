# faceunlock — IR Face Authentication for Linux

A privacy-respecting face authentication system for Linux, leveraging IR cameras for liveness detection and PAM integration.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   PAM       │────▶│  Unix Socket │────▶│  faceunlockd │
│  Module     │     │  (JSON)      │     │  (Rust)      │
│  (C)        │     │              │     │              │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                                    ┌───────────┴───────────┐
                                    │                       │
                              ┌─────▼─────┐          ┌──────▼──────┐
                              │  V4L2 IR  │          │  ONNX       │
                              │  Camera   │          │  Runtime    │
                              └───────────┘          └─────────────┘
```

## Installation

### Prerequisites

- Arch Linux (or similar)
- Kernel ≥ 6.18
- IR camera (V4L2 compatible)
- Rust toolchain
- `libpam-dev` (Debian) / `pam` (Arch)
- `libjson-c-dev` (Debian) / `json-c` (Arch)

### Build

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone <repo-url>
cd faceunlock
make all

# Install (creates faceunlock group, adds to video group)
sudo make install

# Add yourself to the faceunlock group (log out/in after)
sudo usermod -aG faceunlock $(whoami)

# Download models (see below)
sudo bash scripts/download-models.sh
```

## IR Camera Identification

Use `v4l2-ctl` to identify your IR camera:

```bash
# List all video devices
v4l2-ctl --list-devices

# Check formats for a specific device
v4l2-ctl --device=/dev/video2 --list-formats-ext
```

Look for a device supporting `GREY` (8-bit greyscale) format — this is typically the IR camera.

Example output:
```
[0]: 'GREY' (8-bit Greyscale)
    Size: Discrete 640x360
        Interval: Discrete 0.067s (15.000 fps)
```

## Configuration

Edit `/etc/faceunlock/faceunlock.toml`:

```toml
[camera]
device = "/dev/video2"    # Your IR camera device
width = 640
height = 360

[models]
detector = "/usr/share/faceunlock/models/scrfd_10g_bnkps.onnx"
recognizer = "/usr/share/faceunlock/models/adaface_ir50.onnx"

[auth]
similarity_threshold = 0.40
max_frames = 10
liveness_threshold = 0.15

[enrollment]
store_path = "/var/lib/faceunlock/embeddings"
```

## Enrollment

```bash
# Enroll a new user
sudo faceunlock add jenggo

# List enrolled users
faceunlock list

# Test authentication (one-shot)
faceunlock test jenggo --verbose

# Test with live GUI view
faceunlock test jenggo --view

# Remove enrollment
sudo faceunlock remove jenggo
```

## GUI View

The `--view` flag opens a live camera window with face detection and recognition overlay:

```bash
faceunlock-enroll test jenggo --view
```

The window displays:
- **Green bounding box** — face detected and recognized (PASS)
- **Red bounding box** — face detected but not recognized (FAIL)
- **Red dots** — 5 facial landmarks
- **Label** — `PASS XX%` or `FAIL XX%` (recognition confidence)
- **Bottom text** — `det:XX%` (detection confidence)

Press **ESC** to close the window.

The daemon streams frames over the Unix socket — no need to stop the daemon. The camera and models stay loaded for PAM authentication.

## PAM Configuration

Add to `/etc/pam.d/sudo` (before `pam_unix.so`):

```
auth sufficient pam_faceunlock.so
```

For login, sddm, or swaylock, add the same line to their respective PAM config files.

## Security Notes

- Use `sufficient` not `required` in PAM config to allow password fallback
- Embeddings are stored as mathematical vectors (never images)
- Daemon runs as root for V4L2 access; socket permissions (`root:faceunlock 660`) restrict which users can authenticate
- Add users to the `faceunlock` group to allow face authentication: `sudo usermod -aG faceunlock <user>`

## Liveness Detection

The system uses passive IR reflectance analysis:
- Human skin: moderate mean (80-180), non-zero standard deviation
- Screens: very high mean (near 255)
- Printed photos: very low standard deviation

Tune `liveness_threshold` in config based on your camera's characteristics.
