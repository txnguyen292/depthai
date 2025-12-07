# OAK-D Pro Project

A modular Python application for interacting with OAK-D cameras, featuring video recording, object detection, and device management.

## Prerequisites

- **Hardware**: OAK-D Series Camera (e.g., OAK-D Pro, OAK-D Lite)
- **Software**:
    - Python 3.12 or higher
    - [uv](https://github.com/astral-sh/uv) (recommended for dependency management)

## Device Setup (Important)

Before using the library, you must ensure your system is configured to communicate with the OAK-D camera.

### Linux (udev rules)
On Linux, you must configure `udev` rules to allow access to the USB device. Run the following commands:

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### macOS
It is recommended to run the official dependency installer script to ensure all system libraries are present:

```bash
curl -fL https://docs.luxonis.com/install_dependencies.sh | bash
```

### Windows
Windows 10/11 usually works out of the box. If you encounter issues, try using the [DepthAI Windows Installer](https://github.com/luxonis/depthai/releases).

## Installation

1.  **Install `uv`** (if not already installed):
    ```bash
    pip install pipx
    pipx install uv
    ```

2.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd depthai
    ```

3.  **Install dependencies**:
    ```bash
    uv sync
    ```

## Usage

The project uses a CLI (Command Line Interface) powered by `typer`. You can run commands using `uv run oakd` or `uv run python src/cli.py`.

### 1. Check Device Connection
Verify that your OAK-D camera is connected and recognized.

```bash
uv run oakd check-connection
```

### 2. Show Video Stream
Preview the RGB and Depth streams in a window without recording.

```bash
uv run oakd show-video
```

### 3. Record Video
Record synchronized RGB and Depth video streams.

```bash
# Default recording (10s, 30fps, saves to ./output)
uv run oakd record

# Custom duration and output directory
uv run oakd record --duration 20 --output-dir ./my_recordings

# Using a configuration file
uv run oakd record --config config.yml
```

**Options:**
- `-o, --output-dir`: Directory to save recordings (default: `./output`)
- `-d, --duration`: Recording duration in seconds (default: `10`)
- `-f, --fps`: Frames per second (default: `30`)
- `-c, --config`: Path to YAML config file (overrides other options)

### 4. Object Detection
Run MobileNet object detection on the camera stream.

```bash
# Run detection
uv run oakd detect

# Run detection and save video
uv run oakd detect --save-video --output-dir ./detections
```

**Options:**
- `-o, --output-dir`: Directory to save video if enabled (default: `./output`)
- `-c, --confidence`: Confidence threshold (default: `0.5`)
- `-s, --save-video`: Save video of the detection (default: `False`)

## Configuration

You can configure the application using `config.yml`. This file allows you to set default parameters for the camera, output paths, and depth processing.

Example `config.yml`:
```yaml
camera:
  rgb_resolution: [1920, 1080]
  fps: 30
  recording_time: 10

output:
  base_path: "data"
  rgb_filename: "rgb_video.mp4"
  depth_filename: "depth_video.mp4"

depth:
  colormap: "COLORMAP_JET"
  normalize: true
  equalize_hist: true
```

## Development

### Running Tests
This project uses `pytest` for unit testing.

```bash
uv run pytest
```

### Project Structure
- `src/cli.py`: Main entry point for the CLI.
- `src/core/`: Core logic for recording and detection.
- `src/utils/`: Utility functions for config, device management, and visualization.
- `tests/`: Unit tests.
