# Configuration

The `oakd record` command can load a YAML config file via `--config`. The file is merged with defaults; any keys you omit fall back to defaults.

Recordings are written under `output.base_path/data/` (a `data` subfolder is created automatically).

## Supported Keys (Recording)

### `camera`

- `camera.rgb_resolution` (`[int, int]`, default: `[1280, 800]`): RGB preview size as `[width, height]`. Depth frames are resized to match for writing.
- `camera.fps` (`int`, default: `30`): Recording FPS.
- `camera.recording_time` (`int`, default: `10`): Recording duration in seconds.

### `output`

- `output.base_path` (`str`, default: `./output`): Output parent directory. Files are saved under `output.base_path/data/`.
- `output.rgb_filename` (`str`, default: `rgb_video.mp4`): RGB output filename (written inside `output.base_path/data/`).
- `output.depth_filename` (`str`, default: `depth_video.mp4`): Depth output filename (written inside `output.base_path/data/`).

### `depth`

- `depth.colormap` (`str`, default: `COLORMAP_JET`): OpenCV colormap constant name on `cv2` (e.g. `COLORMAP_JET`, `COLORMAP_TURBO`).
- `depth.normalize` (`bool`, default: `true`): Normalize depth frame before coloring.
- `depth.equalize_hist` (`bool`, default: `true`): Histogram equalization on depth frame before coloring.

## Example (Annotated)

```yaml
camera:
  # RGB preview size as [width, height]; depth frames are resized to match for writing.
  rgb_resolution: [1280, 800]
  # Frames per second.
  fps: 30
  # Recording duration in seconds.
  recording_time: 10

output:
  # Parent directory for outputs; recordings are saved to: <base_path>/data/
  base_path: "./output"
  # Output filenames written inside <base_path>/data/
  rgb_filename: "rgb_video.mp4"
  depth_filename: "depth_video.mp4"

depth:
  # OpenCV colormap constant name on cv2 (e.g. COLORMAP_JET, COLORMAP_TURBO).
  colormap: "COLORMAP_JET"
  # Normalize depth to 0-255 before coloring.
  normalize: true
  # Apply histogram equalization before coloring.
  equalize_hist: true
```

## Notes

- `oakd detect` is currently configured via CLI flags (not a YAML config file).
- Extra keys in the YAML file are ignored by `oakd record` unless they match the supported keys above.
