import typer
from pathlib import Path
from typing import Optional
from loguru import logger
from rich.console import Console
from rich.panel import Panel
import depthai as dai
from src.core.recorder import OakDCamera
from src.core.detector import OakDObjectDetectionApp
from src.utils.config import ConfigManager
from src.utils.device import check_connection_status
from src.utils.visualization import show_video_stream

app = typer.Typer()
console = Console()

@app.command()
def check_connection():
    """
    Check connection to OAK-D device and print device details.
    """
    console.print(Panel.fit("OAK-D Connection Check", style="bold cyan"))
    
    try:
        info = check_connection_status()
        console.print("[bold green]Connected to device![/bold green]")
        console.print(f"Device name: {info['device_name']}")
        console.print(f"USB speed: {info['usb_speed']}")
        console.print(f"Connected cameras: {info['connected_cameras']}")
        console.print(f"Available stereo pairs: {info['stereo_pairs']}")
            
    except Exception as e:
        console.print(f"[bold red]Failed to connect to device:[/bold red] {e}")
        logger.exception("Connection check failed")
        raise typer.Exit(code=1)

@app.command()
def show_video():
    """
    Stream and display RGB and Depth video from OAK-D camera.
    """
    console.print(Panel.fit("OAK-D Video Stream", style="bold blue"))
    
    try:
        show_video_stream()
    except Exception as e:
        console.print(f"[bold red]Error during video streaming:[/bold red] {e}")
        logger.exception("Video streaming failed")
        raise typer.Exit(code=1)

@app.command()
def record(
    output_dir: Path = typer.Option(
        Path("./output"),
        "--output-dir", "-o",
        help="Directory to save recordings"
    ),
    duration: int = typer.Option(
        10,
        "--duration", "-d",
        help="Recording duration in seconds"
    ),
    fps: int = typer.Option(
        30,
        "--fps", "-f",
        help="Frames per second"
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to YAML config file. Overrides other options if provided."
    ),
) -> None:
    """
    Record RGB and Depth video from OAK-D camera.
    """
    console.print(Panel.fit("OAK-D Recorder", style="bold magenta"))

    try:
        if config_file:
            if not config_file.exists():
                console.print(f"[red]Config file not found: {config_file}[/red]")
                raise typer.Exit(code=1)
            config = ConfigManager.load_config(str(config_file))
        else:
            config = ConfigManager.create_config_from_args(output_dir, duration, fps)

        logger.info(f"Initializing camera with config: {config}")
        recorder = OakDCamera(config)
        
        with console.status(f"[bold green]Recording for {config['camera']['recording_time']} seconds..."):
            recorder.record()
            
        console.print("[bold green]Recording finished successfully![/bold green]")
        console.print(f"Files saved to: {Path(config['output']['base_path']).resolve()}")

    except Exception as e:
        console.print(f"[bold red]Error during recording:[/bold red] {e}")
        logger.exception("Recording failed")
        raise typer.Exit(code=1)

@app.command()
def detect(
    output_dir: Path = typer.Option(
        Path("./output"),
        "--output-dir", "-o",
        help="Directory to save video (if enabled)"
    ),
    confidence: float = typer.Option(
        0.5,
        "--confidence", "-c",
        help="Confidence threshold for detection"
    ),
    save_video: bool = typer.Option(
        False,
        "--save-video", "-s",
        help="Save video of the detection"
    ),
) -> None:
    """
    Run object detection on OAK-D camera.
    """
    console.print(Panel.fit("OAK-D Object Detection", style="bold green"))
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct output path for video if saving is enabled
    video_path = output_dir / "object_detection.mp4" if save_video else None

    try:
        logger.info(f"Starting detection with confidence {confidence}")
        app = OakDObjectDetectionApp(
            confidence_threshold=confidence,
            save_video=save_video,
            output_path=str(video_path) if video_path else None
        )
        app.run()
    except Exception as e:
        console.print(f"[bold red]Error during detection:[/bold red] {e}")
        logger.exception("Detection failed")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
