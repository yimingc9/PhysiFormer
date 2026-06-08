from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr
import spaces


ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from official_demo_inference.paths import default_checkpoint_path
from official_demo_inference.paths import ensure_default_checkpoint


CKPT_PATH = default_checkpoint_path()


def _tail(text: str, max_chars: int = 18000) -> str:
    if len(text) <= max_chars:
        return text
    return "[log truncated]\n" + text[-max_chars:]


def ensure_checkpoint() -> str:
    ckpt_path, message = ensure_default_checkpoint()
    if ckpt_path.resolve() != default_checkpoint_path().resolve():
        return f"Unexpected checkpoint path: {ckpt_path}"
    return message


def _command_for_example(example: str, sampling_steps: int) -> list[str]:
    common = [
        sys.executable,
        "run_official_demo_inference.py",
        "--include",
        "all",
        "--generations",
        "1",
        "--max-samples",
        "1",
        "--num-sampling-steps",
        str(int(sampling_steps)),
        "--checkpoint",
        str(CKPT_PATH),
        "--device",
        "cuda",
        "--amp",
        os.environ.get("PHYSFORMER_AMP", "bf16"),
        "--overwrite",
        "--save-mp4",
    ]
    if example == "OOD mixed materials":
        return common + [
            "--demo-root",
            "ood_examples",
            "--elastic",
            "horse",
            "--elastic",
            "fish",
            "--elastic",
            "bunny",
            "--rigid",
            "cow",
            "--rigid",
            "teapot",
        ]
    if example == "In-distribution rigid":
        return common + ["--demo-root", "indistri_examples/rigid", "--rigid", "all"]
    if example == "In-distribution elastic":
        return common + ["--demo-root", "indistri_examples/soft", "--elastic", "all"]
    raise ValueError(f"Unknown example: {example}")


def _latest_mp4_since(start_time: float) -> Path | None:
    candidates = sorted(
        [
            path
            for path in ROOT.glob("**/inference.mp4")
            if ".inference_work" not in path.parts and path.stat().st_mtime >= start_time - 1.0
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


@spaces.GPU(duration=300)
def run_inference(example: str, sampling_steps: int, setup_log: str) -> tuple[str | None, str]:
    if not CKPT_PATH.is_file():
        return None, setup_log + "\nCheckpoint is missing; click Run again after the download finishes."

    start_time = time.time()
    cmd = _command_for_example(example, int(sampling_steps))
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".inference_work" / "matplotlib"))

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=900,
    )
    log = setup_log + "\n\n$ " + " ".join(cmd) + "\n" + proc.stdout
    mp4 = _latest_mp4_since(start_time)
    if proc.returncode != 0:
        return None, _tail(log + f"\nInference failed with exit code {proc.returncode}.")
    if mp4 is None:
        return None, _tail(log + "\nInference finished, but no inference.mp4 was found.")
    return str(mp4), _tail(log + f"\nGenerated video: {mp4.relative_to(ROOT)}")


with gr.Blocks(title="PhysFormer") as demo:
    gr.Markdown(
        """
        # PhysFormer Minimal ZeroGPU Demo

        Runs one small PhysFormer rollout and returns the rendered mesh trajectory.
        """
    )
    with gr.Row():
        example = gr.Dropdown(
            choices=["OOD mixed materials", "In-distribution rigid", "In-distribution elastic"],
            value="OOD mixed materials",
            label="Example",
        )
        sampling_steps = gr.Slider(5, 50, value=5, step=1, label="Denoising steps")
    run_button = gr.Button("Run one rollout", variant="primary")
    video = gr.Video(label="Generated rollout")
    log = gr.Textbox(label="Log", lines=18)

    run_button.click(ensure_checkpoint, outputs=log).then(
        run_inference,
        inputs=[example, sampling_steps, log],
        outputs=[video, log],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=8).launch()
