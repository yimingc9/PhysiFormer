from __future__ import annotations

import os
import re
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


def _timing_summary(log: str) -> str:
    values = dict(re.findall(r"\[timing\]\s+([A-Za-z0-9_\[\]\.]+)=([0-9.]+)", log))
    hardware = re.findall(r"\[hardware\]\s+(.+)", log)
    lines = []
    if hardware:
        lines.append("Hardware: " + hardware[-1])
    if "sample[0].gen[0].inference_model_generate_s" in values:
        lines.append(f"Model inference: {values['sample[0].gen[0].inference_model_generate_s']} s")
    elif "sample[0] gen[0] inference_model_generate_s" in log:
        match = re.search(r"\[timing\]\s+sample\[0\]\s+gen\[0\]\s+inference_model_generate_s=([0-9.]+)", log)
        if match:
            lines.append(f"Model inference: {match.group(1)} s")
    fields = [
        ("checkpoint_load_s", "Checkpoint load"),
        ("model_setup_s", "Model setup"),
        ("sample[0].setup_s", "Input setup"),
        ("sample[0].gen[0].postprocess_save_npz_s", "Postprocess/save"),
        ("sample[0].gen[0].render_encode_s", "Render/encode"),
        ("engine_total_wall_s", "Engine total"),
        ("gradio_subprocess_wall_s", "Gradio subprocess wall"),
    ]
    for key, label in fields:
        if key in values:
            lines.append(f"{label}: {values[key]} s")
    if lines:
        return "\n".join(lines)
    return (
        "No timing markers were found in the inference output.\n"
        "The Space is likely running an older copy of src/physformer/scripts/infer.py; "
        "push these changes and restart/rebuild the Space."
    )


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
        "--verbose",
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
def run_inference(example: str, sampling_steps: int, setup_log: str) -> tuple[str | None, str, str]:
    if not CKPT_PATH.is_file():
        log = setup_log + "\nCheckpoint is missing; click Run again after the download finishes."
        return None, "Checkpoint missing.", log

    start_time = time.time()
    subprocess_t0 = time.perf_counter()
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
    subprocess_s = time.perf_counter() - subprocess_t0
    log = (
        setup_log
        + "\n\n$ "
        + " ".join(cmd)
        + "\n"
        + proc.stdout
        + f"\n[timing] gradio_subprocess_wall_s={subprocess_s:.3f}"
    )
    summary = _timing_summary(log)
    mp4 = _latest_mp4_since(start_time)
    if proc.returncode != 0:
        fail_log = log + f"\nInference failed with exit code {proc.returncode}."
        return None, _timing_summary(fail_log), _tail(fail_log)
    if mp4 is None:
        missing_log = log + "\nInference finished, but no inference.mp4 was found."
        return None, _timing_summary(missing_log), _tail(missing_log)
    final_log = log + f"\nGenerated video: {mp4.relative_to(ROOT)}"
    return str(mp4), summary, _tail(final_log)


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
    timing = gr.Textbox(label="Timing summary", lines=8, value="Run a rollout to see timing.")
    log = gr.Textbox(label="Log", lines=18)

    run_button.click(ensure_checkpoint, outputs=log).then(
        run_inference,
        inputs=[example, sampling_steps, log],
        outputs=[video, timing, log],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=8).launch()
