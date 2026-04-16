import os
import gc
import re
import time
import torch
import gradio as gr
import subprocess
from PIL import Image

torch.set_float32_matmul_precision("high")

try:
    from diffusers import QwenImageEditPlusPipeline as QwenEditPipeline
except Exception:
    try:
        from diffusers import QwenImageEditPipeline as QwenEditPipeline
    except Exception:
        QwenEditPipeline = None


# =========================
# CONFIG
# =========================
HF_TOKEN = os.environ.get("HF_TOKEN")

DEFAULT_MODEL = "Qwen/Qwen-Image-Edit-2511"
DEFAULT_LORA = "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA"

DEFAULT_PROMPT = "front side view, elevated shot, move the camera just slightly closer to the object"
DEFAULT_NEGATIVE_PROMPT = (
    "distorted structure, changed object identity and lighting, extra parts, "
    "deformed hull, broken geometry, duplicated parts"
)
DEFAULT_STEPS = 20
DEFAULT_SEED = 91027
DEFAULT_LORA_SCALE = 0.9
DEFAULT_TRUE_CFG_SCALE = 2.0
DEFAULT_USE_LORA = True
DEFAULT_PORT = 7860

from pathlib import Path


PROJECT_ROOT = Path("/content/drive/MyDrive/fyp")
MULTI_VIEW_OUT_DIR = PROJECT_ROOT / "multi_view_output"
TMP_DIR = PROJECT_ROOT / "tmp"
for p in [MULTI_VIEW_OUT_DIR, TMP_DIR]:
    p.mkdir(parents=True, exist_ok=True)

pipe = None
loaded_model_name = None
loaded_lora_name = None


# =========================
# HELPERS
# =========================
def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def unload_pipe():
    global pipe, loaded_model_name, loaded_lora_name
    pipe = None
    loaded_model_name = None
    loaded_lora_name = None
    cuda_cleanup()


def clean_text(x: str) -> str:
    x = (x or "").strip()
    x = x.replace("\n", " ")
    while "  " in x:
        x = x.replace("  ", " ")
    return x


def init_pipe(model_name: str, use_lora: bool, lora_name: str, lora_scale: float):
    global pipe, loaded_model_name, loaded_lora_name

    if QwenEditPipeline is None:
        raise RuntimeError(
            "Qwen Image Edit pipeline is not available in your diffusers install."
        )

    wanted_lora = lora_name if use_lora else ""

    cache_ok = (
        pipe is not None
        and loaded_model_name == model_name
        and loaded_lora_name == wanted_lora
    )
    if cache_ok:
        return pipe

    unload_pipe()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[INFO] torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"[INFO] Selected device = {device}")

    p = QwenEditPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        token=HF_TOKEN if HF_TOKEN else None,
    )

    p.set_progress_bar_config(disable=True)

    if use_lora and lora_name:
        p.load_lora_weights(
            lora_name,
            token=HF_TOKEN if HF_TOKEN else None,
        )

        try:
            if hasattr(p, "set_adapters"):
                p.set_adapters(["default_0"], adapter_weights=[float(lora_scale)])
        except Exception:
            pass

    # FULL GPU ONLY
    if device == "cuda":
        p = p.to("cuda")
        print("[INFO] Pipeline moved to FULL GPU")
    else:
        p = p.to("cpu")
        print("[WARNING] CUDA not available, running on CPU")

    try:
        p.enable_attention_slicing()
    except Exception:
        pass

    pipe = p
    loaded_model_name = model_name
    loaded_lora_name = wanted_lora
    return pipe


def run_edit(
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    seed: int,
    use_lora: bool,
    lora_scale: float,
    true_cfg_scale: float,
):
    if image is None:
        return None, "Please upload an image first."

    prompt = clean_text(prompt)
    negative_prompt = clean_text(negative_prompt)

    if not prompt:
        return None, "Prompt is empty."

    try:
        p = init_pipe(
            model_name=DEFAULT_MODEL,
            use_lora=bool(use_lora),
            lora_name=DEFAULT_LORA,
            lora_scale=float(lora_scale),
        )

        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        generator.manual_seed(int(seed))

        kwargs = {
            "image": image.convert("RGB"),
            "prompt": prompt,
            "generator": generator,
            "num_inference_steps": int(steps),
        }

        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
            kwargs["true_cfg_scale"] = float(true_cfg_scale)

        start_time = time.time()
        result = p(**kwargs)
        out = result.images[0]
        elapsed = time.time() - start_time

        used_cfg = (
            f"Enabled true_cfg_scale={true_cfg_scale}"
            if negative_prompt else
            "CFG disabled (no negative prompt)"
        )
        used_lora = f"LoRA enabled, scale={lora_scale}" if use_lora else "LoRA disabled"

        saved_path = save_pil_image(out, MULTI_VIEW_OUT_DIR, "multi_view_output")

        log_text = (
            f"Prompt: {prompt}\n"
            f"Negative prompt: {negative_prompt if negative_prompt else '[None]'}\n"
            f"{used_cfg}\n"
            f"{used_lora}\n"
            f"Steps: {steps}\n"
            f"Seed: {seed}\n"
            f"Time: {elapsed:.2f} seconds\n"
            f"Saved to: {saved_path}"
        )

        return out, log_text

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def unload_and_message():
    unload_pipe()
    return "Model unloaded."



def save_pil_image(img: Image.Image, out_dir: Path, stem_prefix: str, ext: str = ".png") -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{stem_prefix}_{int(time.time())}"
    out_path = out_dir / f"{stem}{ext}"
    img.save(out_path)
    return str(out_path)

def start_cloudflared(port=7860):
    log_path = "/tmp/cloudflared.log"

    subprocess.run(
        "pkill -f cloudflared",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    cmd = f"cloudflared tunnel --url http://127.0.0.1:{port} > {log_path} 2>&1 &"
    subprocess.Popen(cmd, shell=True)

    pattern = r"https://[-a-zA-Z0-9]+\.trycloudflare\.com"

    for _ in range(30):
        time.sleep(1)
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            match = re.search(pattern, text)
            if match:
                return match.group(0)

    return None


# =========================
# UI
# =========================
with gr.Blocks(title="Qwen Image Edit Multi-View Test") as demo:
    gr.Markdown("## Qwen Image Edit + Multi-View LoRA Test")
    gr.Markdown("Upload an image, edit the prompt, and test viewpoint changes first.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image", height=320)

            prompt_box = gr.Textbox(
                label="Edit Prompt",
                value=DEFAULT_PROMPT,
                lines=3,
            )

            negative_prompt_box = gr.Textbox(
                label="Negative Prompt",
                value=DEFAULT_NEGATIVE_PROMPT,
                lines=3,
            )

            use_lora_box = gr.Checkbox(
                label="Use Multi-View LoRA",
                value=DEFAULT_USE_LORA,
            )

            lora_scale_slider = gr.Slider(
                minimum=0.1,
                maximum=1.2,
                value=DEFAULT_LORA_SCALE,
                step=0.05,
                label="LoRA Scale",
            )

            true_cfg_scale_slider = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=DEFAULT_TRUE_CFG_SCALE,
                step=0.1,
                label="True CFG Scale",
            )

            steps_slider = gr.Slider(
                minimum=10,
                maximum=80,
                value=DEFAULT_STEPS,
                step=1,
                label="Steps",
            )

            seed_slider = gr.Slider(
                minimum=0,
                maximum=1000000,
                value=DEFAULT_SEED,
                step=1,
                label="Seed",
            )

            run_btn = gr.Button("Generate View", variant="primary")
            unload_btn = gr.Button("Unload Model")

        with gr.Column():
            output_image = gr.Image(type="pil", label="Output Image", height=512)
            log_box = gr.Textbox(label="Log", lines=10, interactive=False)

    run_btn.click(
        fn=run_edit,
        inputs=[
            input_image,
            prompt_box,
            negative_prompt_box,
            steps_slider,
            seed_slider,
            use_lora_box,
            lora_scale_slider,
            true_cfg_scale_slider,
        ],
        outputs=[output_image, log_box],
    )

    unload_btn.click(
        fn=unload_and_message,
        inputs=[],
        outputs=[log_box],
    )


if __name__ == "__main__":
    print(f"[INFO] Starting local app on http://0.0.0.0:{DEFAULT_PORT}")

    public_url = start_cloudflared(DEFAULT_PORT)
    if public_url:
        print(f"[INFO] Public URL: {public_url}")
    else:
        print("[WARNING] cloudflared tunnel URL not found.")
        print("Run this to check manually:")
        print("grep -o 'https://[-a-zA-Z0-9]*\\.trycloudflare\\.com' /tmp/cloudflared.log")

    demo.launch(
        server_name="0.0.0.0",
        server_port=DEFAULT_PORT,
        share=False,
        debug=True,
        show_error=True,
    )