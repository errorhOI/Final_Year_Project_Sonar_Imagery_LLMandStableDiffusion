import os
import gc
import re
import time
import uuid
import subprocess
from pathlib import Path
from typing import Tuple, List, Optional

import torch
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import gradio as gr
import plotly.graph_objects as go

from ultralytics import YOLO
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

try:
    from depth_anything_3.api import DepthAnything3
except Exception:
    try:
        from depth_anything_3 import DepthAnything3
    except Exception:
        DepthAnything3 = None

torch.set_float32_matmul_precision("high")


# =========================
# APP CONFIG
# =========================
APP_TITLE = "Unified Sonar → 2D Enhance → 3D Point Cloud"
SHARE_DEFAULT = True
DEFAULT_PORT = int(os.environ.get("PORT", 7860))
CLOUDFLARED_LOG = "/tmp/cloudflared.log"

BASE_DIR = Path("/content/drive/MyDrive/fyp/ui_bridge")
MULTI_VIEW_DIR = BASE_DIR / "multi_view_output"
SDXL_FINAL_DIR = BASE_DIR / "sdxl_final_output"
TMP_DIR = BASE_DIR / "tmp"
DA3_EXPORT_DIR = BASE_DIR / "da3_exports"

for p in [MULTI_VIEW_DIR, SDXL_FINAL_DIR, TMP_DIR, DA3_EXPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

DEFAULT_YOLO_SEG_WEIGHTS = "/content/drive/MyDrive/fyp/yolo_weights/train3/best.pt"
HF_TOKEN = os.environ.get("HF_TOKEN")

DEFAULT_QWEN_VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_SD_MODEL = "RunDiffusion/Juggernaut-XL-v9"
DEFAULT_CONTROLNET = "/content/drive/MyDrive/fyp/fyp_saved_models/controlnet_sdxl_out"

DEFAULT_DA3_MODEL_PRESET = "da3nested-giant-large"

DEFAULT_PARAMS_SEED_QWEN = 700993
DEFAULT_QWEN_IMAGE_SIDE = 768
DEFAULT_QWEN_MAX_TOKENS = 140

DEFAULT_PARAMS_SEED_SDXL = 163272
DEFAULT_STRENGTH = 0.47
DEFAULT_STEPS = 40
DEFAULT_GUIDANCE = 7
DEFAULT_TARGET_LONG_SIDE = 1024
DEFAULT_CN_SCALE = 1.31
DEFAULT_SCHEDULER = "DPM++ 2M Karras"

YOLO_CONF_MAIN_DEFAULT = 0.25
TARGET_SIZE_DEFAULT = 1024

CUTOUT_BG_RGB = (18, 28, 40)
APPLY_SONAR_PREPROCESS = True
DEFAULT_PREPROCESS_MODE = "brighten"

DEFAULT_PC_MAX_POINTS = 30000
DEFAULT_PC_DEPTH_SCALE = 1.0
DEFAULT_PC_Z_EXAGGERATION = 1.2
DEFAULT_PC_CROP_RATIO = 1.0


# =========================
# QWEN PROMPT (UNCHANGED)
# =========================
DEFAULT_IDEA_TEXT = """
You are assisting with sonar-guided underwater image enhancement.
You will be given TWO sonar-derived images of the same underwater object:

Image A: a raw sonar image showing the full scene, including seabed context, shadows, and surrounding structure.
Image B: a segmented cutout showing the main object silhouette and major separated parts.

Goal:
Generate ONE Stable Diffusion XL img2img prompt that converts the object into a clearer, denoised, realistic underwater view.

Analysis requirements:
- First identify what the object most likely represents based only on visible shape and context.
- If the object is uncertain, use cautious wording such as possible wreck-like form.
- Use Image B to understand geometry, silhouette, and any separated or broken components.
- Use Image A to infer orientation, resting posture, and interaction with surrounding sediment.

Constraints:
- Preserve the same viewpoint, orientation, scale, and overall silhouette implied by the images.
- Do not invent unsupported parts, decorations, or restored components.
- Keep surfaces continuous, readable, and geometrically consistent.
- Infer materials, age, biological growth, or surface condition only when visually supported; otherwise keep them generic.
- Do not reference sonar colour schemes, scan artefacts, or false-colour effects.

Environment and appearance rules:
- Depict a realistic underwater photograph with natural light attenuation and subdued color.
- Allow parts of the object to visually merge with darker seabed tones where contact or sediment accumulation is implied.
- Keep seabed subdued, low-contrast, and visually secondary to the object.
- Use restrained natural underwater color variation rather than dramatic cinematic color.
- Avoid fog, particles, caustics, motion blur, or dramatic lighting.
- Use even, neutral underwater illumination that preserves surface geometry and depth cues.

Output:
Return only a concise Stable Diffusion XL img2img prompt.
Do not include explanations, labels, comments, or multiple options.
""".strip()

QUALITY_SUFFIX = (
    "realistic underwater photo, clear but neutral visibility, "
    "soft even underwater lighting, continuous readable surfaces, "
    "preserve silhouette and viewpoint, target is primary focus, background subdued"
)

SMOOTH_SEABED_SUFFIX = (
    "smooth subdued seabed, compacted sediment, low contrast background"
)

COLOR_ENFORCE_SUFFIX = (
    "natural underwater blue-cyan attenuation, restrained realistic color variation, not grayscale"
)

NEGATIVE_PROMPT = (
    "blurry, soft focus, smudged, deformed, melted, extra parts, duplicated, "
    "particles, speckles, white dots, backscatter, marine snow, bubbles, plankton, fish, floating debris, "
    "rocks, gravel, pebbles, coral, rough seabed, uneven terrain, craters, "
    "cartoon, CGI, toy-like, plastic, studio background, seamless background, "
    "sonar colormap, thermal colormap, heatmap, false color, watermark, text, "
    "grayscale, monochrome, black and white, desaturated"
)

BIG_UI_CSS = """
:root{
  --bg0:#0b0f14; --bg1:#0f1520; --panel:#111827; --line:#1f2937;
  --text:#e5e7eb; --muted:#9ca3af;
}
.gradio-container{
  max-width: 100% !important; width:100% !important; margin:0 !important; padding:0 !important;
  background: radial-gradient(1200px 600px at 50% 40%, #1f2937 0%, var(--bg0) 55%, #000 100%) !important;
}
.panel{
  background: rgba(17,24,39,.86) !important;
  border:1px solid var(--line) !important;
  border-radius:16px !important;
}
.panel .wrap{padding:12px !important;}
.small{font-size:12px !important; color:var(--muted) !important;}
textarea, input{border-radius:12px !important;}
.btnrow button{border-radius:12px !important; font-weight:700 !important;}
#rightcol{
  background: rgba(17,24,39,.82) !important;
  border-radius:16px !important;
  padding:10px !important;
  border:1px solid var(--line) !important;
}
.gradio-image, .image-container{margin-bottom: 0 !important;}

#plot_panel {
  min-height: 820px !important;
}
#plot_panel .plot-container {
  min-height: 760px !important;
}
#plot_panel .js-plotly-plot {
  min-height: 760px !important;
}
"""


_qwen_model = None
_qwen_processor = None
_qwen_model_name = None
_yolo_main_model = None
_yolo_main_weights = None
_cn = None
_sdxl_cn_pipe = None
_sdxl_cn_key = None
_sdxl_cn_device = None
_da3_model = None
_da3_model_name = None


def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def unload_qwen():
    global _qwen_model, _qwen_processor, _qwen_model_name
    _qwen_model = None
    _qwen_processor = None
    _qwen_model_name = None
    cuda_cleanup()


def unload_yolo_main():
    global _yolo_main_model, _yolo_main_weights
    _yolo_main_model = None
    _yolo_main_weights = None
    cuda_cleanup()


def unload_sdxl_cn():
    global _cn, _sdxl_cn_pipe, _sdxl_cn_key, _sdxl_cn_device
    _cn = None
    _sdxl_cn_pipe = None
    _sdxl_cn_key = None
    _sdxl_cn_device = None
    cuda_cleanup()


def unload_da3():
    global _da3_model, _da3_model_name
    _da3_model = None
    _da3_model_name = None
    cuda_cleanup()


def clear_images_in_folder(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if not folder.exists():
        return
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            try:
                p.unlink()
            except Exception:
                pass


def clear_output_images():
    clear_images_in_folder(SDXL_FINAL_DIR)
    clear_images_in_folder(MULTI_VIEW_DIR)


def start_cloudflared(port: int = DEFAULT_PORT) -> Optional[str]:
    try:
        subprocess.run(
            "pkill -f cloudflared",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

    try:
        if os.path.exists(CLOUDFLARED_LOG):
            os.remove(CLOUDFLARED_LOG)
    except Exception:
        pass

    cmd = f"cloudflared tunnel --url http://127.0.0.1:{port} > {CLOUDFLARED_LOG} 2>&1 &"
    subprocess.Popen(cmd, shell=True)

    pattern = r"https://[-a-zA-Z0-9]+\.trycloudflare\.com"
    for _ in range(30):
        time.sleep(1)
        if os.path.exists(CLOUDFLARED_LOG):
            try:
                with open(CLOUDFLARED_LOG, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                match = re.search(pattern, text)
                if match:
                    return match.group(0)
            except Exception:
                pass
    return None


def to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def preprocess_sonar_raw(img: Image.Image) -> Image.Image:
    x = img.convert("RGB").convert("L")
    return x.convert("RGB")


def preprocess_sonar_brighten(
    img: Image.Image,
    brightness_factor: float = 1.20,
) -> Image.Image:
    x = img.convert("RGB").convert("L")
    x = ImageEnhance.Brightness(x).enhance(brightness_factor)
    return x.convert("RGB")


def preprocess_sonar(img: Image.Image, mode: str = DEFAULT_PREPROCESS_MODE) -> Image.Image:
    mode = (mode or DEFAULT_PREPROCESS_MODE).strip().lower()
    if mode == "raw":
        return preprocess_sonar_raw(img)
    elif mode == "brighten":
        return preprocess_sonar_brighten(img)
    else:
        raise ValueError(f"Unknown preprocess mode: {mode}")


def resize_long_side_exact(im: Image.Image, long_side: int) -> Image.Image:
    w, h = im.size
    s = long_side / float(max(w, h))
    nw, nh = int(round(w * s)), int(round(h * s))
    nw = max(64, (nw // 64) * 64)
    nh = max(64, (nh // 64) * 64)
    if (nw, nh) != (w, h):
        im = im.resize((nw, nh), Image.BICUBIC)
    return im


def composite_rgba_on_bg(rgba: Image.Image, bg_rgb=(0, 0, 0)) -> Image.Image:
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    bg = Image.new("RGB", rgba.size, bg_rgb).convert("RGBA")
    out = Image.alpha_composite(bg, rgba)
    return out.convert("RGB")


def cutout_preview_for_qwen(cutout_rgba: Image.Image) -> Image.Image:
    return composite_rgba_on_bg(cutout_rgba, bg_rgb=CUTOUT_BG_RGB)


def sonar_to_underwater_rgb(grayish_rgb: Image.Image) -> Image.Image:
    g = grayish_rgb.convert("L")
    g_np = np.array(g).astype(np.float32) / 255.0
    g_np = np.clip((g_np - 0.05) / 0.95, 0, 1)
    g_np = g_np ** 0.9
    deep = np.array([10, 25, 45], dtype=np.float32)
    mid = np.array([25, 85, 105], dtype=np.float32)
    high = np.array([190, 205, 190], dtype=np.float32)
    mid_mask = (g_np <= 0.6).astype(np.float32)[..., None]
    hi_mask = 1.0 - mid_mask
    t1 = np.clip(g_np / 0.6, 0, 1)[..., None]
    rgb1 = deep * (1 - t1) + mid * t1
    t2 = np.clip((g_np - 0.6) / 0.4, 0, 1)[..., None]
    rgb2 = mid * (1 - t2) + high * t2
    rgb = rgb1 * mid_mask + rgb2 * hi_mask
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def build_blue_textured_base(raw_pre: Image.Image, target_long_side: int) -> Image.Image:
    base = sonar_to_underwater_rgb(raw_pre)
    base = resize_long_side_exact(base, int(target_long_side))
    return base


def clean_qwen_prompt_text(text: str) -> str:
    text = (text or "").strip()
    bad_prefixes = ["output:", "prompt:", "here is the prompt:", "img2img prompt:"]
    lowered = text.lower()
    for bp in bad_prefixes:
        if lowered.startswith(bp):
            text = text[len(bp):].strip()
            break
    text = text.replace("\n", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def build_prompt(p: str) -> str:
    p = clean_qwen_prompt_text(p)
    parts = [p, QUALITY_SUFFIX, SMOOTH_SEABED_SUFFIX, COLOR_ENFORCE_SUFFIX]
    return ", ".join([x for x in parts if x])


def make_canny_control_image(img_rgb: Image.Image, low: int, high: int, out_size: Tuple[int, int]) -> Image.Image:
    gray = img_rgb.convert("L")
    np_g = np.array(gray)
    np_g = cv2.medianBlur(np_g, 5)
    low, high = int(low), int(high)
    if high < low:
        low, high = high, low
    edges = cv2.Canny(np_g, low, high)
    edges = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY)[1]
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges_rgb = np.stack([edges, edges, edges], axis=-1)
    pil = Image.fromarray(edges_rgb.astype(np.uint8), mode="RGB")
    return pil.resize(out_size, Image.NEAREST)


def auto_calibrate_canny(raw_img: Image.Image, cutout_rgba: Image.Image, low_candidates=range(20, 160, 10), high_candidates=range(80, 260, 20), bg_penalty=0.6):
    raw_gray = np.array(raw_img.convert("L"))
    raw_gray = cv2.medianBlur(raw_gray, 5)
    mask = np.array(cutout_rgba.split()[-1]) > 0
    if mask.sum() < 100:
        return 40, 200, {"best_score": 0.0}

    best_score = -1e9
    best_pair = (40, 200)
    total_pixels = raw_gray.size
    obj_pixels = int(mask.sum())
    bg_pixels = int(total_pixels - obj_pixels)

    for low in low_candidates:
        for high in high_candidates:
            if high <= low:
                continue
            edges = (cv2.Canny(raw_gray, low, high) > 0)
            obj_edges = int(edges[mask].sum())
            bg_edges = int(edges[~mask].sum())
            obj_rate = obj_edges / max(obj_pixels, 1)
            bg_rate = bg_edges / max(bg_pixels, 1)
            score = obj_rate - bg_penalty * bg_rate
            if score > best_score:
                best_score = score
                best_pair = (low, high)

    return best_pair[0], best_pair[1], {"best_score": best_score}


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def bgra_to_pil_rgba(bgra: np.ndarray) -> Image.Image:
    rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(rgba)


def resize_and_pad_to_square_transparent(bgr: np.ndarray, target: int):
    h, w = bgr.shape[:2]
    scale = min(target / float(w), target / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(resized := bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.zeros((target, target, 4), dtype=np.uint8)
    resized_bgra = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)
    resized_bgra[..., 3] = 255

    x0 = (target - new_w) // 2
    y0 = (target - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized_bgra
    return canvas


def save_pil_image(img: Image.Image, out_dir: Path, stem_prefix: str, ext: str = ".png") -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{stem_prefix}_{uuid.uuid4().hex[:8]}"
    out_path = out_dir / f"{stem}{ext}"
    img.save(out_path)
    return str(out_path)


def build_point_cloud_plot(
    rgb_img: Image.Image,
    depth_np: np.ndarray,
    max_points: int = 30000,
    depth_scale: float = 1.0,
    z_exaggeration: float = 1.2,
    crop_ratio: float = 1.0,
):
    depth = depth_np.astype(np.float32)

    if depth.ndim == 3:
        depth = depth[0]

    h, w = depth.shape
    rgb_img = rgb_img.convert("RGB").resize((w, h), Image.BILINEAR)
    rgb = np.array(rgb_img)

    if crop_ratio < 1.0:
        ch = int(h * crop_ratio)
        cw = int(w * crop_ratio)
        y0 = max(0, (h - ch) // 2)
        x0 = max(0, (w - cw) // 2)
        rgb = rgb[y0:y0 + ch, x0:x0 + cw]
        depth = depth[y0:y0 + ch, x0:x0 + cw]
        h, w = depth.shape

    depth = depth - np.nanmin(depth)
    depth_max = np.nanmax(depth)
    if not np.isfinite(depth_max) or depth_max <= 1e-8:
        depth_max = 1.0
    depth = depth / depth_max

    yy, xx = np.mgrid[0:h, 0:w]
    fx = float(w)
    fy = float(h)
    cx = w / 2.0
    cy = h / 2.0

    z = depth * float(depth_scale)
    x = (xx - cx) / fx
    y = -(yy - cy) / fy
    z = z * float(z_exaggeration)

    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    flat_z = z.reshape(-1)
    flat_rgb = rgb.reshape(-1, 3)

    n = min(len(flat_x), len(flat_y), len(flat_z), len(flat_rgb))
    flat_x = flat_x[:n]
    flat_y = flat_y[:n]
    flat_z = flat_z[:n]
    flat_rgb = flat_rgb[:n]

    valid = np.isfinite(flat_z)
    flat_x = flat_x[valid]
    flat_y = flat_y[valid]
    flat_z = flat_z[valid]
    flat_rgb = flat_rgb[valid]

    if len(flat_z) > max_points:
        idx = np.linspace(0, len(flat_z) - 1, max_points).astype(np.int32)
        flat_x = flat_x[idx]
        flat_y = flat_y[idx]
        flat_z = flat_z[idx]
        flat_rgb = flat_rgb[idx]

    colors = [f"rgb({r},{g},{b})" for r, g, b in flat_rgb]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=flat_x,
                y=flat_y,
                z=flat_z,
                mode="markers",
                marker=dict(
                    size=1.7,
                    color=colors,
                    opacity=0.96,
                ),
            )
        ]
    )

    fig.update_layout(
        height=760,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=1.45, z=0.9)),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def list_images_sorted(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files = sorted(files, key=lambda x: (x.stat().st_mtime, x.name.lower()), reverse=True)
    return files


def open_image_safe(path: Optional[Path]) -> Optional[Image.Image]:
    if path is None or not path.exists():
        return None
    return Image.open(path).convert("RGB")


def get_latest_pair():
    sdxl_files = list_images_sorted(SDXL_FINAL_DIR)
    multi_files = list_images_sorted(MULTI_VIEW_DIR)

    sdxl_path = sdxl_files[0] if sdxl_files else None
    multi_path = multi_files[0] if multi_files else None

    sdxl_img = open_image_safe(sdxl_path)
    multi_img = open_image_safe(multi_path)

    return (
        sdxl_img,
        str(sdxl_path) if sdxl_path else "",
        multi_img,
        str(multi_path) if multi_path else "",
    )


def init_yolo_main(weights_path: str) -> YOLO:
    global _yolo_main_model, _yolo_main_weights
    if _yolo_main_model is not None and _yolo_main_weights == weights_path:
        return _yolo_main_model
    if not weights_path or not os.path.exists(weights_path):
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    _yolo_main_model = YOLO(weights_path)
    _yolo_main_weights = weights_path
    return _yolo_main_model


def build_raw_and_cutout(uploaded_img: Image.Image, yolo_weights: str, yolo_conf_main: float, preprocess_mode: str):
    img_rgb = to_rgb(uploaded_img)
    img_proc = preprocess_sonar(img_rgb, mode=preprocess_mode) if APPLY_SONAR_PREPROCESS else img_rgb

    model = init_yolo_main(yolo_weights)
    results = model.predict(source=img_proc, conf=float(yolo_conf_main), imgsz=max(img_proc.size), verbose=False)

    if not results or results[0] is None or results[0].masks is None or results[0].masks.data is None:
        return img_proc, img_proc.convert("RGBA")

    masks = results[0].masks.data
    if masks.ndim != 3 or masks.shape[0] < 1:
        return img_proc, img_proc.convert("RGBA")

    areas = masks.float().sum(dim=(1, 2))
    idx = int(areas.argmax())
    mask = masks[idx].detach().cpu().numpy().astype(np.uint8) * 255
    mask = cv2.resize(mask, img_proc.size, interpolation=cv2.INTER_NEAREST)

    mask_l = Image.fromarray(mask, mode="L")
    cutout_rgba = img_proc.convert("RGBA")
    cutout_rgba.putalpha(mask_l)
    return img_proc, cutout_rgba


def init_qwen3_vl(model_name: str):
    global _qwen_model, _qwen_processor, _qwen_model_name
    if _qwen_model is not None and _qwen_model_name == model_name:
        return _qwen_model, _qwen_processor

    unload_qwen()
    _qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        dtype="auto",
        token=HF_TOKEN if HF_TOKEN else None,
    )
    _qwen_processor = AutoProcessor.from_pretrained(
        model_name,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    _qwen_model.eval()
    _qwen_model_name = model_name
    return _qwen_model, _qwen_processor


@torch.inference_mode()
def qwen_chat(model, processor, images: List[Image.Image], user_text: str, max_new_tokens: int) -> str:
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": user_text})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=int(max_new_tokens))
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    out = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return (out[0] if out else "").strip()


def init_sdxl_controlnet_pipe(base_model: str, controlnet_model: str):
    global _cn, _sdxl_cn_pipe, _sdxl_cn_key, _sdxl_cn_device

    key = (base_model, controlnet_model, "cn_img2img")
    if _sdxl_cn_pipe is not None and _sdxl_cn_key == key:
        return _sdxl_cn_pipe, _sdxl_cn_device

    unload_sdxl_cn()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    _cn = ControlNetModel.from_pretrained(
        controlnet_model,
        torch_dtype=dtype,
        use_safetensors=True,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    pipe_kwargs = dict(
        controlnet=_cn,
        torch_dtype=dtype,
        use_safetensors=True,
        token=HF_TOKEN if HF_TOKEN else None,
    )
    if "Juggernaut-XL-v9" in base_model or "Juggernaut" in base_model:
        pipe_kwargs["variant"] = "fp16"

    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(base_model, **pipe_kwargs)
    pipe.set_progress_bar_config(disable=True)

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    if device == "cuda":
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
        pipe = pipe.to("cuda")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    else:
        pipe = pipe.to("cpu")

    _sdxl_cn_pipe = pipe
    _sdxl_cn_key = key
    _sdxl_cn_device = device
    return _sdxl_cn_pipe, _sdxl_cn_device


def apply_sdxl_scheduler(pipe, scheduler_name: str):
    name = (scheduler_name or DEFAULT_SCHEDULER).strip()
    current = pipe.scheduler.config
    if name == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(current)
    elif name == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(current)
    elif name == "Euler A":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(current)
    elif name == "DPM++ 2M":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(current, use_karras_sigmas=False)
    elif name == "DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(current, use_karras_sigmas=True)
    else:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(current)
    return pipe


def init_da3_model(model_preset: str = DEFAULT_DA3_MODEL_PRESET):
    global _da3_model, _da3_model_name

    if DepthAnything3 is None:
        raise RuntimeError("Depth Anything 3 is not installed. Install the official package/repo first.")

    if _da3_model is not None and _da3_model_name == model_preset:
        return _da3_model

    unload_da3()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preset_candidates = [model_preset, model_preset.lower(), model_preset.upper()]
    last_error = None

    for preset in preset_candidates:
        try:
            model = DepthAnything3.from_pretrained(f"depth-anything/{preset}").to(device)
            _da3_model = model
            _da3_model_name = model_preset
            return _da3_model
        except Exception as e:
            last_error = e

    for preset in preset_candidates:
        try:
            model = DepthAnything3(model_name=preset).to(device)
            _da3_model = model
            _da3_model_name = model_preset
            return _da3_model
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Failed to initialize Depth Anything 3 model '{model_preset}': {last_error}")


def extract_depth_from_prediction(prediction):
    if prediction is None:
        raise RuntimeError("DA3 returned no prediction object.")

    if hasattr(prediction, "depth"):
        depth_maps = prediction.depth
    elif isinstance(prediction, dict) and "depth" in prediction:
        depth_maps = prediction["depth"]
    else:
        raise RuntimeError("DA3 prediction does not contain a depth output.")

    if isinstance(depth_maps, list):
        if len(depth_maps) == 0:
            raise RuntimeError("DA3 depth output list is empty.")
        depth_np = np.array(depth_maps[0], dtype=np.float32)
    else:
        depth_np = np.array(depth_maps[0], dtype=np.float32) if getattr(depth_maps, "ndim", 0) == 3 else np.array(depth_maps, dtype=np.float32)

    if depth_np.ndim == 3:
        depth_np = depth_np[0]

    return depth_np


def run_da3_pointcloud(
    input_image: Image.Image,
    input_path: str,
    source_label: str,
    max_points: int,
    depth_scale: float,
    z_exaggeration: float,
    crop_ratio: float,
):
    if input_image is None:
        return None, f"No image available for 3D generation from {source_label}."

    try:
        model = init_da3_model(DEFAULT_DA3_MODEL_PRESET)

        run_id = uuid.uuid4().hex[:8]
        export_dir = DA3_EXPORT_DIR / f"scene_{run_id}"
        export_dir.mkdir(parents=True, exist_ok=True)

        temp_input_path = save_pil_image(
            input_image.convert("RGB"),
            TMP_DIR,
            f"da3_input_{run_id}",
            ext=".png",
        )

        try:
            prediction = model.inference(
                image=[temp_input_path],
                export_dir=str(export_dir),
                export_format="mini_npz-glb",
                num_max_points=int(max_points),
                conf_thresh_percentile=40.0,
                show_cameras=False,
            )
        except TypeError:
            prediction = model.inference(image=[temp_input_path])

        depth_np = extract_depth_from_prediction(prediction)

        target_w, target_h = input_image.size
        if depth_np.shape[:2] != (target_h, target_w):
            depth_np = cv2.resize(
                depth_np,
                (target_w, target_h),
                interpolation=cv2.INTER_CUBIC,
            )

        fig = build_point_cloud_plot(
            rgb_img=input_image,
            depth_np=depth_np,
            max_points=int(max_points),
            depth_scale=float(depth_scale),
            z_exaggeration=float(z_exaggeration),
            crop_ratio=float(crop_ratio),
        )

        log = (
            f"3D run completed.\n"
            f"3D source: {source_label}\n"
            f"3D source path: {input_path if input_path else 'In-memory image'}\n"
            f"DA3 model preset: {DEFAULT_DA3_MODEL_PRESET}\n"
            f"Input image size: {input_image.size}\n"
            f"Depth shape after resize: {depth_np.shape}"
        )
        return fig, log

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def run_da3_from_sdxl(
    sdxl_img: Image.Image,
    sdxl_path: str,
    max_points: int,
    depth_scale: float,
    z_exaggeration: float,
    crop_ratio: float,
):
    return run_da3_pointcloud(
        input_image=sdxl_img,
        input_path=sdxl_path,
        source_label="SDXL Image",
        max_points=max_points,
        depth_scale=depth_scale,
        z_exaggeration=z_exaggeration,
        crop_ratio=crop_ratio,
    )


def run_da3_from_multiview(
    multi_img: Image.Image,
    multi_path: str,
    max_points: int,
    depth_scale: float,
    z_exaggeration: float,
    crop_ratio: float,
):
    return run_da3_pointcloud(
        input_image=multi_img,
        input_path=multi_path,
        source_label="Multi-View Image",
        max_points=max_points,
        depth_scale=depth_scale,
        z_exaggeration=z_exaggeration,
        crop_ratio=crop_ratio,
    )


def run_2d_pipeline(
    uploaded_raw_sonar: Image.Image,
    preprocess_mode: str,
    sdxl_strength: float,
    sdxl_guidance: float,
    sdxl_steps: int,
    sdxl_seed: int,
    sdxl_target_long_side: int,
    sdxl_cn_scale: float,
    sdxl_scheduler: str,
):
    if uploaded_raw_sonar is None:
        return None, "", None, "", "", "Please upload a raw sonar image first."

    try:
        clear_output_images()

        yolo_weights = DEFAULT_YOLO_SEG_WEIGHTS
        yolo_conf_main = YOLO_CONF_MAIN_DEFAULT
        target_size = TARGET_SIZE_DEFAULT

        qwen_seed = DEFAULT_PARAMS_SEED_QWEN
        qwen_image_side = DEFAULT_QWEN_IMAGE_SIDE
        qwen_max_tokens = DEFAULT_QWEN_MAX_TOKENS

        raw_pre, cutout_rgba = build_raw_and_cutout(
            uploaded_raw_sonar,
            yolo_weights,
            yolo_conf_main=float(yolo_conf_main),
            preprocess_mode=preprocess_mode,
        )
        cut_preview = cutout_preview_for_qwen(cutout_rgba)

        raw_qwen = resize_long_side_exact(raw_pre, int(qwen_image_side))
        cut_qwen = resize_long_side_exact(cut_preview, int(qwen_image_side))

        model, processor = init_qwen3_vl(DEFAULT_QWEN_VL_MODEL)
        torch.manual_seed(int(qwen_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(qwen_seed))

        prompt_text = qwen_chat(
            model,
            processor,
            [raw_qwen, cut_qwen],
            DEFAULT_IDEA_TEXT,
            max_new_tokens=int(qwen_max_tokens),
        )
        prompt_text = clean_qwen_prompt_text(prompt_text)
        unload_qwen()

        if not (prompt_text or "").strip():
            return raw_pre, "", None, "", "", "Qwen returned an empty prompt."

        used_prompt = build_prompt(prompt_text)

        init_img = build_blue_textured_base(raw_pre, int(sdxl_target_long_side))
        auto_low, auto_high, _ = auto_calibrate_canny(raw_pre, cutout_rgba)
        control_img = make_canny_control_image(
            to_rgb(raw_pre),
            low=auto_low,
            high=auto_high,
            out_size=init_img.size,
        )

        pipe, device = init_sdxl_controlnet_pipe(DEFAULT_SD_MODEL, DEFAULT_CONTROLNET)
        pipe = apply_sdxl_scheduler(pipe, sdxl_scheduler)
        gen = torch.Generator(device=device).manual_seed(int(sdxl_seed))

        out_img = pipe(
            prompt=used_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=init_img,
            control_image=control_img,
            strength=float(sdxl_strength),
            guidance_scale=float(sdxl_guidance),
            num_inference_steps=int(sdxl_steps),
            controlnet_conditioning_scale=float(sdxl_cn_scale),
            generator=gen,
        ).images[0]

        sdxl_path = save_pil_image(out_img, SDXL_FINAL_DIR, "sdxl_final_output", ext=".png")

        bgr_orig = pil_to_bgr(out_img)
        prepped_bgra = resize_and_pad_to_square_transparent(bgr_orig, int(target_size))
        prepped_rgba = bgra_to_pil_rgba(prepped_bgra)
        _ = save_pil_image(prepped_rgba, TMP_DIR, f"resized_{int(target_size)}", ext=".png")

        sdxl_img, sdxl_first_path, multi_img, multi_first_path = get_latest_pair()

        log_text = (
            "Run 2D completed.\n"
            "Previous images in sdxl_final_output and multi_view_output were cleared.\n"
            "New SDXL image saved.\n"
            "If you generate a new multi-view result externally, click Reload / Get Multi-View Image."
        )

        return raw_pre, sdxl_img, multi_img, sdxl_first_path, prompt_text, log_text

    except Exception as e:
        return None, None, None, "", "", f"{type(e).__name__}: {e}"


def reload_multi_view(current_sdxl_path: str):
    sdxl_img = open_image_safe(Path(current_sdxl_path)) if current_sdxl_path else None
    multi_files = list_images_sorted(MULTI_VIEW_DIR)
    multi_path = multi_files[0] if multi_files else None
    multi_img = open_image_safe(multi_path)

    log_text = (
        f"Reloaded latest multi-view image.\n"
        f"Multi-view available: {'Yes' if multi_img is not None else 'No'}"
    )

    return multi_img, str(multi_path) if multi_path else "", log_text


with gr.Blocks(title=APP_TITLE, fill_width=True, css=BIG_UI_CSS) as demo:
    state_sdxl_path = gr.State("")
    state_multi_path = gr.State("")

    gr.Markdown(f"## {APP_TITLE}")

    with gr.Row(equal_height=False):
        with gr.Column(scale=3, min_width=360):
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown(
                    "<div class='wrap'>"
                    "<div style='color:#e5e7eb;font-weight:700;'>Raw Sonar Input</div>"
                    "<div class='small'>Upload the raw sonar image here. Run 2D will use this uploaded image directly.</div>"
                    "</div>"
                )

                raw_sonar_input = gr.Image(type="pil", label="Upload Raw Sonar Image", height=300)

                preprocess_mode = gr.Dropdown(
                    choices=["raw", "brighten"],
                    value=DEFAULT_PREPROCESS_MODE,
                    label="Preprocess Mode"
                )

                with gr.Row(elem_classes=["btnrow"]):
                    btn_run2d = gr.Button("Run 2D", variant="primary")
                    btn_reload_multi = gr.Button("Reload / Get Multi-View Image")

                with gr.Row(elem_classes=["btnrow"]):
                    btn_run3d_sdxl = gr.Button("Run 3D (SDXL Image)", variant="primary")
                    btn_run3d_multi = gr.Button("Run 3D (Multi-View Image)", variant="primary")

            with gr.Group(elem_classes=["panel"]):
                gr.Markdown(
                    "<div class='wrap'>"
                    "<div style='color:#e5e7eb;font-weight:700;'>Settings</div>"
                    "<div class='small'>SDXL and point cloud controls.</div>"
                    "</div>"
                )

                with gr.Accordion("SDXL Settings", open=False):
                    sdxl_strength = gr.Slider(0.05, 0.95, value=float(DEFAULT_STRENGTH), step=0.01, label="SDXL strength")
                    sdxl_guidance = gr.Slider(1.0, 15.0, value=float(DEFAULT_GUIDANCE), step=0.1, label="SDXL guidance")
                    sdxl_steps = gr.Slider(20, 150, value=int(DEFAULT_STEPS), step=1, label="SDXL steps")
                    sdxl_seed = gr.Slider(0, 1_000_000, value=int(DEFAULT_PARAMS_SEED_SDXL), step=1, label="SDXL seed")
                    sdxl_target_long_side = gr.Slider(512, 1536, value=int(DEFAULT_TARGET_LONG_SIDE), step=64, label="SDXL long-side")
                    sdxl_cn_scale = gr.Slider(0.0, 1.5, value=float(DEFAULT_CN_SCALE), step=0.01, label="ControlNet scale")
                    sdxl_scheduler = gr.Dropdown(
                        choices=["Euler A", "Euler", "DDIM", "DPM++ 2M", "DPM++ 2M Karras"],
                        value=DEFAULT_SCHEDULER,
                        label="SDXL scheduler"
                    )

                with gr.Accordion("3D Point Cloud Settings", open=False):
                    pc_max_points = gr.Slider(5000, 80000, value=DEFAULT_PC_MAX_POINTS, step=1000, label="Max points")
                    pc_depth_scale = gr.Slider(0.2, 3.0, value=DEFAULT_PC_DEPTH_SCALE, step=0.1, label="Depth scale")
                    pc_z_exaggeration = gr.Slider(0.5, 3.0, value=DEFAULT_PC_Z_EXAGGERATION, step=0.1, label="Z exaggeration")
                    pc_crop_ratio = gr.Slider(0.3, 1.0, value=DEFAULT_PC_CROP_RATIO, step=0.05, label="Center crop ratio")

        with gr.Column(scale=5, min_width=780):
            with gr.Group(elem_classes=["panel"], elem_id="plot_panel"):
                gr.Markdown(
                    "<div class='wrap'>"
                    "<div style='color:#e5e7eb;font-weight:700;'>Final 3D Result</div>"
                    "<div class='small'>Depth Anything 3 point cloud result.</div>"
                    "</div>"
                )
                point_cloud_plot = gr.Plot(label="3D Point Cloud")

        with gr.Column(scale=2, min_width=320, elem_id="rightcol"):
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown(
                    "<div class='wrap'>"
                    "<div style='color:#e5e7eb;font-weight:700;'>Preprocessed Raw Sonar</div>"
                    "<div class='small'>The uploaded sonar image after preprocessing.</div>"
                    "</div>"
                )
                raw_preview_img = gr.Image(type="pil", label="Preprocessed Raw Sonar", height=180, interactive=False)

            with gr.Group(elem_classes=["panel"]):
                gr.Markdown(
                    "<div class='wrap'>"
                    "<div style='color:#e5e7eb;font-weight:700;'>SDXL Image</div>"
                    "<div class='small'>Latest image from sdxl_final_output.</div>"
                    "</div>"
                )
                sdxl_folder_img = gr.Image(type="pil", label="SDXL View", height=180, interactive=False)

            with gr.Group(elem_classes=["panel"]):
                gr.Markdown(
                    "<div class='wrap'>"
                    "<div style='color:#e5e7eb;font-weight:700;'>Multi-View Image</div>"
                    "<div class='small'>Latest image from multi_view_output. Reload it with the middle button.</div>"
                    "</div>"
                )
                multi_folder_img = gr.Image(type="pil", label="Multi-View", height=180, interactive=False)

            with gr.Group(elem_classes=["panel"]):
                gr.Markdown(
                    "<div class='wrap'>"
                    "<div style='color:#e5e7eb;font-weight:700;'>Qwen Prompt</div>"
                    "<div class='small'>Prompt generated from raw image + YOLO cutout.</div>"
                    "</div>"
                )
                qwen_prompt_box = gr.Textbox(label="Qwen Prompt", lines=8, interactive=False)

            with gr.Group(elem_classes=["panel"]):
                gr.Markdown(
                    "<div class='wrap'>"
                    "<div style='color:#e5e7eb;font-weight:700;'>Logs</div>"
                    "<div class='small'>Status messages only.</div>"
                    "</div>"
                )
                with gr.Accordion("Show logs", open=False):
                    log_status = gr.Textbox(label="Status", lines=10, interactive=False)

    btn_run2d.click(
        fn=run_2d_pipeline,
        inputs=[
            raw_sonar_input,
            preprocess_mode,
            sdxl_strength,
            sdxl_guidance,
            sdxl_steps,
            sdxl_seed,
            sdxl_target_long_side,
            sdxl_cn_scale,
            sdxl_scheduler,
        ],
        outputs=[
            raw_preview_img,
            sdxl_folder_img,
            multi_folder_img,
            state_sdxl_path,
            qwen_prompt_box,
            log_status,
        ],
    )

    btn_reload_multi.click(
        fn=reload_multi_view,
        inputs=[state_sdxl_path],
        outputs=[
            multi_folder_img,
            state_multi_path,
            log_status,
        ],
    )

    btn_run3d_sdxl.click(
        fn=run_da3_from_sdxl,
        inputs=[
            sdxl_folder_img,
            state_sdxl_path,
            pc_max_points,
            pc_depth_scale,
            pc_z_exaggeration,
            pc_crop_ratio,
        ],
        outputs=[point_cloud_plot, log_status],
    )

    btn_run3d_multi.click(
        fn=run_da3_from_multiview,
        inputs=[
            multi_folder_img,
            state_multi_path,
            pc_max_points,
            pc_depth_scale,
            pc_z_exaggeration,
            pc_crop_ratio,
        ],
        outputs=[point_cloud_plot, log_status],
    )


if __name__ == "__main__":
    print(f"[INFO] Starting local app on http://0.0.0.0:{DEFAULT_PORT}")

    public_url = start_cloudflared(DEFAULT_PORT)
    if public_url:
        print(f"[INFO] Public URL: {public_url}")
    else:
        print("[WARNING] Cloudflare tunnel URL not found automatically.")
        print("[INFO] Try this manually in terminal:")
        print(r"grep -o 'https://[-a-zA-Z0-9]*\.trycloudflare\.com' /tmp/cloudflared.log")

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=DEFAULT_PORT,
        share=SHARE_DEFAULT,
        debug=True,
        show_error=True,
    )