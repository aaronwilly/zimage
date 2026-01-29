import os
import sys
import time
import subprocess
from pathlib import Path

# Force all model/cache downloads into the project folder (not C:\Users\<you>\.cache\...).
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_CACHE_DIR = PROJECT_DIR / "models_cache"
MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face + Transformers + Diffusers cache locations.
# Set these early so downstream imports pick them up.
os.environ.setdefault("HF_HOME", str(MODELS_CACHE_DIR))
HF_HUB_DIR = MODELS_CACHE_DIR / "hub"
# Canonical cache env var (keep deprecated alias for compatibility).
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_CACHE_DIR / "transformers"))
os.environ.setdefault("DIFFUSERS_CACHE", str(MODELS_CACHE_DIR / "diffusers"))
# Slow connections often hit the default 10s timeout on large safetensors.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1200")  # seconds (20 min/file)
# Some Windows/proxy setups fail with HF's Xet backend and leave *.incomplete blobs.
# Disabling it forces the regular HTTP download path.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch

# Global speed flags (RTX 4070 friendly)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except Exception:
        pass

# Windows terminals often default to cp1252, which can crash when Gradio prints emoji.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import spaces  # Hugging Face Spaces runtime helper (e.g., @spaces.GPU)
except ModuleNotFoundError:
    # Local/dev fallback: run without HF Spaces' ZeroGPU helpers.
    class _SpacesShim:
        @staticmethod
        def GPU(func=None, **_kwargs):
            # Supports both usages:
            # - @spaces.GPU
            # - @spaces.GPU(duration=...)
            if callable(func):
                return func

            def _decorator(f):
                return f

            return _decorator

    spaces = _SpacesShim()
import gradio as gr
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download

REPO_ID = "Tongyi-MAI/Z-Image-Turbo"
pipe = None


def _pick_device_and_dtype():
    if torch.cuda.is_available():
        # Force fp16 for speed on RTX 4070 (often faster than bf16 in diffusers paths)
        return "cuda", torch.float16
    return "cpu", torch.float32


def _ensure_model_downloaded(repo_id: str) -> str:
    """
    Download (with resume) and return a local folder path.
    This avoids half-downloaded cache snapshots causing cryptic 'missing model.safetensors' errors.
    """
    repo_folder = HF_HUB_DIR / ("models--" + repo_id.replace("/", "--"))
    blobs_dir = repo_folder / "blobs"
    locks_dir = HF_HUB_DIR / ".locks" / ("models--" + repo_id.replace("/", "--"))

    def _cleanup_partials():
        if blobs_dir.exists():
            for p in blobs_dir.glob("*.incomplete"):
                try:
                    p.unlink()
                except Exception:
                    pass
        if locks_dir.exists():
            for p in locks_dir.glob("*.lock"):
                try:
                    p.unlink()
                except Exception:
                    pass

    def _progress_signature():
        if not blobs_dir.exists():
            return (0, 0, 0)
        incompletes = list(blobs_dir.glob("*.incomplete"))
        if not incompletes:
            return (0, 0, 0)
        total_size = 0
        latest_mtime = 0
        for p in incompletes:
            try:
                st = p.stat()
                total_size += st.st_size
                latest_mtime = max(latest_mtime, int(st.st_mtime))
            except Exception:
                continue
        return (len(incompletes), total_size, latest_mtime)

    # Run the Hub download in a separate Python process so we can safely terminate
    # and retry if the network stalls (common on slow links with large safetensors).
    result_path = MODELS_CACHE_DIR / "_snapshot_download_result.txt"
    child_code = r"""
from huggingface_hub import snapshot_download
import os, sys
from pathlib import Path

repo_id = sys.argv[1]
out_path = Path(sys.argv[2])
cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
local_dir = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, max_workers=1)
out_path.write_text(local_dir, encoding="utf-8")
"""

    max_attempts = 5
    stall_seconds = 300  # 5 minutes without any *.incomplete growth/mtime change

    for attempt in range(1, max_attempts + 1):
        _cleanup_partials()
        try:
            result_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

        print(f"Downloading model snapshot (attempt {attempt}/{max_attempts})...")
        proc = subprocess.Popen(
            [sys.executable, "-c", child_code, repo_id, str(result_path)],
            env=os.environ.copy(),
        )

        last_sig = _progress_signature()
        last_change = time.monotonic()
        stalled = False

        while proc.poll() is None:
            time.sleep(15)
            sig = _progress_signature()
            if sig != last_sig:
                last_sig = sig
                last_change = time.monotonic()
            elif sig != (0, 0, 0) and (time.monotonic() - last_change) > stall_seconds:
                stalled = True
                print(
                    "Download appears stalled (no progress for "
                    f"{stall_seconds}s). Restarting download..."
                )
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
                break

        if stalled:
            _cleanup_partials()
            continue

        rc = proc.wait()
        if rc != 0:
            _cleanup_partials()
            raise RuntimeError(f"Model download process failed (exit code {rc}).")

        if not result_path.exists():
            _cleanup_partials()
            raise RuntimeError("Model download did not produce a snapshot path (unexpected).")

        local_dir = result_path.read_text(encoding="utf-8").strip()
        if not local_dir:
            _cleanup_partials()
            raise RuntimeError("Model download returned an empty snapshot path (unexpected).")

        return local_dir

    raise RuntimeError(
        "Model download repeatedly stalled.\n\n"
        "This is almost always caused by an unstable/slow connection or a proxy/firewall cutting large downloads.\n"
        "Try:\n"
        "- Use a different network/VPN\n"
        "- Ensure `cdn-lfs.huggingface.co` is reachable\n"
        "- Run `huggingface-cli login` if the repo is gated\n"
    )


def _validate_snapshot_has_weights(local_dir: str) -> None:
    """Fail fast with a helpful message if weights were not downloaded."""
    root = Path(local_dir)
    # Most HF repos will contain at least one safetensors/bin file when fully downloaded.
    has_weights = any(root.rglob("*.safetensors")) or any(root.rglob("pytorch_model*.bin"))
    if has_weights:
        return
    raise RuntimeError(
        "Model files were not downloaded (no *.safetensors / *.bin found).\n\n"
        "Common causes:\n"
        "- Your network/firewall blocks Hugging Face large-file downloads (cdn-lfs.huggingface.co)\n"
        "- The model repo is gated and requires accepting terms / logging in\n"
        "- A previous interrupted download left partial cache state\n\n"
        f"Current cache folder: {MODELS_CACHE_DIR}\n"
        "Fix:\n"
        "- Try `huggingface-cli login` (or set HF_TOKEN), then rerun\n"
        "- Ensure you can access `https://huggingface.co` and `https://cdn-lfs.huggingface.co`\n"
        "- Delete the folder `models_cache/hub/models--Tongyi-MAI--Z-Image-Turbo` and retry\n"
    )


def get_pipe() -> DiffusionPipeline:
    global pipe
    if pipe is not None:
        return pipe

    device, dtype = _pick_device_and_dtype()
    print(f"Loading {REPO_ID} pipeline on {device} ({dtype})...")

    try:
        local_dir = _ensure_model_downloaded(REPO_ID)
        _validate_snapshot_has_weights(local_dir)
        pipe = DiffusionPipeline.from_pretrained(
            local_dir,
            torch_dtype=dtype,
            # Reduce RAM spikes during load (especially important on Windows).
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    except Exception as e:
        # Common scenario: interrupted download / proxy / no internet.
        cache_hint = str(MODELS_CACHE_DIR / "hub")
        msg = (
            "Failed to download/load the model files.\n\n"
            "- If this happened mid-download, delete the broken cache folder and retry.\n"
            f"  Cache path is usually: {cache_hint}\n"
            "- Or set HF_HOME to a fresh folder and rerun.\n\n"
            f"Root error: {e}"
        )
        raise RuntimeError(msg) from e

    pipe.to(device)
    if device == "cuda":
        print("CUDA available:", torch.cuda.is_available())
        print("GPU:", torch.cuda.get_device_name(0))
        print("dtype:", dtype)
    return pipe

# (Optional) AoTI compilation + FA3 (Spaces-only). Keep disabled for local runs.
# pipe.transformer.layers._repeated_blocks = ["ZImageTransformerBlock"]
# spaces.aoti_blocks_load(pipe.transformer.layers, "zerogpu-aoti/Z-Image", variant="fa3")

@spaces.GPU
def generate_image(prompt, height, width, num_inference_steps, seed, randomize_seed, progress=gr.Progress(track_tqdm=True)):
    try:
        pipe = get_pipe()
    except Exception as e:
        raise gr.Error(str(e))

    if randomize_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    if torch.cuda.is_available():
        generator = torch.Generator("cuda").manual_seed(int(seed))
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = pipe(
                prompt=prompt,
                height=int(height),
                width=int(width),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=0.0,
                generator=generator,
            )
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"Inference seconds: {t1 - t0:.2f}s")
    else:
        generator = torch.Generator("cpu").manual_seed(int(seed))
        t0 = time.time()
        with torch.inference_mode():
            out = pipe(
                prompt=prompt,
                height=int(height),
                width=int(width),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=0.0,
                generator=generator,
            )
        t1 = time.time()
        print(f"(CPU) Inference seconds: {t1 - t0:.2f}s")

    image = out.images[0]
    return image, seed

# Example prompts
examples = [
    ["Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp, bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda, blurred colorful distant lights."],
    ["A majestic dragon soaring through clouds at sunset, scales shimmering with iridescent colors, detailed fantasy art style"],
    ["Cozy coffee shop interior, warm lighting, rain on windows, plants on shelves, vintage aesthetic, photorealistic"],
    ["Astronaut riding a horse on Mars, cinematic lighting, sci-fi concept art, highly detailed"],
    ["Portrait of a wise old wizard with a long white beard, holding a glowing crystal staff, magical forest background"],
]

# Custom theme with modern aesthetics (Gradio 6)
custom_theme = gr.themes.Soft(
    primary_hue="yellow",
    secondary_hue="amber",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    text_size="lg",
    spacing_size="md",
    radius_size="lg"
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    block_title_text_weight="600",
)

# Build the Gradio interface
with gr.Blocks(fill_height=True) as demo:
    # Header
    gr.Markdown(
        """
        # 🎨 Z-Image-Turbo
        **Ultra-fast AI image generation** • Generate stunning images in just 8 steps
        """,
        elem_classes="header-text"
    )
    
    with gr.Row(equal_height=False):
        # Left column - Input controls
        with gr.Column(scale=1, min_width=320):
            prompt = gr.Textbox(
                label="✨ Your Prompt",
                placeholder="Describe the image you want to create...",
                lines=5,
                max_lines=10,
                autofocus=True,
            )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    height = gr.Slider(
                        minimum=512,
                        maximum=2048,
                        value=1024,
                        step=64,
                        label="Height",
                        info="Image height in pixels"
                    )
                    width = gr.Slider(
                        minimum=512,
                        maximum=2048,
                        value=1024,
                        step=64,
                        label="Width",
                        info="Image width in pixels"
                    )
                
                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=9,
                    step=1,
                    label="Inference Steps",
                    info="9 steps = 8 DiT forwards (recommended)"
                )
                
                with gr.Row():
                    randomize_seed = gr.Checkbox(
                        label="🎲 Random Seed",
                        value=True,
                    )
                    seed = gr.Number(
                        label="Seed",
                        value=42,
                        precision=0,
                        visible=False,
                    )
                
                def toggle_seed(randomize):
                    return gr.Number(visible=not randomize)
                
                randomize_seed.change(
                    toggle_seed,
                    inputs=[randomize_seed],
                    outputs=[seed]
                )
            
            generate_btn = gr.Button(
                "🚀 Generate Image",
                variant="primary",
                size="lg",
                scale=1
            )
            
            # Example prompts
            gr.Examples(
                examples=examples,
                inputs=[prompt],
                label="💡 Try these prompts",
                examples_per_page=5,
            )
        
        # Right column - Output
        with gr.Column(scale=1, min_width=320):
            output_image = gr.Image(
                label="Generated Image",
                type="pil",
                show_label=False,
                height=600,
                buttons=["download", "share"],
            )
            
            used_seed = gr.Number(
                label="🎲 Seed Used",
                interactive=False,
                container=True,
            )
    
    # Footer credits
    gr.Markdown(
        """
        ---
        <div style="text-align: center; opacity: 0.7; font-size: 0.9em; margin-top: 1rem;">
        <strong>Model:</strong> <a href="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo" target="_blank">Tongyi-MAI/Z-Image-Turbo</a> (Apache 2.0 License) • 
        <strong>Demo by:</strong> <a href="https://x.com/realmrfakename" target="_blank">@mrfakename</a> • 
        <strong>Redesign by:</strong> AnyCoder • 
        <strong>Optimizations:</strong> <a href="https://huggingface.co/multimodalart" target="_blank">@multimodalart</a> (FA3 + AoTI)
        </div>
        """,
        elem_classes="footer-text"
    )
    
    # Connect the generate button
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, height, width, num_inference_steps, seed, randomize_seed],
        outputs=[output_image, used_seed],
    )
    
    # Also allow generating by pressing Enter in the prompt box
    prompt.submit(
        fn=generate_image,
        inputs=[prompt, height, width, num_inference_steps, seed, randomize_seed],
        outputs=[output_image, used_seed],
    )

if __name__ == "__main__":
    demo.launch(
        theme=custom_theme,
        css="""
        .header-text h1 {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header-text p {
            font-size: 1.1rem !important;
            color: #64748b !important;
            margin-top: 0 !important;
        }
        
        .footer-text {
            padding: 1rem 0;
        }
        
        .footer-text a {
            color: #f59e0b !important;
            text-decoration: none !important;
            font-weight: 500;
        }
        
        .footer-text a:hover {
            text-decoration: underline !important;
        }
        
        /* Mobile optimizations */
        @media (max-width: 768px) {
            .header-text h1 {
                font-size: 1.8rem !important;
            }
            
            .header-text p {
                font-size: 1rem !important;
            }
        }
        
        /* Smooth transitions */
        button, .gr-button {
            transition: all 0.2s ease !important;
        }
        
        button:hover, .gr-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Better spacing */
        .gradio-container {
            max-width: 1400px !important;
            margin: 0 auto !important;
        }
        """,
        footer_links=[
            "api",
            "gradio"
        ],
        mcp_server=True
    )