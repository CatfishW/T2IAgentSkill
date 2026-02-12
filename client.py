#!/usr/bin/env python3
"""
Z-Image Studio — CLI Client
Comprehensive client for the Z-Image-Turbo API at https://mc.agaii.org/T2IAPI/

Usage examples:
    # Simple generation
    python3 client.py "a cute cat in a garden"

    # With options
    python3 client.py "sunset over mountains" --size 1024x768 --steps 6 --batch 2

    # Save to specific directory
    python3 client.py "cyberpunk city" -o ./my_images/

    # Batch from file (one prompt per line)
    python3 client.py --file prompts.txt --size 768x768

    # JSON output (for piping)
    python3 client.py "a robot" --json

    # Custom API endpoint
    python3 client.py "a dog" --api http://localhost:23530
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ── Config ──
DEFAULT_API = "https://mc.agaii.org/T2IAPI"
DEFAULT_SIZE = "768x768"
DEFAULT_STEPS = 4
DEFAULT_CFG = 1.0
DEFAULT_SAMPLER = "euler"
DEFAULT_FORMAT = "png"

# ── Colors ──
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

    @staticmethod
    def disable():
        for attr in ["RESET", "BOLD", "DIM", "RED", "GREEN", "YELLOW", "BLUE", "MAGENTA", "CYAN"]:
            setattr(C, attr, "")


def print_banner():
    print(f"""
{C.MAGENTA}{C.BOLD}  ╔══════════════════════════════╗
  ║   Z-Image Studio CLI Client  ║
  ║   ─────────────────────────   ║
  ║   Powered by Z-Image-Turbo   ║
  ╚══════════════════════════════╝{C.RESET}
""")


def safe_filename(prompt, max_len=50):
    """Convert prompt to a safe filename."""
    safe = re.sub(r'[^a-zA-Z0-9\s]', '', prompt).strip()
    safe = re.sub(r'\s+', '_', safe)
    return safe[:max_len] if safe else "image"


def check_server(api_base):
    """Check if the server is reachable."""
    try:
        req = Request(f"{api_base}/sdapi/v1/options", method="GET")
        with urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass
    return False


def generate_images(api_base, prompt, negative_prompt="", width=768, height=768,
                    steps=DEFAULT_STEPS, cfg_scale=DEFAULT_CFG, seed=-1,
                    batch=1, sampler=DEFAULT_SAMPLER, output_format=DEFAULT_FORMAT):
    """
    Call the image generation API and return a list of base64 image strings.
    """
    body = {
        "prompt": prompt,
        "size": f"{width}x{height}",
        "n": batch,
        "output_format": output_format,
    }

    # Build sd_cpp_extra_args
    extra = {}
    if negative_prompt:
        extra["negative_prompt"] = negative_prompt
    if steps > 0:
        extra["sample_steps"] = steps
    if cfg_scale >= 0:
        extra["cfg_scale"] = cfg_scale
    if seed >= 0:
        extra["seed"] = seed
    if sampler:
        extra["sampling_method"] = sampler

    if extra:
        body["prompt"] = prompt + f" <sd_cpp_extra_args>{json.dumps(extra)}</sd_cpp_extra_args>"

    data = json.dumps(body).encode("utf-8")
    req = Request(
        f"{api_base}/v1/images/generations",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        try:
            err = json.loads(error_body)
            raise RuntimeError(f"API error {e.code}: {err.get('error', error_body)}")
        except json.JSONDecodeError:
            raise RuntimeError(f"API error {e.code}: {error_body}")
    except URLError as e:
        raise RuntimeError(f"Connection failed: {e.reason}")

    images = []
    for item in result.get("data", []):
        b64 = item.get("b64_json", "")
        if b64:
            images.append(b64)

    return images


def save_image(b64_data, filepath, output_format="png"):
    """Save a base64-encoded image to disk."""
    img_bytes = base64.b64decode(b64_data)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(img_bytes)
    return len(img_bytes)


def format_size(num_bytes):
    """Human-readable file size."""
    for unit in ["B", "KB", "MB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def generate_one(args, prompt, prompt_idx=0, total_prompts=1):
    """Generate images for a single prompt."""
    w, h = map(int, args.size.split("x"))

    prefix = f"[{prompt_idx + 1}/{total_prompts}] " if total_prompts > 1 else ""
    short_prompt = prompt[:60] + ("..." if len(prompt) > 60 else "")

    print(f"{C.CYAN}{C.BOLD}{prefix}Prompt:{C.RESET} {short_prompt}")
    print(f"  {C.DIM}Size: {w}x{h} | Steps: {args.steps} | CFG: {args.cfg} | "
          f"Seed: {args.seed} | Sampler: {args.sampler} | Batch: {args.batch}{C.RESET}")

    start = time.time()
    try:
        images = generate_images(
            api_base=args.api,
            prompt=prompt,
            negative_prompt=args.negative or "",
            width=w,
            height=h,
            steps=args.steps,
            cfg_scale=args.cfg,
            seed=args.seed,
            batch=args.batch,
            sampler=args.sampler,
            output_format=args.format,
        )
    except RuntimeError as e:
        print(f"  {C.RED}✗ Error: {e}{C.RESET}")
        return []

    elapsed = time.time() - start

    if not images:
        print(f"  {C.RED}✗ No images returned{C.RESET}")
        return []

    saved_paths = []
    for i, b64 in enumerate(images):
        name = safe_filename(prompt)
        ts = int(time.time() * 1000)
        suffix = f"_{i + 1}" if len(images) > 1 else ""
        filename = f"{name}{suffix}_{ts}.{args.format}"
        filepath = Path(args.output) / filename

        size = save_image(b64, filepath, args.format)
        saved_paths.append(str(filepath))
        print(f"  {C.GREEN}✓ Saved:{C.RESET} {filepath} ({format_size(size)})")

    print(f"  {C.DIM}Generated {len(images)} image(s) in {elapsed:.1f}s{C.RESET}")
    print()

    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        prog="t2i",
        description="Z-Image Studio CLI — Generate images from text prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "a cute cat in a garden"
  %(prog)s "sunset over mountains" --size 1024x768 --steps 6
  %(prog)s --file prompts.txt --batch 2 -o ./output/
  %(prog)s "a robot" --json --api http://localhost:23530
        """,
    )

    parser.add_argument("prompt", nargs="?", help="Text prompt for image generation")
    parser.add_argument("--file", "-f", help="File with prompts (one per line)")
    parser.add_argument("--output", "-o", default="./generated", help="Output directory (default: ./generated)")
    parser.add_argument("--size", "-s", default=DEFAULT_SIZE, help=f"Image size WxH (default: {DEFAULT_SIZE})")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"Sampling steps (default: {DEFAULT_STEPS})")
    parser.add_argument("--cfg", type=float, default=DEFAULT_CFG, help=f"CFG scale (default: {DEFAULT_CFG})")
    parser.add_argument("--seed", type=int, default=-1, help="RNG seed (-1 for random)")
    parser.add_argument("--batch", "-b", type=int, default=1, help="Number of images per prompt (1-4)")
    parser.add_argument("--sampler", default=DEFAULT_SAMPLER,
                        choices=["euler", "euler_a", "dpm++2m", "dpm++2s_a", "lcm", "heun"],
                        help=f"Sampling method (default: {DEFAULT_SAMPLER})")
    parser.add_argument("--negative", "-n", default="", help="Negative prompt")
    parser.add_argument("--format", default=DEFAULT_FORMAT, choices=["png", "jpeg"], help="Output format")
    parser.add_argument("--api", default=DEFAULT_API, help=f"API base URL (default: {DEFAULT_API})")
    parser.add_argument("--json", action="store_true", help="Output JSON with base64 data (no file saving)")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        C.disable()

    # Validate
    if not args.prompt and not args.file:
        parser.error("Either provide a prompt or use --file")

    if "x" not in args.size:
        parser.error("Size must be in WxH format (e.g. 768x768)")

    args.batch = max(1, min(4, args.batch))

    # Collect prompts
    prompts = []
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        except FileNotFoundError:
            print(f"{C.RED}Error: File not found: {args.file}{C.RESET}", file=sys.stderr)
            sys.exit(1)
    if args.prompt:
        prompts.insert(0, args.prompt)

    if not prompts:
        print(f"{C.RED}Error: No prompts to process{C.RESET}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print_banner()

    # Check server
    if not args.quiet:
        print(f"{C.DIM}Checking server: {args.api}{C.RESET}")

    if not check_server(args.api):
        print(f"{C.RED}✗ Server at {args.api} is not reachable{C.RESET}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"{C.GREEN}✓ Server online{C.RESET}")
        print(f"{C.DIM}Output directory: {args.output}{C.RESET}")
        print()

    # JSON mode
    if args.json:
        all_results = []
        for prompt in prompts:
            w, h = map(int, args.size.split("x"))
            try:
                images = generate_images(
                    api_base=args.api, prompt=prompt, negative_prompt=args.negative,
                    width=w, height=h, steps=args.steps, cfg_scale=args.cfg,
                    seed=args.seed, batch=args.batch, sampler=args.sampler,
                    output_format=args.format,
                )
                all_results.append({"prompt": prompt, "images": images, "error": None})
            except RuntimeError as e:
                all_results.append({"prompt": prompt, "images": [], "error": str(e)})
        print(json.dumps(all_results, indent=2))
        return

    # Normal mode
    all_paths = []
    for i, prompt in enumerate(prompts):
        paths = generate_one(args, prompt, i, len(prompts))
        all_paths.extend(paths)

    if not args.quiet and all_paths:
        print(f"{C.GREEN}{C.BOLD}Done!{C.RESET} Generated {len(all_paths)} image(s) total.")
    elif not args.quiet:
        print(f"{C.YELLOW}No images were generated.{C.RESET}")


if __name__ == "__main__":
    main()
