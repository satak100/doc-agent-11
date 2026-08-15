#!/usr/bin/env python3
"""Fast, resumable PaddleOCR-VL 1.6 text-spotting baseline.

Designed for mixed Bengali/English photographed prescriptions on a single GPU.
The output is raw model text with spatial coordinates; it is not clinically
validated and must not be used to make medication decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from transformers import AutoModelForImageTextToText, AutoProcessor


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
SPOTTING_MAX_PIXELS = 2048 * 28 * 28
UPSCALE_THRESHOLD = 1500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--only-truncated-from",
        type=Path,
        help="Process only paths marked hit_max_new_tokens in another results.jsonl",
    )
    return parser.parse_args()


def image_paths(root: Path) -> list[Path]:
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    # Nearby dimensions reduce padding and improve batched throughput.
    sized: list[tuple[int, int, str, Path]] = []
    for path in paths:
        try:
            with Image.open(path) as im:
                w, h = im.size
            sized.append((max(w, h), w * h, str(path), path))
        except Exception as exc:
            print(f"SKIP unreadable {path}: {exc}", flush=True)
    return [item[-1] for item in sorted(sized)]


def load_image(path: Path) -> tuple[Image.Image, tuple[int, int]]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    original_size = image.size
    if image.width < UPSCALE_THRESHOLD and image.height < UPSCALE_THRESHOLD:
        image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
    return image, original_size


def already_done(jsonl_path: Path) -> set[str]:
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                if record.get("status") == "ok":
                    done.add(record["relative_path"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def make_inputs(processor: Any, images: list[Image.Image], device: torch.device) -> Any:
    image_processor = processor.image_processor
    shortest_edge = getattr(image_processor, "min_pixels", None)
    if shortest_edge is None:
        # Transformers 5.14 exposes the same value via SizeDict.
        shortest_edge = image_processor.size.shortest_edge
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Spotting:"},
                ],
            }
        ]
        for image in images
    ]
    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": shortest_edge,
                "longest_edge": SPOTTING_MAX_PIXELS,
            }
        },
    )
    return inputs.to(device)


def infer_batch(
    model: Any,
    processor: Any,
    device: torch.device,
    paths: list[Path],
    input_root: Path,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    images: list[Image.Image] = []
    original_sizes: list[tuple[int, int]] = []
    for path in paths:
        image, size = load_image(path)
        images.append(image)
        original_sizes.append(size)

    started = time.perf_counter()
    inputs = make_inputs(processor, images, device)
    prompt_width = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_new_tokens,
        )
    new_token_ids = generated[:, prompt_width:]
    texts = processor.batch_decode(new_token_ids, skip_special_tokens=True)
    eos_id = processor.tokenizer.eos_token_id
    generated_lengths = []
    for token_row in new_token_ids.tolist():
        try:
            generated_lengths.append(token_row.index(eos_id) + 1)
        except ValueError:
            generated_lengths.append(len(token_row))
    elapsed = time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)

    records = []
    for path, size, text, generated_length in zip(
        paths, original_sizes, texts, generated_lengths
    ):
        stat = path.stat()
        records.append(
            {
                "status": "ok",
                "relative_path": str(path.relative_to(input_root)),
                "width": size[0],
                "height": size[1],
                "bytes": stat.st_size,
                "task": "spotting",
                "model": MODEL_ID,
                "text": text.strip(),
                "generated_tokens": generated_length,
                "hit_max_new_tokens": generated_length >= max_new_tokens,
                "batch_seconds": round(elapsed, 4),
                "batch_items": len(paths),
                "seconds_per_image": round(elapsed / len(paths), 4),
                "peak_gpu_mib": round(peak_mib, 1),
            }
        )
    return records


def run_with_oom_fallback(
    model: Any,
    processor: Any,
    device: torch.device,
    paths: list[Path],
    input_root: Path,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    try:
        return infer_batch(model, processor, device, paths, input_root, max_new_tokens)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(paths) == 1:
            raise
        midpoint = len(paths) // 2
        return run_with_oom_fallback(
            model, processor, device, paths[:midpoint], input_root, max_new_tokens
        ) + run_with_oom_fallback(
            model, processor, device, paths[midpoint:], input_root, max_new_tokens
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this optimized runner")

    args.output.mkdir(parents=True, exist_ok=True)
    per_image_dir = args.output / "json"
    per_image_dir.mkdir(exist_ok=True)
    jsonl_path = args.output / "results.jsonl"
    done = already_done(jsonl_path)

    paths = image_paths(args.input)
    if args.only_truncated_from:
        truncated: set[str] = set()
        with args.only_truncated_from.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("hit_max_new_tokens") is True:
                    truncated.add(record["relative_path"])
        paths = [p for p in paths if str(p.relative_to(args.input)) in truncated]
        print(f"selected_truncated_images={len(paths)}", flush=True)
    paths = [p for p in paths if str(p.relative_to(args.input)) not in done]
    if args.limit:
        paths = paths[: args.limit]
    print(f"pending_images={len(paths)} already_completed={len(done)}", flush=True)
    if not paths:
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    print(f"model_loaded={args.model} device={torch.cuda.get_device_name(0)}", flush=True)

    total_started = time.perf_counter()
    completed = 0
    errors = 0
    with jsonl_path.open("a", encoding="utf-8", buffering=1) as jsonl:
        for offset in range(0, len(paths), args.batch_size):
            batch = paths[offset : offset + args.batch_size]
            torch.cuda.reset_peak_memory_stats(device)
            try:
                records = run_with_oom_fallback(
                    model, processor, device, batch, args.input, args.max_new_tokens
                )
            except Exception as exc:
                records = [
                    {
                        "status": "error",
                        "relative_path": str(path.relative_to(args.input)),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    for path in batch
                ]

            for record in records:
                jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                safe_name = record["relative_path"].replace("/", "__") + ".json"
                with (per_image_dir / safe_name).open("w", encoding="utf-8") as handle:
                    json.dump(record, handle, ensure_ascii=False, indent=2)
                if record["status"] == "ok":
                    completed += 1
                else:
                    errors += 1

            elapsed = time.perf_counter() - total_started
            rate = completed / elapsed if elapsed else 0.0
            print(
                f"completed={completed}/{len(paths)} elapsed_s={elapsed:.1f} "
                f"images_per_s={rate:.3f}",
                flush=True,
            )

    summary = {
        "model": args.model,
        "task": "spotting",
        "newly_completed": completed,
        "errors": errors,
        "previously_completed": len(done),
        "elapsed_seconds": round(time.perf_counter() - total_started, 3),
        "gpu": torch.cuda.get_device_name(0),
    }
    with (args.output / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    # Reduces allocator fragmentation for variable-sized document images.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
