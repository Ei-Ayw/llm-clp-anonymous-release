"""LLM-based free-form identity counterfactual generation via Ollama."""
from __future__ import annotations

import os
import sys
import time
import argparse
import threading
from pathlib import Path
from typing import Any, Optional, Union
from difflib import SequenceMatcher

import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.paths import ROOT_DIR
from ..counterfactual.prompts import PROMPT_FREE_FORM


def _analyze_changes(original: str, rewritten: str) -> dict[str, Any]:
    """
    Compare the original and rewritten text with a character-level diff.

    Returns edit spans, a normalized edit ratio, and simple word-set deltas
    used for diagnostics.
    """
    matcher = SequenceMatcher(None, original, rewritten)
    edit_spans: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        edit_spans.append({
            'type': tag,  # 'replace' | 'delete' | 'insert'
            'original': original[i1:i2],
            'rewritten': rewritten[j1:j2],
            'original_pos': i1,
            'rewritten_pos': j1,
        })

    orig_words = set(original.lower().split())
    cf_words = set(rewritten.lower().split())
    edit_ratio = 1 - matcher.ratio()

    return {
        'edit_spans': edit_spans,
        'edit_ratio': round(edit_ratio, 4),
        'words_removed': sorted(orig_words - cf_words),
        'words_added': sorted(cf_words - orig_words),
    }


class OllamaGenerator:
    """Ollama API wrapper for the free-form counterfactual prompt."""

    def __init__(self, model: str = "qwen2.5:3b",
                 base_url: str = "http://127.0.0.1:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, text: str) -> Optional[str]:
        """Generate one rewritten text; return ``None`` after repeated failures."""
        prompt = PROMPT_FREE_FORM.format(text=text)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 384,
                            "num_ctx": 2048,
                        },
                    },
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()
                result = (data.get("response") or "").strip()
                result = result.strip('"').strip("'").strip()
                return result or None
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2.0 * (attempt + 1)
                    print(f"  [Ollama Error] {e}, "
                          f"retry {attempt+1}/{max_retries} (wait {wait_time}s)")
                    time.sleep(wait_time)
                else:
                    print(f"  [Ollama Error] {e} (max retries reached)")
                    return None


def generate_counterfactuals_for_dataset(
    df: pd.DataFrame,
    generator: Any,
    text_col: str = 'text',
    id_col: Optional[str] = 'post_id',
    num_cf_per_sample: int = 2,
    save_path: Optional[Union[str, Path]] = None,
    resume: bool = True,
    max_workers: int = 20,
) -> pd.DataFrame:
    """
    Generate LLM counterfactuals for a DataFrame.

    The generator keeps the filtering deliberately light: each example receives
    up to ``num_cf_per_sample`` generations, and character-level edit metadata
    is recorded for diagnostics.
    """
    # Resume from partial output when requested.
    existing = set()
    existing_records: list[dict[str, Any]] = []
    if resume and save_path and os.path.exists(save_path):
        existing_df = pd.read_parquet(save_path)
        for _, row in existing_df.iterrows():
            key = f"{row['post_id']}_{row.get('cf_index', 0)}"
            existing.add(key)
            existing_records.append(row.to_dict())
        print(f"  [Resume] {len(existing)} existing records")

    # Build generation tasks.
    tasks: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        text = row[text_col]
        pid = row[id_col] if (id_col and id_col in df.columns) \
              else f"sample_{idx}"

        for cf_idx in range(num_cf_per_sample):
            key = f"{pid}_{cf_idx}"
            if key in existing:
                continue
            tasks.append({
                'pid': pid,
                'text': text,
                'cf_index': cf_idx,
                'key': key,
            })

    print(f"  [Tasks] {len(tasks)} to generate "
          f"(skipped {len(existing)} existing)")

    if not tasks:
        print("  [Done] Nothing to generate")
        return pd.DataFrame(existing_records) if existing_records \
               else pd.DataFrame()

    # Generate with a worker pool. The CLI uses one worker by default for Ollama.
    records = list(existing_records)
    failed = 0
    lock = threading.Lock()
    save_lock = threading.Lock()

    def process_one(task: dict[str, Any]) -> Optional[dict[str, Any]]:
        cf_text = generator.generate(task['text'])
        if cf_text and cf_text != task['text'] and len(cf_text) > 5:
            changes = _analyze_changes(task['text'], cf_text)
            return {
                'post_id': task['pid'],
                'original_text': task['text'],
                'cf_text': cf_text,
                'cf_index': task['cf_index'],
                'method': 'llm_free',
                **changes,
            }
        return None

    pbar = tqdm(total=len(tasks), desc="LLM FREE_FORM")
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, t): t for t in tasks}

        for future in as_completed(futures):
            result = future.result()
            with lock:
                if result:
                    records.append(result)
                else:
                    failed += 1
                completed += 1
                pbar.update(1)

                if save_path and len(records) > 0 and completed % 500 == 0:
                    with save_lock:
                        pd.DataFrame(records).to_parquet(save_path,
                                                         index=False)

    pbar.close()

    result_df = pd.DataFrame(records)
    if save_path:
        result_df.to_parquet(save_path, index=False)
        new_count = len(result_df) - len(existing_records)
        print(f"\n  [Done] {new_count} new, {len(result_df)} total "
              f"(failures: {failed})")
        print(f"  Saved to: {save_path}")

    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM counterfactuals (free-form prompt).")
    parser.add_argument("--dataset", type=str, default="hatexplain",
                        choices=["hatexplain", "toxigen", "dynahate"],
                        help="Dataset name")
    parser.add_argument("--split", type=str, default="train",
                        help="Data split: train/val/test")
    parser.add_argument("--num_cf", type=int, default=2,
                        help="Counterfactuals per sample (default: 2)")
    parser.add_argument("--model", type=str, default="qwen2.5:3b",
                        help="Ollama model name")
    parser.add_argument("--base_url", type=str,
                        default="http://127.0.0.1:11434",
                        help="Ollama API base URL")
    args = parser.parse_args()

    # Load data.
    data_dir = str(ROOT_DIR / "data" / "causal_fair")
    data_path = os.path.join(data_dir,
                             f"{args.dataset}_{args.split}.parquet")
    if not os.path.exists(data_path):
        print(f"[Error] File not found: {data_path}")
        print("  Place normalized parquet files under data/causal_fair first.")
        return

    df = pd.read_parquet(data_path)
    print(f"[Data] {args.dataset}/{args.split}: {len(df)} samples")

    # Initialize generator.
    generator = OllamaGenerator(model=args.model, base_url=args.base_url)
    max_workers = 1  # single-thread to avoid overloading Ollama
    print(f"[Generator] {args.model} / {max_workers} workers")

    # Output path.
    save_path = os.path.join(
        data_dir, f"{args.dataset}_{args.split}_cf_llm.parquet"
    )

    # Determine ID column.
    id_col = 'post_id' if 'post_id' in df.columns else None
    if id_col is None:
        df['post_id'] = [f"sample_{i}" for i in range(len(df))]
        id_col = 'post_id'

    # Generate counterfactuals.
    generate_counterfactuals_for_dataset(
        df, generator,
        text_col='text', id_col=id_col,
        num_cf_per_sample=args.num_cf,
        save_path=save_path,
        max_workers=max_workers,
    )


if __name__ == "__main__":
    main()
