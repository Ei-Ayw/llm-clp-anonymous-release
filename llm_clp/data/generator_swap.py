"""Deterministic lexical-swap counterfactual generation."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# Identity substitutions grouped by broad demographic dimension.
IDENTITY_SWAP_GROUPS: dict[str, dict[str, list[str]]] = {
    'race': {
        'black': ['white', 'asian', 'hispanic'],
        'white': ['black', 'asian', 'hispanic'],
        'african': ['european', 'asian'],
        'asian': ['african', 'european'],
        'hispanic': ['black', 'white'],
    },
    'religion': {
        'muslim': ['christian', 'jewish', 'buddhist'],
        'christian': ['muslim', 'jewish', 'buddhist'],
        'jewish': ['muslim', 'christian'],
        'islam': ['christianity', 'judaism'],
        'islamic': ['christian', 'jewish'],
        'mosque': ['church', 'synagogue', 'temple'],
        'church': ['mosque', 'synagogue', 'temple'],
        'quran': ['bible', 'torah'],
        'bible': ['quran', 'torah'],
    },
    'gender': {
        'women': ['men'],
        'men': ['women'],
        'woman': ['man'],
        'man': ['woman'],
        'she': ['he'],
        'he': ['she'],
        'her': ['his'],
        'his': ['her'],
        'mother': ['father'],
        'father': ['mother'],
    },
    'sexual_orientation': {
        'gay': ['straight', 'heterosexual'],
        'lesbian': ['straight', 'heterosexual'],
        'homosexual': ['heterosexual'],
        'lgbtq': ['heterosexual'],
        'queer': ['straight'],
    },
    'disability': {
        'disabled': ['abled', 'healthy'],
        'mental illness': ['physical health'],
        'mentally ill': ['physically healthy'],
    },
}

# Flattened mapping: source_word -> target_words.
FLAT_SWAP_MAP: dict[str, list[str]] = {}
for category, group_map in IDENTITY_SWAP_GROUPS.items():
    for source, targets in group_map.items():
        FLAT_SWAP_MAP[source] = targets


def _replace_with_case_handling(text: str, source_word: str, target_word: str) -> str:
    """Replace a term with word-boundary matching while preserving case."""
    pattern = re.compile(r'\b' + re.escape(source_word) + r'\b', flags=re.IGNORECASE)

    def repl(match: re.Match) -> str:
        matched = match.group(0)
        if matched.isupper():
            return target_word.upper()
        if matched[0].isupper():
            return target_word.capitalize()
        return target_word

    return pattern.sub(repl, text)


def generate_swap_counterfactuals(
    text: str,
    max_cf: int = 3,
) -> list[dict[str, Any]]:
    """
    Generate lexical-swap counterfactuals with word-boundary matching.
    """
    results: list[dict[str, Any]] = []

    for source_word, target_words in FLAT_SWAP_MAP.items():
        pattern = re.compile(r'\b' + re.escape(source_word) + r'\b', flags=re.IGNORECASE)
        if pattern.search(text):
            for target_word in target_words[:max_cf]:
                cf_text = _replace_with_case_handling(text, source_word, target_word)

                if cf_text != text:
                    results.append({
                        'cf_text': cf_text,
                        'source_word': source_word,
                        'target_word': target_word,
                        'method': 'swap',
                    })

    return results[:max_cf]


def batch_generate_swap(
    texts: list[str],
    post_ids: Optional[list[str | int]] = None,
    max_cf_per_sample: int = 3,
) -> pd.DataFrame:
    """
    Generate swap counterfactuals for a list of texts.
    """
    records: list[dict[str, Any]] = []
    for i, text in enumerate(texts):
        pid = post_ids[i] if post_ids is not None else f"sample_{i}"
        cfs = generate_swap_counterfactuals(text, max_cf=max_cf_per_sample)
        for cf in cfs:
            records.append({
                'post_id': pid,
                'original_text': text,
                **cf,
            })

    return pd.DataFrame(records)


def generate_swap_for_split(
    dataset: str,
    split: str = "train",
    data_dir: Path = Path("data/causal_fair"),
    max_cf_per_sample: int = 1,
) -> pd.DataFrame:
    """
    Generate swap counterfactuals for one dataset split.
    """
    input_path = data_dir / f"{dataset}_{split}.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path)
    texts = df["text"].tolist()
    post_ids = df.get("post_id", None)

    num_identity_terms = sum(len(targets) for targets in FLAT_SWAP_MAP.values())
    print(f"Generating swap counterfactuals for {dataset}_{split}...")
    print(f"  Input samples: {len(df)}")
    print(f"  Swap protocol: {len(FLAT_SWAP_MAP)} identity terms, {num_identity_terms} total swap pairs")

    cf_df = batch_generate_swap(
        texts,
        post_ids if post_ids is not None else None,
        max_cf_per_sample,
    )

    coverage = cf_df['post_id'].nunique() / len(df) * 100 if len(df) > 0 else 0
    print(f"  Generated CFs: {len(cf_df)}")
    print(f"  Samples with CF: {cf_df['post_id'].nunique()} / {len(df)}")
    print(f"  Coverage: {coverage:.1f}%")
    if coverage < 100:
        print(f"  Note: {len(df) - cf_df['post_id'].nunique()} samples have no matching identity terms in swap protocol")
    return cf_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate swap counterfactuals (CDA-Swap baseline)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hatexplain",
        choices=["hatexplain", "toxigen", "dynahate"],
        help="Dataset name (default: hatexplain)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Data split to generate for (default: train)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/causal_fair",
        help="Directory containing parquet files (default: data/causal_fair)",
    )
    parser.add_argument(
        "--max_cf_per_sample",
        type=int,
        default=1,
        help="Max counterfactuals per sample (default: 1)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output parquet path (default: {dataset}_{split}_cf_swap.parquet in data_dir)",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if args.output_path is None:
        output_path = data_dir / f"{args.dataset}_{args.split}_cf_swap.parquet"
    else:
        output_path = Path(args.output_path)

    output_path.parent.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("Swap Counterfactual Generator (CDA-Swap Baseline)")
    print("=" * 80)
    print(f"Protocol: IDENTITY_SWAP_GROUPS ({len(FLAT_SWAP_MAP)} identity terms)")
    print(f"Categories: {', '.join(IDENTITY_SWAP_GROUPS.keys())}")
    print("=" * 80)

    cf_df = generate_swap_for_split(
        dataset=args.dataset,
        split=args.split,
        data_dir=data_dir,
        max_cf_per_sample=args.max_cf_per_sample,
    )

    print(f"\nSaving to: {output_path}")
    cf_df.to_parquet(output_path, index=False)

    print("\nDone!")
    print(f"Next step: You can now use this for Davani baseline or ablation experiments.")


if __name__ == "__main__":
    main()
