"""
PHANTASM Dataset Loader
========================
Integrates HuggingFace Datasets Hub with PHANTASM's three pillars.

Supported source datasets:
  - vectara/hallucinated-faithfulness-benchmark  (HGT)
  - potsawee/wiki_bio_gpt3_hallucination         (HGT + UC)
  - lmsys/toxic-chat                             (UC)
  - Vectara/FaithDial                            (CMN)
  - Custom PHANTASM synthetic benchmark          (all pillars)

Author: Vignesh S
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

PHANTASM_DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "hallucination_bench": {
        "hf_path": "vectara/hallucinated-faithfulness-benchmark",
        "pillar": "hgt",
        "text_column": "generated_story",
        "label_column": "hallucination",
        "description": "1,000 passage-level hallucination annotations from Vectara.",
    },
    "wiki_bio_hallucination": {
        "hf_path": "potsawee/wiki_bio_gpt3_hallucination",
        "pillar": "hgt+uc",
        "text_column": "annotation",
        "label_column": "annotation",
        "description": "GPT-3 Wikipedia bio hallucinations with human annotations.",
    },
    "faith_dial": {
        "hf_path": "Vectara/FaithDial",
        "pillar": "cmn",
        "text_column": "response",
        "label_column": "BEGIN",
        "description": "Faithfulness annotations for dialog — ideal for CMN training.",
    },
    "phantasm_synthetic": {
        "hf_path": None,
        "pillar": "all",
        "description": "PHANTASM-generated synthetic benchmark (offline, no download needed).",
    },
}


# ---------------------------------------------------------------------------
# Core dataset class
# ---------------------------------------------------------------------------

@dataclass
class PHANTASMExample:
    """Single training / evaluation example for PHANTASM."""
    text: str
    reference: Optional[str]
    label: Optional[Any]
    pillar: str
    metadata: Dict = field(default_factory=dict)


class PHANTASMTorchDataset(Dataset):
    """
    PyTorch Dataset wrapping a list of PHANTASMExample objects.
    Tokenizes on the fly.
    """

    def __init__(
        self,
        examples: List[PHANTASMExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "text": ex.text,
            "pillar": ex.pillar,
        }
        if ex.reference is not None:
            ref_enc = self.tokenizer(
                ex.reference,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item["ref_input_ids"] = ref_enc["input_ids"].squeeze(0)
            item["ref_attention_mask"] = ref_enc["attention_mask"].squeeze(0)
        if ex.label is not None:
            item["label"] = torch.tensor(ex.label, dtype=torch.float32)
        return item


class PHANTASMDatasetLoader:
    """
    Loads, preprocesses, and serves datasets for PHANTASM training.

    Usage
    -----
    >>> loader = PHANTASMDatasetLoader(tokenizer)
    >>> train_ds, val_ds = loader.load("wiki_bio_hallucination", split_ratio=0.9)
    >>> dl = loader.get_dataloader(train_ds, batch_size=16)

    Parameters
    ----------
    tokenizer  : PreTrainedTokenizer
    max_length : int   Maximum token length per example.
    cache_dir  : str   HuggingFace datasets cache directory.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir

    def load(
        self,
        dataset_name: str = "phantasm_synthetic",
        split_ratio: float = 0.9,
        max_samples: Optional[int] = None,
    ) -> Tuple[PHANTASMTorchDataset, PHANTASMTorchDataset]:
        """
        Load a named dataset and return (train_dataset, val_dataset).

        Parameters
        ----------
        dataset_name : str   Key in PHANTASM_DATASET_REGISTRY.
        split_ratio  : float Train fraction (remainder is validation).
        max_samples  : int   Cap total examples (useful for quick experiments).
        """
        if dataset_name not in PHANTASM_DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {list(PHANTASM_DATASET_REGISTRY.keys())}"
            )

        cfg = PHANTASM_DATASET_REGISTRY[dataset_name]

        if cfg["hf_path"] is not None:
            examples = self._load_from_hub(cfg, max_samples)
        else:
            examples = self._load_synthetic(max_samples)

        split_idx = int(len(examples) * split_ratio)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        train_ds = PHANTASMTorchDataset(train_examples, self.tokenizer, self.max_length)
        val_ds = PHANTASMTorchDataset(val_examples, self.tokenizer, self.max_length)
        return train_ds, val_ds

    def get_dataloader(
        self,
        dataset: PHANTASMTorchDataset,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_from_hub(
        self,
        cfg: Dict[str, Any],
        max_samples: Optional[int],
    ) -> List[PHANTASMExample]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "Install `datasets` with: pip install datasets"
            ) from e

        raw = load_dataset(cfg["hf_path"], split="train", cache_dir=self.cache_dir)
        if max_samples:
            raw = raw.select(range(min(max_samples, len(raw))))

        examples: List[PHANTASMExample] = []
        text_col = cfg.get("text_column", "text")
        label_col = cfg.get("label_column", None)

        for row in raw:
            text = str(row.get(text_col, ""))
            if not text.strip():
                continue
            label_raw = row.get(label_col, None) if label_col else None
            label = self._parse_label(label_raw)
            examples.append(
                PHANTASMExample(
                    text=text,
                    reference=None,
                    label=label,
                    pillar=cfg["pillar"],
                )
            )
        return examples

    def _load_synthetic(self, max_samples: Optional[int]) -> List[PHANTASMExample]:
        """Returns a small hand-crafted synthetic benchmark (no download)."""
        raw_pairs = [
            (
                "The Eiffel Tower was constructed in 1492 by Napoleon Bonaparte as a symbol of French imperialism.",
                "The Eiffel Tower was built in 1889 by Gustave Eiffel for the 1889 World's Fair.",
                1.0,
            ),
            (
                "Water boils at 50 degrees Celsius at sea level under standard conditions.",
                "Water boils at 100 degrees Celsius at sea level under standard atmospheric pressure.",
                1.0,
            ),
            (
                "Albert Einstein won the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.",
                "Albert Einstein won the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.",
                0.0,
            ),
            (
                "The human brain uses approximately 85% of its total capacity at any given moment.",
                "The human brain is active throughout its entirety, debunking the '10% myth'.",
                1.0,
            ),
            (
                "Python was created by Guido van Rossum and first released in 1991.",
                "Python was created by Guido van Rossum and first released in 1991.",
                0.0,
            ),
            (
                "The Amazon River flows through Africa and empties into the Indian Ocean.",
                "The Amazon River flows through South America and empties into the Atlantic Ocean.",
                1.0,
            ),
            (
                "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight.",
                "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight.",
                0.0,
            ),
            (
                "Shakespeare was born in Stratford-upon-Avon in 1564 and died in 1616.",
                "Shakespeare was born in Stratford-upon-Avon in 1564 and died in 1616.",
                0.0,
            ),
            (
                "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
                "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
                0.0,
            ),
            (
                "DNA stands for Deoxyribonucleic Acid and carries genetic instructions.",
                "DNA stands for Deoxyribonucleic Acid and carries genetic instructions.",
                0.0,
            ),
        ]

        examples: List[PHANTASMExample] = []
        for text, ref, label in raw_pairs:
            examples.append(
                PHANTASMExample(
                    text=text,
                    reference=ref,
                    label=label,
                    pillar="all",
                    metadata={"source": "phantasm_synthetic"},
                )
            )

        if max_samples:
            examples = examples[:max_samples]
        return examples

    @staticmethod
    def _parse_label(raw: Any) -> Optional[float]:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            mapping = {"yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0,
                       "hallucinated": 1.0, "faithful": 0.0}
            return mapping.get(raw.lower(), None)
        return None


# ---------------------------------------------------------------------------
# Benchmark evaluation helper
# ---------------------------------------------------------------------------

class PHANTASMBenchmark:
    """
    Standardised benchmark evaluation for all three PHANTASM pillars.

    Usage
    -----
    >>> bench = PHANTASMBenchmark(pipeline, tokenizer)
    >>> results = bench.run(dataset_name="phantasm_synthetic")
    >>> bench.print_report(results)
    """

    def __init__(self, pipeline: Any, tokenizer: PreTrainedTokenizer) -> None:
        self.pipeline = pipeline
        self.tokenizer = tokenizer

    def run(
        self,
        dataset_name: str = "phantasm_synthetic",
        max_samples: int = 50,
    ) -> Dict[str, Any]:
        loader = PHANTASMDatasetLoader(self.tokenizer)
        _, val_ds = loader.load(dataset_name, max_samples=max_samples)

        results: List[Dict] = []
        for i in range(len(val_ds)):
            item = val_ds[i]
            report = self.pipeline.analyze(item["text"])
            results.append(
                {
                    "text": item["text"][:100],
                    "predicted_risk": report.competency_atlas.overall_hallucination_risk,
                    "true_label": item.get("label", None),
                    "tier": report.uncertainty.reliability_tier,
                }
            )

        return self._aggregate(results)

    def _aggregate(self, results: List[Dict]) -> Dict[str, Any]:
        risks = [r["predicted_risk"] for r in results]
        tiers = [r["tier"] for r in results]

        import statistics
        return {
            "n_samples": len(results),
            "mean_hallucination_risk": round(statistics.mean(risks), 4),
            "std_hallucination_risk": round(statistics.stdev(risks) if len(risks) > 1 else 0.0, 4),
            "tier_distribution": {
                "crystal": tiers.count("crystal"),
                "solid": tiers.count("solid"),
                "fluid": tiers.count("fluid"),
                "vapor": tiers.count("vapor"),
            },
            "per_example": results,
        }

    @staticmethod
    def print_report(results: Dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print("  PHANTASM BENCHMARK REPORT")
        print("=" * 60)
        print(f"  Samples evaluated  : {results['n_samples']}")
        print(f"  Mean halluc. risk  : {results['mean_hallucination_risk']:.4f}")
        print(f"  Std  halluc. risk  : {results['std_hallucination_risk']:.4f}")
        print("\n  Reliability Tier Distribution:")
        for tier, count in results["tier_distribution"].items():
            bar = "█" * count
            print(f"    {tier:8s} : {bar} ({count})")
        print("=" * 60 + "\n")
