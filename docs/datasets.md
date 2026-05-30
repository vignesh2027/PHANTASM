# Datasets

PHANTASM integrates with HuggingFace Datasets Hub out of the box.

## Supported datasets

| Name | HF Path | Pillar | Size |
|---|---|---|---|
| `hallucination_bench` | `vectara/hallucinated-faithfulness-benchmark` | HGT | 1K |
| `wiki_bio_hallucination` | `potsawee/wiki_bio_gpt3_hallucination` | HGT + UC | 7.8K |
| `faith_dial` | `Vectara/FaithDial` | CMN | 36K |
| `phantasm_synthetic` | Built-in (no download) | All | Extensible |

## Loading a dataset

```python
from transformers import AutoTokenizer
from phantasm.datasets.loader import PHANTASMDatasetLoader

tokenizer = AutoTokenizer.from_pretrained("gpt2")
loader = PHANTASMDatasetLoader(tokenizer, max_length=256)

train_ds, val_ds = loader.load("wiki_bio_hallucination", split_ratio=0.9, max_samples=1000)
train_dl = loader.get_dataloader(train_ds, batch_size=16, shuffle=True)
```

## Running the benchmark

```python
from phantasm.datasets.loader import PHANTASMBenchmark

bench = PHANTASMBenchmark(pipeline, tokenizer)
results = bench.run("phantasm_synthetic", max_samples=50)
PHANTASMBenchmark.print_report(results)
```

## Adding custom datasets

Register your dataset in `PHANTASM_DATASET_REGISTRY` in `phantasm/datasets/loader.py`:

```python
PHANTASM_DATASET_REGISTRY["my_dataset"] = {
    "hf_path": "my_org/my_dataset",
    "pillar": "hgt",
    "text_column": "output",
    "label_column": "is_hallucinated",
    "description": "My custom hallucination dataset.",
}
```
