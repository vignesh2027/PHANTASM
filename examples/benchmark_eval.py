"""
Run the PHANTASM benchmark on phantasm_synthetic data.
Run:
    python examples/benchmark_eval.py
"""

from phantasm import PHANTASMPipeline
from phantasm.datasets.loader import PHANTASMBenchmark
from phantasm.utils.evaluation import PHANTASMEvaluator
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

pipeline = PHANTASMPipeline.from_pretrained("gpt2")

# Benchmark via PHANTASMBenchmark
bench = PHANTASMBenchmark(pipeline, tokenizer)
results = bench.run("phantasm_synthetic", max_samples=10)
PHANTASMBenchmark.print_report(results)

# Full evaluation with labels
samples = [
    {"text": "The Eiffel Tower was built in 1492 by Napoleon.", "label": 1.0},
    {"text": "Python was created by Guido van Rossum in 1991.", "label": 0.0},
    {"text": "Water boils at 50°C at sea level.", "label": 1.0},
    {"text": "DNA stands for Deoxyribonucleic Acid.", "label": 0.0},
]

evaluator = PHANTASMEvaluator(pipeline)
metrics = evaluator.evaluate(samples)
PHANTASMEvaluator.print_summary(metrics)
