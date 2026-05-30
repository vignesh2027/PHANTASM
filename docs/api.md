# API Reference

## PHANTASMPipeline

```python
class PHANTASMPipeline:
    @classmethod
    def from_pretrained(cls, model_name: str, device: str = "cpu", **kwargs) -> PHANTASMPipeline
    def analyze(self, text: str, reference_text: str = None, domain: str = "general") -> PHANTASMReport
```

## PHANTASMReport

```python
@dataclass
class PHANTASMReport:
    competency_atlas: CompetencyAtlas
    mined_hypotheses: List[Hypothesis]
    uncertainty: CrystalizedUncertainty
    synthesis_summary: str
    actionable_insights: List[str]

    def to_dict(self) -> dict
```

## HallucinationGradientTracer

```python
class HallucinationGradientTracer:
    def __init__(self, model, tokenizer, threshold=0.35, device="cpu")
    def trace(self, text: str) -> CompetencyAtlas
    def batch_trace(self, texts: List[str]) -> List[CompetencyAtlas]

def score_hallucination_risk(generation, reference=None, method="entropy") -> float
```

## ConfabulationMiningNetwork

```python
class ConfabulationMiningNetwork(nn.Module):
    def __init__(self, vocab_size=50257, hidden_dim=256, novelty_threshold=0.45, plausibility_threshold=0.50)
    def forward(self, confab_ids, fact_ids, ...) -> Tuple[Tensor, Tensor]
    def mine(self, confab_ids, fact_ids, texts, domain="general") -> List[Hypothesis]
```

## UncertaintyCrystallizer

```python
class UncertaintyCrystallizer:
    def __init__(self, model, temperature=1.5, mc_samples=20, coverage=0.90, device="cpu")
    def crystallize(self, input_ids, attention_mask=None) -> CrystalizedUncertainty
    def batch_crystallize(self, input_ids_list, ...) -> List[CrystalizedUncertainty]
```

## PHANTASMDatasetLoader

```python
class PHANTASMDatasetLoader:
    def __init__(self, tokenizer, max_length=256, cache_dir=None)
    def load(self, dataset_name, split_ratio=0.9, max_samples=None) -> Tuple[Dataset, Dataset]
    def get_dataloader(self, dataset, batch_size=16, shuffle=True) -> DataLoader
```

## PHANTASMTrainer

```python
class PHANTASMTrainer:
    def __init__(self, model, train_loader, val_loader, config, callbacks=None)
    def train(self) -> Dict[str, List[float]]
    def save_history(self, path: str)
```

## PHANTASMMetrics

```python
class PHANTASMMetrics:
    @staticmethod
    def hgt_metrics(y_true, y_pred, threshold=0.5) -> dict
    @staticmethod
    def cmn_metrics(hypotheses_per_query, k=5) -> dict
    @staticmethod
    def uc_metrics(confidences, correct, n_bins=10) -> dict
    @staticmethod
    def phantasm_score(hgt_f1, cmn_novelty, uc_ece) -> float
```
