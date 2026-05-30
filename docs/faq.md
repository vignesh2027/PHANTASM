# FAQ

## Does PHANTASM require retraining the base model?

No. PHANTASM is entirely post-hoc and inference-time. It wraps any existing HuggingFace causal LM without modifying its weights.

## Which models are supported?

Any HuggingFace `AutoModelForCausalLM`-compatible model: GPT-2, GPT-J, LLaMA, Falcon, Mistral, Phi, Gemma, and others.

## How fast is HGT?

HGT requires one forward-backward pass (gradient computation). On a consumer GPU, this takes roughly 1.5–2× the time of a standard forward pass. For production use, consider caching `CompetencyAtlas` results or using the offline `score_hallucination_risk()` function.

## Does UC change the model's outputs?

No. UC only calibrates the confidence score and produces a reliability tier. The actual token probabilities and generated text are unchanged.

## How many MC-Dropout samples does UC need?

20–30 samples give stable estimates for most use cases. Reduce to 10 for latency-sensitive production; increase to 50+ for high-stakes medical or legal applications.

## Can I use individual pillars without the full pipeline?

Yes. HGT, CMN, and UC are fully independent modules. Import them from `phantasm.core.hgt`, `phantasm.core.cmn`, and `phantasm.core.uc` respectively.

## How do I cite PHANTASM?

```bibtex
@software{phantasm2026,
  author  = {Vignesh S},
  title   = {PHANTASM: Probabilistic Hallucination-Aware Neural Transformation with Adaptive Synthesis Method},
  year    = {2026},
  url     = {https://github.com/vignesh2027/PHANTASM},
  license = {Apache-2.0}
}
```

## Who built PHANTASM?

PHANTASM was designed and built by **Vignesh S**, B.Tech CSE student at Takshashila University, Chennai, India.
GitHub: [vignesh2027](https://github.com/vignesh2027)
