---
license: other
license_name: SUPRA Non-Commercial License (TBD)
license_link: LICENSE
base_model: GSAI-ML/LLaDA-8B-Instruct
tags:
- quantized
- llada
---

# SUPRA: LLaDA-8B-Instruct Model Configuration

This model release contains the shared deployment artifact used by the
GSM8K and HumanEval default configurations of the accompanying algorithm code.
Transformer weights are W4 with one BF16 scale per output row, activations
are per-row A4/A8, Attention is Q8/K8/P8/V8, and the LM head is W8.

`artifact_w4/`, `artifact_w8_head/` and `silu_table_monotone.json` form one
deployment model. Manifests describe numerical formats, tensor locations and
model identity. `model_metadata/` contains the configuration, tokenizer and
weight index used for source identification. Load this artifact format with the accompanying
`evaluation.model.QuantizedLLaDALM` implementation.

Calibration uses WikiText-2 train, GSM8K train and decontaminated Personahub
code training records. Task configuration files define the default execution settings. Artifacts
produced by calibration can be evaluated with the accompanying evaluation framework.

The base model is GSAI-ML/LLaDA-8B-Instruct at revision
`08b83a6feb34df1a6011b80c3c00c7563e963b07`. Its
[upstream model card](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/blob/08b83a6feb34df1a6011b80c3c00c7563e963b07/README.md)
declares MIT. The base model, tokenizer and configuration retain their upstream
terms. The source model's card and any supplied license and notice files are
preserved in `upstream/`.
SUPRA contributions use the provisional Non-Commercial License (final terms TBD), with Fast-dLLM/LLaDA
attribution. See the accompanying code's README, CALIBRATION and UPSTREAM
documents for execution and scoring details.
