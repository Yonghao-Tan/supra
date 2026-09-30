# Source And Dependency Attribution

The LLaDA model implementation and configuration retain NVIDIA's Apache-2.0
headers from [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM), whose
headers attribute the original model to [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA).
The project adds quantized operators, cache/state generation, calibration,
capture/replay, explicit deployment configuration and evaluation integration.

The Fast-dLLM source reference is commit
`a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b`. Model and generator modifications
are maintained in this repository. Original copyright and license headers are retained alongside the
upstream Apache-2.0 text in LICENSE-APACHE-2.0. Research using this implementation should cite
Fast-dLLM and LLaDA using their upstream citation information.

The evaluation environment uses Python 3.9. GSM scoring uses lm_eval 0.4.8's
regex filters and exact_match implementation.

Quantization uses fixed R1/R2/R4 rotations following
[SpinQuant](https://arxiv.org/abs/2405.16406) and
[GPTQ](https://arxiv.org/abs/2210.17323) weight quantization with one BF16 scale
per output row. The implementations are in `llada/quantization/rotation.py`
and `llada/quantization/gptq.py`.

Initial calibration loads `Salesforce/wikitext:wikitext-2-raw-v1:train`
at revision `b08601e04326c79dfdd32d625aee71d232d685c3`, and records that revision
and the loaded dataset fingerprint. WikiText's dataset
card lists CC-BY-SA-3.0 and GFDL.

The base model revision is
`GSAI-ML/LLaDA-8B-Instruct@08b83a6feb34df1a6011b80c3c00c7563e963b07`, whose
model card declares MIT. Model weights, tokenizer and datasets retain their
respective licenses.
