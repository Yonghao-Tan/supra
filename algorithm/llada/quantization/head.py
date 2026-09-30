"""Export sample-free, per-output-row W8 LM head weights."""

import argparse
import json
from pathlib import Path
import torch
from quantization.artifact import (
    LM_HEAD,
    SpinQuantArtifactReader,
    SpinQuantLMHeadArtifactWriter,
)
from quantization.numeric import quantize_symmetric_w8


def export_rtn_head(args: argparse.Namespace, *, weight_bits: int) -> None:
    from quantization.artifact import require_train_only_artifact

    if weight_bits != 8:
        raise ValueError("head RTN requires W8")
    parent = SpinQuantArtifactReader(args.parent_artifact, verify_files=False)
    require_train_only_artifact(parent)
    if parent.head_quantized:
        raise ValueError("head RTN requires the original rotated BF16 head parent")
    source = parent.read_bf16(LM_HEAD)
    if not bool(torch.isfinite(source).all()):
        raise ValueError("head RTN requires finite source weights")
    quantized = quantize_symmetric_w8(source)
    writer = SpinQuantLMHeadArtifactWriter(
        args.parent_artifact,
        args.output_artifact,
        calibration=dict(train_only_calibration=True, method="weight_only_no_samples"),
        quantization=dict(
            algorithm=f"symmetric_per_output_row_w{weight_bits}",
            bits=weight_bits,
            sym=True,
            signed_integer_range=[-127, 127],
            signed_zero_point=0,
            qzeros=False,
            group_size=-1,
            scale="bf16_per_output_row_max_abs_div_127",
            rounding="round_to_nearest_even",
            activation_bits=8,
            activation_scale="dynamic_bf16_per_row_max_abs_div_127",
            accumulator="int32",
            arithmetic_id="gminus1_bf16_dual_scale/v1",
            artifact_selection_metric=f"single_predeclared_w{weight_bits}_format_no_selection",
        ),
    )
    try:
        manifest = writer.write(
            quantized,
            quantization_statistics=dict(method="RTN"),
        )
    except Exception:
        writer.abort()
        raise
    print(
        json.dumps(
            dict(
                output_artifact=str(args.output_artifact.resolve()),
                total_file_bytes=manifest["total_file_bytes"],
                weight_bits=weight_bits,
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export-rtn-w8",))
    parser.add_argument("--parent-artifact", type=Path, required=True)
    parser.add_argument("--output-artifact", type=Path, required=True)
    args = parser.parse_args()
    export_rtn_head(args, weight_bits=8)


if __name__ == "__main__":
    main()
