#!/usr/bin/env python3
"""Build a small, model-free set of hardware numerical and protocol checks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from artifact_paths import artifact_directory, artifact_root
from simulation_runtime import DEFAULT_VERILATOR, RTL_ROOT, run_command, runtime_resource_check
from run_testcase import silu_defines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--list-cases", action="store_true",
                        help="list focused checks without building or creating run files")
    parser.add_argument("--verilator", type=Path, default=DEFAULT_VERILATOR)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--feature2-reference", type=Path, action="append",
                        help="run PSME against captured generator inputs and results")
    parser.add_argument("--feature1-reference", type=Path, action="append",
                        help="run ATSE against captured dependency, history and selection")
    parser.add_argument("--boundary-reference", type=Path, action="append",
                        help="run focused ATSE against boundary selection history and score inputs")
    parser.add_argument("--precision-reference", type=Path, action="append",
                        help="run ATSE precision allocation against captured forwards")
    case_option = parser.add_argument("--case", action="append", choices=("numeric", "pe", "quantizer", "sram", "stream", "hidden-transfer", "softmax", "matmul-activation", "matmul-output", "elementwise", "attention-context-writer", "axi-master", "ddr-model", "layer-loader", "token-loader", "psme", "atse", "dependency", "invalidation", "refresh-score", "sequencer", "testcase-checker"))
    args = parser.parse_args()
    selected = args.case or ["numeric", "pe", "quantizer", "sram", "layer-loader", "testcase-checker"]
    if args.list_cases:
        for name in case_option.choices:
            print(name + (" (default)" if name in selected else ""))
        return
    if not args.run_id:
        parser.error("--run-id is required when running checks")
    if args.jobs < 1:
        parser.error("jobs must be positive")
    for name in ("feature2_reference", "feature1_reference", "boundary_reference", "precision_reference"):
        paths = getattr(args, name)
        if paths:
            try:
                setattr(args, name, [path.resolve(strict=True) for path in paths])
            except FileNotFoundError as error:
                parser.error(str(error))
    if args.feature2_reference and "psme" not in selected:
        parser.error("--feature2-reference requires --case psme")
    if args.feature1_reference and "atse" not in selected:
        parser.error("--feature1-reference requires --case atse")
    if args.boundary_reference and ("atse" not in selected or args.feature1_reference):
        parser.error("--boundary-reference requires --case atse and excludes --feature1-reference")
    if args.precision_reference and ("atse" not in selected or args.feature1_reference or args.boundary_reference):
        parser.error("--precision-reference requires --case atse and excludes other ATSE references")
    needs_rtl = bool(set(selected) - {"numeric", "testcase-checker"})
    tool = args.verilator.resolve(strict=True) if needs_rtl else None
    run = artifact_directory("unit_tests", args.run_id)
    if any(run.iterdir()):
        raise FileExistsError(f"use an empty run directory: {run}")
    resource = runtime_resource_check(run, run / "resources.log", 1, 1)
    if resource["status"] != "PASS":
        raise RuntimeError(resource["errors"])
    temporary = run / "tmp"
    temporary.mkdir()
    environment = dict(os.environ, CCACHE_DISABLE="1", TMPDIR=str(temporary),
                       SUPRA_ARTIFACT_ROOT=str(artifact_root()))
    results = []

    def command(name, argv, required=""):
        results.append(run_command(name, list(map(str, argv)), run / f"{name}.log",
                                   environment=environment, required_text=required, stream_output=True))

    if needs_rtl:
        command("verilator_version", [tool, "--version"], "Verilator 5.050")
    numeric = run / "rtl_numeric.o"
    numeric_defines = silu_defines({})
    command("numeric_build", ["gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                              "-ffp-contract=off", *numeric_defines, "-c", "cmodel/rtl_numeric.c", "-o", numeric])
    if "numeric" in selected:
        binary = run / "numeric_test"
        command("numeric_test_build", ["gcc", "-std=c11", "-O2", "cmodel/test_rtl_numeric.c", numeric,
                                        "-lm", "-o", binary])
        command("numeric_test", [binary], "PASS")

    if "testcase-checker" in selected:
        for name in ("head_checkpoint_checker", "attention_checkpoint_checker", "testcase_memory_actions", "embedding_read_checker"):
            binary = run / (name + "_test")
            command(name + "_build", ["g++", "-std=c++17", "-O2",
                "-I.", "-Ithird_party/dramsim3/ext/headers", f"tb/{name}_test.cpp", "-o", binary])
            command(name + "_test", [binary, run / (name + "_data")], "PASS")
        post = run / "forward_postprocess_model.o"
        command("handoff_post_build", ["gcc", "-std=c11", "-O2", "-ffp-contract=off", "-c",
                                       "cmodel/forward_postprocess_model.c", "-o", post])
        binary = run / "testcase_handoff_actions_test"
        command("handoff_build", ["g++", "-std=c++17", "-O2", "-I.",
            "-Ithird_party/dramsim3/ext/headers", "tb/testcase_handoff_actions_test.cpp", post, numeric,
            "-lm", "-o", binary])
        command("handoff_test", [binary], "PASS")

    configurations = []
    if "axi-master" in selected:
        configurations.append(("axi_parameter_64", "axi_master",
            ["-GDATA_WIDTH=64", "-GMAX_BURST_BYTES=4096",
             "-CFLAGS", "-DAXI_PARAMETER_CHECK", "rtl/memory/axi_master.sv"], [], []))
        for width in (128, 256):
            configurations.append((f"axi_master_{width}", "axi_master",
                [f"-GDATA_WIDTH={width}", "rtl/memory/axi_master.sv"], [], []))
    if "ddr-model" in selected:
        initial = run / "ddr_initial.bin"
        expected = run / "ddr_expected.bin"
        initial.write_bytes(bytes(range(64)))
        expected.write_bytes(bytes(range(160, 176)) + bytes(range(16, 64)))
        configurations.append(("ddr_model", "axi4_ddr_model",
            ["-GBASE_ADDRESS=0", "-GMEMORY_BYTES=8192", "-GBACKING_BYTES=256",
             "tb/axi4_ddr_model.sv"], [],
            [f"+DDR_INITIAL_IMAGE={initial}", f"+DDR_EXPECTED_IMAGE={expected}"]))
    if "sequencer" in selected:
        configurations.append(("sequencer", "final_output_sequencer",
                               ["-f", "filelists/production.f"], [], []))
    if "dependency" in selected:
        configurations.append(("dependency", "attention_dependency_update",
                               ["-f", "filelists/production.f"], [numeric], []))
    if "pe" in selected:
        configurations.append(("pe", "mixed_precision_pe", ["-f", "filelists/pe.f"], [], []))
    if "quantizer" in selected:
        configurations.append(("quantizer", "activation_quantizer",
                               ["rtl/compute/quant/activation_quantizer.sv", "rtl/compute/bf16/bf16_mul_pipe.sv"],
                               [numeric], []))
    if "sram" in selected:
        for depth in (1024, 2048):
            configurations.append((f"sram{depth}", "local_sram_macro",
                                   [f"-GDEPTH={depth}", "rtl/memory/local_sram_macro.sv", "sim/memory/local_sram_array.sv"],
                                   [], [depth]))
    if "layer-loader" in selected:
        configurations.append(("layer_loader", "layer_address_loader", ["-f", "filelists/production.f"], [], []))
    if "token-loader" in selected:
        configurations.append(("token_loader", "token_issue_embedding_top", ["-f", "filelists/production.f",
                               "tb/harness/token_issue_embedding_top.sv"], [], [81]))
    if "stream" in selected:
        configurations.append(("stream", "memory_stream_width_adapter",
                               ["rtl/memory/memory_stream_width_adapter.sv"], [], []))
    if "hidden-transfer" in selected:
        sources = ["rtl/config/local_memory_layout_pkg.sv", "rtl/memory/hidden_transfer.sv"]
        configurations.append(("hidden_transfer", "hidden_transfer", sources, [], []))
        configurations.append(("hidden_transfer_synthesis", "hidden_transfer",
                               ["-DSYNTHESIS", *sources], [], ["--control-errors"]))
    if "softmax" in selected:
        configurations.append(("softmax_engine", "softmax_engine_start_top",
                               ["rtl/operators/softmax/softmax_engine.sv",
                                "tb/harness/softmax_row_pipeline_parent_stub.sv",
                                "tb/harness/softmax_engine_start_top.sv"], [], []))
        configurations.append(("softmax_row", "softmax_row_regression_top",
                               ["-f", "filelists/production.f",
                                "tb/harness/softmax_row_regression_top.sv"], [numeric], []))
    if "matmul-output" in selected:
        configurations.append(("matmul_output", "matmul_output_controller_top",
                               ["-f", "filelists/production.f", "sim/memory/local_sram_array.sv",
                                "tb/harness/matmul_output_controller_top.sv"], [numeric], []))
    if "attention-context-writer" in selected:
        configurations.append(("attention_context_writer", "attention_context_writer",
                               ["rtl/operators/attention/attention_context_writer.sv"], [], []))
    if "elementwise" in selected:
        configurations.append(("elementwise", "elementwise_engine",
                               ["-GFFN_FEATURES=16", "-f", "filelists/production.f"], [], []))
    if "matmul-activation" in selected:
        configurations.append(("matmul_activation", "matmul_activation_controller",
                               ["rtl/common/ready_valid_fifo.sv",
                                "rtl/operators/matmul/matmul_activation_controller.sv"], [], []))
    if "psme" in selected:
        post = run / "forward_postprocess_model.o"
        command("post_model_build", ["gcc", "-std=c11", "-O2", "-ffp-contract=off", "-c",
                                     "cmodel/forward_postprocess_model.c", "-o", post])
        configurations.append(("psme", "draft_verify_state_controller_top", ["-f", "filelists/production.f",
                               "tb/harness/draft_verify_state_controller_top.sv"], [numeric, post],
                               (["--observed-feature2", *[str(path.resolve(strict=True)) for path in args.feature2_reference]]
                                if args.feature2_reference else ["cases/control/uaps_future_token_admission.json"])))
    for case, top in (("invalidation", "attention_invalidation"), ("refresh-score", "refresh_score_update")):
        if case in selected:
            configurations.append((case, top, ["-f", "filelists/production.f"], [numeric],
                ["cases/control/atse_attention_dependencies.json",
                 "cases/control/atse_cross_block_pending_refresh.json"]))
    if "atse" in selected:
        refresh = run / "token_refresh_model.o"
        command("refresh_model_build", ["gcc", "-std=c11", "-O2", "-ffp-contract=off", "-c",
                                        "cmodel/token_refresh_model.c", "-o", refresh])
        configurations.append(("atse", "attention_guided_token_selector_tb", ["-f", "filelists/production.f",
                               "tb/harness/attention_guided_token_selector_tb.sv"], [numeric, refresh],
                               (["--observed-regular", *[str(path.resolve(strict=True)) for path in args.feature1_reference]]
                                if args.feature1_reference else
                                ["--observed-boundary", *[str(path.resolve(strict=True)) for path in args.boundary_reference]]
                                if args.boundary_reference else
                                ["--precision-reference", *[str(path.resolve(strict=True)) for path in args.precision_reference]]
                                if args.precision_reference else ["cases/control/atse_cross_block_token_selection.json"])))
    for name, top, sources, objects, runtime in configurations:
        build = run / name
        build.mkdir()
        testbench = {"matmul_output": "matmul_output_controller",
                     "softmax_engine": "softmax_engine_start",
                     "softmax_row": "softmax_row",
                     "psme": "draft_verify_state_controller",
                     "atse": "attention_guided_token_selector",
                     "ddr_model": "axi4_ddr_image_io",
                     "token_loader": "next_row_embedding"}.get(name, top)
        if name.startswith(("axi_master_", "axi_parameter_")):
            testbench = "axi_write_error_stall"
        model = "V" + top
        command(name + "_build", [tool, "--cc", "--exe", "--build", "-j", args.jobs, "--threads", "1",
                                  "--assert", "-Wno-fatal", "--top-module", top, "--prefix", model, "--Mdir", build,
                                  *numeric_defines, *sources, RTL_ROOT / f"tb/{testbench}_regression.cpp", *objects,
                                  "-CFLAGS", f"-std=c++17 -O2 {' '.join(numeric_defines)} -I{RTL_ROOT / 'third_party/dramsim3/ext/headers'}"])
        command(name + "_test", [build / model, *runtime], "PASS")
        if name == "atse" and not any((args.feature1_reference, args.boundary_reference,
                                       args.precision_reference)):
            command("atse_boundary_precision", [build / model, "--observed-boundary",
                    "cases/control/feature12_boundary_precision.json",
                    "cases/control/feature12_block_initialization.json"], "PASS")
            command("atse_context_precision", [build / model, "--precision-reference",
                    "cases/control/feature12_context_precision.json"], "PASS")
        if name.startswith("axi_master_"):
            for suffix, flags in (("last", ["+stall_last"]),
                    ("error", ["+response_error"]),
                    ("error_last", ["+response_error", "+stall_last"]),
                    ("abort", ["+abort_request"]),
                    ("abort_last", ["+abort_request", "+stall_last"])):
                command(name + "_" + suffix, [build / model, *flags], "PASS")
            for delay in (0, 1, 3):
                for label, flags in (("okay", []), ("error", ["+response_error"]),
                                     ("abort", ["+abort_request"])):
                    command(f"{name}_aw_{label}_{delay}",
                            [build / model, "+stall_aw", f"+aw_delay={delay}", *flags], "PASS")
    summary = {"status": "PASS", "scope": "model-free numerical, packed-event and SRAM/PE/config protocol checks",
               "verilator": str(tool), "runtime_threads": 1, "cases": selected, "tests": results}
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "cases": selected, "summary": str(run / "summary.json")}))


if __name__ == "__main__":
    main()
