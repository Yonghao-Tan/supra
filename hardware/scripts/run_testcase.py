#!/usr/bin/env python3
"""Run a prepared testcase and compare its expected raw DDR regions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import shlex
import time

from artifact_paths import artifact_directory
from simulation_runtime import (DEFAULT_VERILATOR, DRAMSIM3_SOURCES, INCLUDE_FLAGS,
                            RTL_ROOT, run_command, runtime_resource_check)
from build_forward_postprocess_config import TYPE_SIZE, pack_schema

TESTCASE_FEATURES = ["dependency_merge_initial", "multiple_checkpoints", "head_checkpoints", "head_launch_selection", "head_phase_selection", "head_candidates", "attention_checkpoints", "copies", "initial_segments", "handoff_actions", "check_embedding_reads", "ddr_idle_trace", "axi_read_trace", "axi_write_trace", "physical_transfer_bytes", "command_profile", "strided_expected", "execution_expected"]
TESTCASE_FEATURES.extend(("actual_metadata_layout", "post_block_completions"))


def source_identity(paths: list[str], root: Path = RTL_ROOT) -> str:
    """Bind a reusable binary to its actual source contents, including dirty edits."""
    digest = hashlib.sha256()
    for name in sorted(set(paths)):
        path = root / name
        path.resolve(strict=True).relative_to(root.resolve())
        digest.update(name.encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def compiled_source_inputs(build: Path, root: Path = RTL_ROOT) -> list[str]:
    # Compiler/Verilator dependencies include transitive headers and nested filelists.
    # Generated C++ and system/tool headers are covered by the build/tool version,
    # while this record covers maintained input files only.
    paths = {"scripts/run_testcase.py", "scripts/simulation_runtime.py"}
    dependencies = list(build.rglob("*.d"))
    if not dependencies:
        raise ValueError("build has no source dependencies; cannot record reusable binary")
    for dependency in dependencies:
        for token in shlex.split(dependency.read_text().replace("\\\n", " ")):
            if token.endswith(":"):
                continue
            path = Path(token)
            # Verilator dependencies use the source working directory; GCC
            # object dependencies use the generated makefile's directory.
            candidates = [path] if path.is_absolute() else [root/path, dependency.parent/path]
            for candidate in candidates:
                try:
                    relative = candidate.resolve(strict=True).relative_to(root.resolve())
                except (ValueError, FileNotFoundError):
                    continue
                if candidate.is_file():
                    paths.add(relative.as_posix())
    if not any(name.endswith(".sv") for name in paths) or "tb/testcase_runner.cpp" not in paths:
        raise ValueError("incomplete RTL/TB build dependencies; rebuild with dependency output")
    return sorted(paths)


def validate_binary_sources(built: dict, root: Path = RTL_ROOT) -> None:
    paths = built.get("source_inputs")
    if not paths or not built.get("source_identity"):
        raise ValueError("reused binary has no source identity; fresh build required")
    try:
        identity = source_identity(paths, root)
    except (OSError, ValueError) as error:
        raise ValueError("reused binary source inputs unavailable; fresh build required") from error
    if identity != built["source_identity"]:
        raise ValueError("reused binary sources changed; fresh build required")


def validate_prepared_configurations(case: Path, cfg: dict) -> int:
    """Check every prepared execution before simulating its expensive predecessors.

    Read the same initial image and ordered overlays as the testbench.
    """
    for item in cfg.get("expected", []):
        if "execution_index" in item and (type(item["execution_index"]) is not int or
                not 0 <= item["execution_index"] < len(cfg["executions"])):
            raise ValueError("expected execution_index must identify an existing execution")
    mapping = json.loads((case.parent / cfg["memory_map"]).read_text())
    regions = mapping["memory_map"]
    image = case.parent / cfg["ddr_image"]
    segments = [(mapping["base_address"], image.stat().st_size, image)]
    segments.extend((item["address"], item["bytes"], case.parent / item["path"])
                    for item in cfg.get("initial_segments", []))

    def mapped(address, size, write=False):
        if size < 0 or not any(row["base"] <= address and address + size <= row["limit"]
                               and (not write or row["access"] != "read_only")
                               for row in regions):
            raise ValueError(f"unmapped {'write' if write else 'read'} range {address:#x}+{size}")

    def read(address, size):
        mapped(address, size)
        result = bytearray(size)
        present = bytearray(size)
        for base, length, path in segments:
            first, last = max(address, base), min(address + size, base + length)
            if first >= last:
                continue
            with path.open("rb") as handle:
                handle.seek(first - base)
                payload = handle.read(last - first)
            if len(payload) != last - first:
                raise ValueError(f"short configuration input: {path}")
            result[first-address:last-address] = payload
            present[first-address:last-address] = b"\1" * len(payload)
        if not all(present):
            raise ValueError(f"configuration is not initialized at {address:#x}")
        return bytes(result)

    def decode(name, address):
        schema = json.loads((RTL_ROOT / "config" / f"{name}.json").read_text())
        if address % schema["alignment_bytes"]:
            raise ValueError(f"{name} address is not aligned: {address:#x}")
        payload = read(address, schema["size_bytes"])
        values = {field["name"]: int.from_bytes(
            payload[field["offset"]:field["offset"] + TYPE_SIZE[field["type"]]],
            "little", signed=field["type"].startswith("i")) for field in schema["fields"]}
        pack_schema(schema, values, name)
        return values

    attention_checks = cfg.get("attention_checkpoints", [])
    if isinstance(attention_checks, dict):
        attention_checks = [attention_checks]
    for index, execution in enumerate(cfg["executions"]):
        values = decode("execution_config", execution["config_address"])
        if values["start_layer"] + values["layer_count"] > 32:
            raise ValueError(f"execution {index}: layer range exceeds 32")
        for check in attention_checks:
            if check.get("execution_index", 0) != index:
                continue
            layer = check.get("layer", 0)
            if type(layer) is not int or not values["start_layer"] <= layer < values["start_layer"] + values["layer_count"]:
                raise ValueError(f"execution {index}: Attention checkpoint layer is outside its layer range")
            preserves_input = layer == values["start_layer"] or (layer == 31 and values["flags"] & (1 << 13))
            if not preserves_input and any(item["name"] in
                    ("preserved_hidden", "preserved_hidden_refill") for item in check.get("expected", [])):
                raise ValueError(f"execution {index}: hidden preservation checkpoint needs an input-preservation layer")
        for copy in execution.get("copies", []):
            if copy["bytes"] <= 0:
                raise ValueError("empty inter-execution copy")
            mapped(copy["source_address"], copy["bytes"])
            mapped(copy["destination_address"], copy["bytes"], write=True)
        for action in execution.get("handoff_actions", []):
            if action["kind"] not in ("execution_from_metadata", "state_to_cross_block_inputs", "full_sequence_from_state", "packed_boundary_from_state"):
                raise ValueError(f"unknown handoff action {action['kind']}")
            if action["config_address"] != execution["config_address"]:
                raise ValueError("handoff config differs from launched execution")
            mapped(action["metadata_address"], action["metadata_capacity"])
            if action["kind"] == "full_sequence_from_state":
                if not 1 <= action["sequence"] <= 2048:
                    raise ValueError("baseline sequence is outside 1..2048")
                mapped(action["metadata_address"], action["metadata_capacity"], write=True)
                mapped(action["source_state_address"], 32 * 32)
                mapped(action["table_address"], action["sequence"] * 4, write=True)
            if "history_address" in action:
                mapped(action["history_address"], 16+action["sequence"], write=True)
                mapped(action["token_table_address"], action["sequence"]*8, write=True)
                mapped(action["source_state_address"], action["source_state_count"]*32)
                mapped(action["source_event_address"], 64)
                mapped(action["source_metadata_address"], action["source_metadata_capacity"])
            if action["kind"] == "packed_boundary_from_state":
                mapped(action["metadata_address"], action["metadata_capacity"], write=True)
                mapped(action["table_address"], ((action["sequence"]+7)//8)*64, write=True)
                if action["jobs_capacity"]:
                    mapped(action["jobs_address"], action["jobs_capacity"]*64, write=True)
                mapped(action["probability_config_address"], 80, write=True)
                mapped(action["refresh_address"], 176, write=True)
                if "next_state_address" in action:
                    mapped(action["next_state_address"], action["next_state_count"]*32, write=True)
            if "layout_address" in action:
                mapped(action["layout_address"], action["layout_capacity"], write=True)
                if "state_address" in action:
                    mapped(action["state_address"], action["state_count"] * 32, write=True)
        if values["flags"] & 4:
            post = decode("forward_postprocess_config", values["forward_postprocess_configuration_base"])
            if values["refresh_configuration_offset"]:
                refresh_address = execution["config_address"] + values["refresh_configuration_offset"]
                refresh = decode("token_refresh_config", refresh_address)
                if refresh["bytes"] == 208:
                    read(refresh_address + 176, 32)
                if refresh["flags"] & 128 and not post["flags"] & 1:
                    raise ValueError("paired pending after post requires action-confidence history")
            for offset in range(values["generation_block_count"]):
                block = decode("draft_verify_block_config", post["block_configuration_base"] + offset * 32)
                if block["observed_mask"] and not block["flags"] & 2:
                    raise ValueError("current block cannot specify a future observed_mask")
    return len(cfg["executions"])


def silu_defines(case):
    mode = case.get("silu_mode", "monotone")
    if mode not in ("monotone", "explicit"):
        raise ValueError(f"unsupported fixed SiLU mode: {mode}")
    segments = case.get("silu_segments")
    if mode == "explicit":
        if not isinstance(segments, list) or len(segments) != 16 or any(
                type(value) is not int or not 0 <= value <= 0xffffffff for value in segments):
            raise ValueError("explicit SiLU requires sixteen raw uint32 segments")
        # Identical coefficients use one build setting across captured and
        # bundled cases; the RTL package remains the default table source.
        package = (RTL_ROOT / "rtl/compute/bf16/bf16_silu_pwl_pkg.sv").read_text()
        table = package.split("`elsif SUPRA_SILU_MONOTONE", 1)[1].split("`else", 1)[0]
        default = [int(value, 16) for value in re.findall(r"segment_coefficients = 32'h([0-9a-fA-F]{8});", table)]
        if len(default) == 16 and segments == default:
            return ["-DSUPRA_SILU_MONOTONE"]
        return [f"-DSUPRA_SILU_SEGMENT_{i}=32'h{value:08x}" for i, value in enumerate(segments)]
    if segments is not None:
        raise ValueError("silu_segments requires explicit mode")
    return ["-DSUPRA_SILU_MONOTONE"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=RTL_ROOT / "cases/head_state_update_32tokens/case.json")
    parser.add_argument("--list-cases", action="store_true",
                        help="list included prepared cases without building or running")
    parser.add_argument("--run-id", default=f"testcase-{time.time_ns()}")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--verilator", type=Path, default=DEFAULT_VERILATOR)
    parser.add_argument("--binary", type=Path, help="reuse a previously built testcase_runner executable")
    parser.add_argument("--build-only", action="store_true", help="build the simulator for later runs")
    parser.add_argument("--validate-only", action="store_true",
                        help="check prepared files and execution configs without building or simulating")
    parser.add_argument("--ddr-idle-trace", action="store_true",
                        help="record intervals with no pending requests and drained read/write paths")
    parser.add_argument("--dram-trace", action="store_true",
                        help="record DRAMSim3 transaction addresses, read/write operations and DRAM cycles")
    parser.add_argument("--axi-read-trace", action="store_true",
                        help="record every accepted AXI read request for focused calibration")
    parser.add_argument("--axi-write-trace", action="store_true",
                        help="record accepted AXI write addresses and WSTRB for focused calibration")
    args = parser.parse_args()
    if args.list_cases:
        for path in sorted((RTL_ROOT / "cases").rglob("case.json")):
            record = json.loads(path.read_text())
            if record.get("schema") == "supra-testcase/v1":
                print(f"{path.relative_to(RTL_ROOT)}\t{len(record['executions'])} execution(s)")
        return 0
    if args.threads < 1 or args.jobs < 1:
        parser.error("threads and jobs must be positive")
    case = args.case.resolve(strict=True)
    cfg = json.loads(case.read_text())
    if cfg.get("schema") != "supra-testcase/v1" or not cfg.get("executions") or not cfg.get("expected"):
        parser.error("case requires supra-testcase/v1, executions, and expected regions")
    defines = silu_defines(cfg)
    for name in ("ddr_image", "memory_map", "dramsim3_config"):
        (case.parent / cfg[name]).resolve(strict=True)
    for item in cfg["expected"]:
        if (case.parent / item["path"]).stat().st_size != item["bytes"]:
            parser.error("expected file size mismatch")
    for item in cfg.get("initial_segments", []):
        if (case.parent / item["path"]).stat().st_size != item["bytes"]:
            parser.error("initial segment file size mismatch")
    required_features = {"physical_transfer_bytes", *cfg.get("required_features", [])}
    for execution in cfg["executions"]:
        if "expected_post_block_completions" in execution:
            expected = execution["expected_post_block_completions"]
            if not isinstance(expected, list) or any(type(value) is not bool for value in expected):
                parser.error("expected_post_block_completions must be a boolean array")
            required_features.add("post_block_completions")
    if not required_features.issubset(TESTCASE_FEATURES):
        parser.error("testcase requires unsupported runner features")
    if any("execution_index" in item for item in cfg["expected"]):
        required_features.add("execution_expected")
    if any("element_bytes" in item or "stride_bytes" in item for item in cfg["expected"]):
        required_features.add("strided_expected")
    required_features.update(name for name in ("head_checkpoints", "attention_checkpoints", "initial_segments") if name in cfg)
    checkpoint_sets = {}
    for name in ("head_checkpoints", "attention_checkpoints"):
        value = cfg.get(name)
        entries = value if isinstance(value, list) else [value] if value is not None else []
        if isinstance(value, list):
            required_features.add("multiple_checkpoints")
            if not value or any(not isinstance(entry, dict) or "execution_index" not in entry for entry in value):
                parser.error(f"{name} list requires objects with explicit execution_index")
        checkpoint_sets[name] = entries
        scopes = set()
        for entry in entries:
            index = entry.get("execution_index", 0)
            if type(index) is not int or not 0 <= index < len(cfg["executions"]):
                parser.error(f"{name} execution_index outside launches")
            scope = (index, entry.get("layer")) if name == "attention_checkpoints" else index
            if scope in scopes:
                parser.error(f"duplicate {name} scope")
            scopes.add(scope)
            if name == "head_checkpoints" and "execution_index" in entry:
                required_features.add("head_launch_selection")
            if "candidates" in entry:
                required_features.add("head_candidates")
            for item in entry.get("expected", []) + entry.get("candidates", []):
                if (case.parent / item["path"]).stat().st_size != item["bytes"]:
                    parser.error(f"{name} expected file size mismatch")
    if {entry.get("execution_index", 0) for entry in checkpoint_sets["head_checkpoints"]} & {
            entry.get("execution_index", 0) for entry in checkpoint_sets["attention_checkpoints"]}:
        required_features.add("head_phase_selection")
    required_features.update(name for name in ("copies", "handoff_actions", "check_embedding_reads")
                             if any(name in execution for execution in cfg["executions"]))
    count = validate_prepared_configurations(case, cfg)
    if args.validate_only:
        print(f"Validated {count} prepared execution configurations: {case}", flush=True)
        return 0
    if args.ddr_idle_trace:
        required_features.add("ddr_idle_trace")
    if args.axi_read_trace:
        required_features.add("axi_read_trace")
    if args.axi_write_trace:
        required_features.add("axi_write_trace")
    run = artifact_directory("testcase_runner", args.run_id)
    if any(run.iterdir()):
        parser.error(f"run directory already contains files: {run}")
    logs = run / "logs"; logs.mkdir()
    build = run / "build"; build.mkdir()
    data = run / "data"; data.mkdir()
    for command in (["test", "-d", str(build)], ["test", "-w", str(build)],
                    ["df", "-h", str(build)], ["df", "-i", str(build)]):
        subprocess.run(command, check=True)
    resource = runtime_resource_check(run, logs / "resources.log", args.threads, 1)
    if resource["status"] != "PASS":
        raise RuntimeError("; ".join(resource["errors"]))
    environment = os.environ.copy()
    environment["CCACHE_DISABLE"] = "1"
    environment["OBJCACHE"] = ""
    environment["SUPRA_TRACE_DDR_IDLE"] = "1" if args.ddr_idle_trace else "0"
    environment["SUPRA_TRACE_AXI_READ"] = "1" if args.axi_read_trace else "0"
    environment["SUPRA_TRACE_AXI_WRITE"] = "1" if args.axi_write_trace else "0"
    temporary = run / "tmp"; temporary.mkdir()
    environment.update({name: str(temporary) for name in ("TMPDIR", "TMP", "TEMP")})
    tool = args.verilator.resolve(strict=True)
    version = subprocess.check_output([str(tool), "--version"], text=True).strip()
    if not version.startswith("Verilator 5.050 "):
        raise RuntimeError(f"expected Verilator 5.050, found {version}")
    (logs / "tool.log").write_text(f"{tool}\n{version}\n")
    settings = {"runtime_threads": args.threads, "generation_jobs": args.jobs, "build_jobs": args.jobs}
    binary = args.binary.resolve(strict=True) if args.binary else build / "Mdir/testcase_runner"
    if args.binary:
        built = json.loads((binary.parent / "testcase_build.json").read_text())
        validate_binary_sources(built)
        if any(built.get(key) != value for key, value in
               {"silu_defines": defines, "threads": args.threads, "verilator": version}.items()):
            raise ValueError("reused binary differs in SiLU table, runtime threads or Verilator version")
        if built.get("dram_trace", False) != args.dram_trace:
            raise ValueError("reused binary has a different --dram-trace setting; rebuild with the requested setting")
        if not required_features.issubset(built.get("testcase_features", [])):
            raise ValueError("reused binary lacks required testcase features; rebuild this case")
    if not args.binary:
        binary.parent.mkdir()
        numeric_objects = []
        for name in ("rtl_numeric", "forward_postprocess_model"):
            obj = build / (name + ".o")
            run_command(name, ["gcc", "-std=c11", "-O2", "-ffp-contract=off", "-MMD", "-c",
                              f"cmodel/{name}.c", "-o", str(obj)],
                        logs / (name + ".log"), environment=environment)
            numeric_objects.append(str(obj))
        trace_flags = ["-DADDR_TRACE"] if args.dram_trace else []
        command = [str(tool), *defines, "--cc", "--exe", "--build", "--assert",
                   "--language", "1800-2017", "-Wall", "-Wno-fatal", "-Wno-WIDTH",
                   "-Wno-TIMESCALEMOD", "--output-split", "20000", "--output-split-cfuncs", "500",
                   "--verilate-jobs", str(args.jobs), "--build-jobs", str(args.jobs),
                   "--threads", str(args.threads), "-f", "filelists/dramsim3_verification.f",
                   "--top-module", "supra_top_tb", "-GUSE_DRAMSIM3=1",
                   "--Mdir", str(binary.parent), "tb/testcase_runner.cpp", *numeric_objects, *DRAMSIM3_SOURCES,
                   "tb/dramsim3/dramsim3_backend.cpp", "-CFLAGS",
                   " ".join(["-std=c++17", "-O2", "-DFMT_HEADER_ONLY=1", *trace_flags, *INCLUDE_FLAGS, f"-I{RTL_ROOT}"]),
                   "-o", "testcase_runner"]
        run_command("build", command, logs / "build.log", environment=environment,
                    stream_output=True, thread_settings=settings)
        inputs = compiled_source_inputs(build)
        (binary.parent / "testcase_build.json").write_text(json.dumps(
            {"silu_defines": defines, "threads": args.threads, "verilator": version,
             "dram_trace": args.dram_trace,
             "source_inputs": inputs, "source_identity": source_identity(inputs),
             "testcase_features": TESTCASE_FEATURES}, indent=2) + "\n")
    (run / "run.json").write_text(json.dumps({"case": str(case), "binary": str(binary),
        "verilator": str(tool), "version": version, "threads": args.threads,
        "silu_defines": defines, "ddr_idle_trace": args.ddr_idle_trace,
        "dram_trace": args.dram_trace,
        "axi_read_trace": args.axi_read_trace,
        "axi_write_trace": args.axi_write_trace}, indent=2) + "\n")
    if args.build_only:
        print(f"BUILT {binary}; simulation not run", flush=True)
        return 0
    command = ["stdbuf", "-oL", "-eL", str(binary), str(case), str(data), str(args.threads)]
    replay = run_command("replay", command, logs / "replay.log", environment=environment,
                         stream_output=True, thread_settings=settings, expected_runtime_threads=args.threads)
    run_record = json.loads((run / "run.json").read_text())
    run_record["simulation_wall_seconds"] = replay["elapsed_seconds"]
    (run / "run.json").write_text(json.dumps(run_record, indent=2) + "\n")
    summary = json.loads((data / "summary.json").read_text())
    if summary["status"] != "PASS":
        return 1
    for field in ("physical_read_bytes", "physical_write_bytes"):
        values = [execution.get(field) for execution in summary.get("executions", [])]
        if any(type(value) is not int or value < 0 or value % 32 for value in values):
            raise RuntimeError(f"binary did not record valid per-execution {field}")
        if summary.get(field) != sum(values):
            raise RuntimeError(f"binary {field} does not equal the execution sum")
    if summary["physical_read_bytes"] != summary.get("read_bytes"):
        raise RuntimeError("physical read bytes differ from full-width DDR reads")
    if summary["physical_write_bytes"] < summary.get("write_bytes", 0):
        raise RuntimeError("physical write bytes are smaller than enabled WSTRB bytes")
    if args.dram_trace and not any(path.stat().st_size for path in (data / "dramsim3").glob("*addr.trace")):
        raise RuntimeError("DRAMSim3 transaction trace is missing or empty")
    if args.ddr_idle_trace and any(not (data / execution.get("ddr_idle_trace", "missing")).is_file()
                                   for execution in summary["executions"]):
        raise RuntimeError("binary did not record each requested DDR idle trace")
    if args.axi_read_trace and not (data / "axi_ar_trace.tsv").is_file():
        raise RuntimeError("binary did not record the requested AXI read trace")
    if args.axi_write_trace and any(not (data / name).is_file() for name in
                                    ("axi_aw_trace.tsv", "axi_w_trace.tsv")):
        raise RuntimeError("binary did not record the requested AXI write traces")
    if "head_checkpoints" in cfg and summary.get("head_checkpoints", {}).get("status") != "PASS":
        raise RuntimeError("binary did not complete the required head checkpoint comparisons")
    if "attention_checkpoints" in cfg and summary.get("attention_checkpoints", {}).get("status") != "PASS":
        raise RuntimeError("binary did not complete the required Attention checkpoint comparisons")
    expected_copies = sum(copy["bytes"] for execution in cfg["executions"]
                          for copy in execution.get("copies", []))
    if expected_copies and summary.get("host_copy_bytes") != expected_copies:
        raise RuntimeError("binary did not execute the required inter-execution copies")
    expected_initial_bytes = sum(segment["bytes"] for segment in cfg.get("initial_segments", []))
    if expected_initial_bytes and summary.get("initial_segment_bytes") != expected_initial_bytes:
        raise RuntimeError("binary did not load the required initial DDR segments")
    expected_actions = sum(len(execution.get("handoff_actions", []))
                           for execution in cfg["executions"])
    if expected_actions and summary.get("handoff_action_count") != expected_actions:
        raise RuntimeError("binary did not execute the required state/metadata handoff actions")
    print(f"PASS testcase run: {data / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
