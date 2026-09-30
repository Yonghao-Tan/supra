"""Export compact logical hardware work from completed evaluation traces."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "supra-algorithm-workload/v2"
MASKED, TENTATIVE, LOCKED = 0, 1, 2


def _task_family(name: str) -> str:
    if name.startswith("gsm8k"):
        return "gsm8k"
    raise ValueError(f"unsupported task name: {name!r}")


def _positions(event: dict[str, Any], name: str) -> list[int]:
    values = [int(value) for value in event.get(name, [])]
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate positions")
    return values


def _phase(event: dict[str, Any], trace: dict[str, Any]) -> str:
    kind = event.get("forward_kind")
    if kind == "full_sequence":
        return "full_sequence"
    if kind == "local_block":
        return "regular"
    if kind == "boundary_refresh":
        block_initialization_block = int(
            (trace.get("feature1_parameters") or {}).get(
                "cross_block_full_prefix_oracle_block", -1
            )
        )
        return (
            "block_initialization"
            if int(event["block_index"]) == block_initialization_block
            else "boundary"
        )
    if kind in {
        "local_confirmation",
        "local_forced_finish",
    }:
        return "repair"
    raise ValueError(f"unsupported forward_kind: {kind!r}")


def _segment(
    first_layer: int,
    last_layer: int,
    query_tokens: int,
    a4_tokens: int,
    a8_tokens: int,
    kv_write_tokens: int,
    *,
    l31_output_tokens: int | None = None,
    l31_output_a4_tokens: int | None = None,
    l31_output_a8_tokens: int | None = None,
) -> dict[str, int]:
    result = {
        "first_layer": first_layer,
        "last_layer": last_layer,
        "layer_count": last_layer - first_layer + 1,
        "query_tokens": query_tokens,
        "a4_tokens": a4_tokens,
        "a8_tokens": a8_tokens,
        "kv_write_tokens": kv_write_tokens,
    }
    if a4_tokens + a8_tokens != query_tokens:
        raise ValueError("segment A4/A8 counts do not match query tokens")
    if not 0 <= kv_write_tokens <= query_tokens:
        raise ValueError("segment K/V writes exceed query tokens")
    if l31_output_tokens is not None:
        if last_layer != 31 or not first_layer <= 31:
            raise ValueError("l31_output_tokens requires a segment covering L31")
        if not 0 <= l31_output_tokens <= query_tokens:
            raise ValueError("L31 output tokens exceed L31 query tokens")
        if l31_output_a4_tokens is None or l31_output_a8_tokens is None:
            raise ValueError("L31 output precision counts are required")
        if l31_output_a4_tokens + l31_output_a8_tokens != l31_output_tokens:
            raise ValueError("L31 output A4/A8 counts do not match output tokens")
        if l31_output_a4_tokens > a4_tokens or l31_output_a8_tokens > a8_tokens:
            raise ValueError("L31 output precision exceeds L31 query precision")
        result["l31_output_tokens"] = l31_output_tokens
        result["l31_output_a4_tokens"] = l31_output_a4_tokens
        result["l31_output_a8_tokens"] = l31_output_a8_tokens
    elif l31_output_a4_tokens is not None or l31_output_a8_tokens is not None:
        raise ValueError("non-L31 segment contains L31 output precision")
    return result


def _consumer_positions(
    event: dict[str, Any], trace: dict[str, Any], input_positions: list[int]
) -> tuple[list[int], list[int], list[int]]:
    prediction = _positions(event, "prediction_positions")
    future = _positions(event, "next_progress_positions")
    input_set = set(input_positions)
    prediction_set, future_set = set(prediction), set(future)
    if not prediction_set <= input_set or not future_set <= input_set:
        raise ValueError("L31 consumer positions must be executed query positions")
    if prediction_set & future_set:
        raise ValueError("current and future prediction positions overlap")

    protocol = trace.get("generation_protocol") or {}
    block_length = int(protocol.get("block_length", 0))
    prompt_tokens = int(trace.get("prompt_token_count", -1))
    block_index = int(event.get("block_index", -1))
    if block_length <= 0 or prompt_tokens < 0 or block_index < 0:
        raise ValueError("trace lacks prompt, block length, or block index")
    current_start = prompt_tokens + block_index * block_length
    current_end = current_start + block_length
    retained = sorted(
        position
        for position in input_set
        if current_start <= position < current_end
        and position not in prediction_set
        and position not in future_set
    )
    return prediction, future, retained


class _PrecisionReplay:
    """Replay the Feature2 state needed by L31 output precision."""

    def __init__(self, trace: dict[str, Any]) -> None:
        protocol = trace.get("generation_protocol") or {}
        self.prompt = int(trace["prompt_token_count"])
        self.block = int(protocol["block_length"])
        generated = trace.get("generated_token_ids")
        if not isinstance(generated, list) or len(generated) != 1:
            raise ValueError("precision replay requires batch size one")
        self.generated = len(generated[0])
        self.state = [MASKED] * self.generated
        self.age = [-1] * self.generated
        feature3 = trace.get("feature3_parameters") or {}
        self.maturity_age = int(feature3.get("maturity_age", 3))
        self.policy = str(feature3.get("precision_policy", "original"))
        if self.policy not in {"all_a8", "mature_only", "masked_only", "original"}:
            raise ValueError(f"unsupported precision policy: {self.policy}")
        precision = trace.get("feature2_precision_parameters") or {}
        self.context_a8_rows = int(precision.get("context_a8_rows", 0))


    def _index(self, position: int) -> int:
        index = int(position) - self.prompt
        if not 0 <= index < self.generated:
            raise ValueError(f"generation position is outside the request: {position}")
        return index

    def _bits(self, position: int) -> int:
        index = self._index(position)
        state, age = self.state[index], self.age[index]
        mature = state == LOCKED and age >= self.maturity_age
        use_a4 = {
            "all_a8": False,
            "mature_only": mature,
            "masked_only": state == MASKED,
            "original": state == MASKED or mature,
        }[self.policy]
        return 4 if use_a4 else 8

    def output_counts(
        self,
        event: dict[str, Any],
        phase: str,
        input_positions: list[int],
        positions: tuple[list[int], list[int], list[int]],
    ) -> tuple[int, int]:
        block_index = int(event["block_index"])
        start = self.prompt + block_index * self.block
        current = range(start, start + self.block)
        current_bits = [self._bits(position) for position in current]
        expected = (current_bits.count(4), current_bits.count(8))
        recorded = (int(event["a4_rows"]), int(event["a8_rows"]))
        if expected != recorded:
            raise ValueError(
                f"replayed current precision {expected} differs from trace {recorded}"
            )

        output = set().union(*map(set, positions))
        if phase == "block_initialization":
            bits = int(event.get("boundary_deep_bits", 0))
            if bits in (4, 8):
                query_bits = {position: bits for position in input_positions}
            elif int(event["active_a4_rows"]) == len(input_positions):
                query_bits = {position: 4 for position in input_positions}
            elif int(event["active_a8_rows"]) == len(input_positions):
                query_bits = {position: 8 for position in input_positions}
            else:
                raise ValueError("mixed full block initialization lacks per-position row bits")
        elif phase == "full_sequence":
            a4 = int(event["layer0_a4_rows"])
            a8 = int(event["layer0_a8_rows"])
            if a4 == len(input_positions) and a8 == 0:
                query_bits = {position: 4 for position in input_positions}
            elif a8 == len(input_positions) and a4 == 0:
                query_bits = {position: 8 for position in input_positions}
            elif event.get("cache_initialization_activation_policy") == "default":
                query_bits = {position: 8 for position in input_positions}
                if not set(current) <= set(query_bits):
                    raise ValueError("mixed full-sequence precision lacks current queries")
                query_bits.update(zip(current, current_bits))
                if (sum(bit == 4 for bit in query_bits.values()),
                        sum(bit == 8 for bit in query_bits.values())) != (a4, a8):
                    raise ValueError("mixed full-sequence precision differs from L0 counts")
            else:
                raise ValueError("mixed full-sequence precision lacks per-position row bits")
        elif phase == "boundary":
            current_bits = [int(bit) for bit in event.get("layer0_current_row_bits", [])]
            if len(current_bits) != self.block or any(
                bit not in (4, 8) for bit in current_bits
            ):
                raise ValueError("boundary lacks current-row precision")
            query_bits = {
                start + offset: bit for offset, bit in enumerate(current_bits)
            }
            if not output <= set(query_bits):
                raise ValueError("boundary L31 output is not confined to the current block")
        else:
            added = set(_positions(event, "next_added_positions"))
            if not added <= set(input_positions):
                raise ValueError("added future positions are not executed queries")
            base_positions = [position for position in input_positions if position not in added]
            # A confirmation-only repair can still execute MASKED current
            # queries. They retain state precision even without a head consumer.
            required = {
                position for position in base_positions
                if start <= position < start + self.block
                and self.state[self._index(position)] != LOCKED
            }
            query_bits = {position: 4 for position in base_positions}
            for position in required:
                query_bits[position] = self._bits(position)
            if self.policy == "all_a8":
                query_bits = {position: 8 for position in base_positions}
            else:
                context = [position for position in base_positions if position not in required]
                scores = event.get("dependency_score_by_row")
                region_start = int(event.get("region_start", -1))
                region_end = int(event.get("region_end", -1))
                if not isinstance(scores, list) or len(scores) != region_end - region_start:
                    raise ValueError("regular forward lacks dependency scores")
                if any(not region_start <= position < region_end for position in context):
                    raise ValueError("regular context outside recorded dependency region")
                ranked = sorted(
                    context,
                    key=lambda position: -float(scores[position - region_start]),
                )
                for position in ranked[: min(len(ranked), self.context_a8_rows)]:
                    query_bits[position] = 8
            for position in added:
                query_bits[position] = self._bits(position)
            if set(query_bits) != set(input_positions):
                raise ValueError("regular precision replay did not cover every query")

        replayed_active = (
            sum(bit == 4 for bit in query_bits.values()),
            sum(bit == 8 for bit in query_bits.values()),
        )
        active = (int(event["active_a4_rows"]), int(event["active_a8_rows"]))
        if phase == "full_sequence":
            active = (int(event["layer0_a4_rows"]), int(event["layer0_a8_rows"]))
        if phase != "boundary" and replayed_active != active:
            raise ValueError(
                f"replayed query precision {replayed_active} differs from trace {active}"
            )
        output_bits = [query_bits[position] for position in output]
        result = (output_bits.count(4), output_bits.count(8))
        if result[0] > active[0] or result[1] > active[1]:
            raise ValueError("L31 output precision exceeds executed query precision")
        return result

    def advance(self, event: dict[str, Any]) -> None:
        start_locked = {
            index for index, state in enumerate(self.state) if state == LOCKED
        }

        def assign(name: str, state: int, age: int) -> None:
            for position in _positions(event, name):
                index = self._index(position)
                self.state[index] = state
                self.age[index] = age

        assign("confirmed_positions", LOCKED, 0)
        assign("remasked_positions", MASKED, -1)
        assign("direct_locked_positions", LOCKED, 0)
        assign("stable_tentative_positions", TENTATIVE, -1)
        assign("fallback_tentative_positions", TENTATIVE, -1)
        assign("forced_finish_positions", LOCKED, 0)
        assign("tail_bypassed_positions", LOCKED, 0)

        current_start = self.prompt + int(event["block_index"]) * self.block
        current_end = current_start + self.block
        for index in start_locked:
            position = self.prompt + index
            if current_start <= position < current_end and self.state[index] == LOCKED:
                self.age[index] += 1

        future_age_positions = _positions(event, "next_progress_positions")
        future_start_locked = {
            self._index(position)
            for position in future_age_positions
            if self.state[self._index(position)] == LOCKED
        }
        admitted = set(_positions(event, "next_admitted_positions"))
        progress = set(_positions(event, "next_progress_positions"))
        if not admitted <= progress:
            raise ValueError("future admissions are not a subset of future predictions")
        if any(self.state[self._index(position)] != MASKED for position in admitted):
            raise ValueError("future admission did not start from MASKED")
        direct = set(_positions(event, "next_direct_locked_positions"))
        tentative = set(_positions(event, "next_tentative_positions"))
        if direct & tentative or direct | tentative != admitted:
            raise ValueError("future admission states do not partition admitted positions")
        confirmed = set(_positions(event, "next_confirmed_positions"))
        remasked = set(_positions(event, "next_remasked_positions"))
        if confirmed & remasked or not (confirmed | remasked) <= progress:
            raise ValueError("future confirmation outcomes are not disjoint executed positions")
        if any(self.state[self._index(p)] != TENTATIVE for p in confirmed | remasked):
            raise ValueError("future confirmation did not start from TENTATIVE")
        assign("next_confirmed_positions", LOCKED, 0)
        assign("next_remasked_positions", MASKED, -1)
        assign("next_direct_locked_positions", LOCKED, 0)
        assign("next_tentative_positions", TENTATIVE, -1)
        for index in future_start_locked:
            if self.state[index] == LOCKED:
                self.age[index] += 1


def convert_event(
    event: dict[str, Any],
    trace: dict[str, Any],
    event_index: int,
    output_precision: tuple[int, int],
) -> dict[str, Any]:
    input_positions = _positions(event, "input_positions")
    if not input_positions:
        length = int(event.get("input_length", 0))
        start = int(event.get("input_start", 0))
        input_positions = list(range(start, start + length))
    query_tokens = len(input_positions)
    if "refresh_positions" not in event:
        raise ValueError("event lacks the K/V write positions")
    refresh_positions = _positions(event, "refresh_positions")
    if not set(refresh_positions) <= set(input_positions):
        raise ValueError("K/V writes are not executed query positions")
    deep_kv_write_tokens = len(refresh_positions)

    prediction, future, retained = _consumer_positions(event, trace, input_positions)

    l31_positions = set(prediction) | set(future) | set(retained)
    lm_head_positions = set(prediction) | set(future)
    l31_output_tokens = len(l31_positions)
    lm_head_tokens = len(lm_head_positions)
    l31_output_a4_tokens, l31_output_a8_tokens = output_precision
    if lm_head_tokens > l31_output_tokens or l31_output_tokens > query_tokens:
        raise ValueError("invalid L31 output subset")

    phase = _phase(event, trace)
    l0_query = int(event["layer0_rows"])
    l0_a4 = int(event["layer0_a4_rows"])
    l0_a8 = int(event["layer0_a8_rows"])
    # A boundary's full L0 query is separate from input_positions, which records
    # the selected deep-layer queries. All L0 query tokens update the L0 cache.
    l0_kv_write_tokens = (
        l0_query if l0_query != query_tokens else deep_kv_write_tokens
    )
    if phase == "full_sequence":
        deep_a4, deep_a8 = l0_a4, l0_a8
        if query_tokens != l0_query:
            raise ValueError("full-sequence L0 and deep query counts differ")
    else:
        deep_a4 = int(event["active_a4_rows"])
        deep_a8 = int(event["active_a8_rows"])
        if deep_a4 + deep_a8 != query_tokens:
            raise ValueError("deep A4/A8 counts do not match executed positions")

    if l0_query == query_tokens and l0_a4 == deep_a4 and l0_a8 == deep_a8:
        segments = [
            _segment(
                0,
                30,
                query_tokens,
                deep_a4,
                deep_a8,
                deep_kv_write_tokens,
            ),
            _segment(
                31,
                31,
                query_tokens,
                deep_a4,
                deep_a8,
                deep_kv_write_tokens,
                l31_output_tokens=l31_output_tokens,
                l31_output_a4_tokens=l31_output_a4_tokens,
                l31_output_a8_tokens=l31_output_a8_tokens,
            ),
        ]
    else:
        segments = [
            _segment(0, 0, l0_query, l0_a4, l0_a8, l0_kv_write_tokens),
            _segment(
                1,
                30,
                query_tokens,
                deep_a4,
                deep_a8,
                deep_kv_write_tokens,
            ),
            _segment(
                31,
                31,
                query_tokens,
                deep_a4,
                deep_a8,
                deep_kv_write_tokens,
                l31_output_tokens=l31_output_tokens,
                l31_output_a4_tokens=l31_output_a4_tokens,
                l31_output_a8_tokens=l31_output_a8_tokens,
            ),
        ]

    return {
        "event_index": event_index,
        "phase": phase,
        "layer0_keep_global_cache": bool(event.get("layer0_keep_global_cache", False)),
        "sequence_length": int(event["cache_sequence_length"]),
        "segments": segments,
        "prediction_tokens": len(prediction),
        "future_prediction_tokens": len(future),
        "retained_hidden_tokens": len(retained),
        "lm_head_tokens": lm_head_tokens,
        "prediction_positions": prediction,
        "future_prediction_positions": future,
        "retained_hidden_positions": retained,
    }


def convert_request(trace: dict[str, Any], shard_id: str = "shard_00") -> dict[str, Any]:
    events = trace.get("trace")
    nfe = int(trace.get("nfe", -1))
    if not isinstance(events, list) or len(events) != nfe or nfe <= 0:
        raise ValueError("trace event count does not match NFE")
    generated_ids = trace.get("generated_token_ids")
    if not isinstance(generated_ids, list) or len(generated_ids) != 1:
        raise ValueError("workload export requires batch size one generated token IDs")
    replay = _PrecisionReplay(trace)
    converted = []
    for index, event in enumerate(events):
        input_positions = _positions(event, "input_positions")
        if not input_positions:
            length = int(event.get("input_length", 0))
            start = int(event.get("input_start", 0))
            input_positions = list(range(start, start + length))
        phase = _phase(event, trace)
        consumers = _consumer_positions(event, trace, input_positions)
        output_precision = replay.output_counts(
            event, phase, input_positions, consumers
        )
        converted.append(
            convert_event(event, trace, index, output_precision)
        )
        replay.advance(event)
    doc_id = int(trace["doc_id"])
    return {
        "request_id": f"{shard_id}:{doc_id}",
        "shard_id": shard_id,
        "doc_id": doc_id,
        "doc_hash": trace.get("doc_hash"),
        "prompt_tokens": int(trace["prompt_token_count"]),
        "generated_tokens": len(generated_ids[0]),
        "nfe": nfe,
        "events": converted,
    }


def export_workload(root: Path, task: str) -> dict[str, Any]:
    requests = []
    seen = set()
    seen_hashes = set()
    shards = sorted(path for path in root.glob("shard_*") if path.is_dir())
    if not shards:
        raise FileNotFoundError(f"no shards below {root}")
    for shard in shards:
        exit_code = shard / "exit_code"
        if not exit_code.is_file() or exit_code.read_text().strip() != "0":
            raise ValueError(f"unfinished evaluation shard: {shard}")
        path = shard / "trace.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                trace = json.loads(line)
                if _task_family(str(trace.get("task_name", ""))) != task:
                    raise ValueError(f"task mismatch in {path}")
                request = convert_request(trace, shard.name)
                if request["request_id"] in seen:
                    raise ValueError(f"duplicate request_id={request['request_id']}")
                if not isinstance(request["doc_hash"], str):
                    raise ValueError(f"missing doc_hash in {path}")
                if request["doc_hash"] in seen_hashes:
                    raise ValueError(f"duplicate doc_hash={request['doc_hash']}")
                seen.add(request["request_id"])
                seen_hashes.add(request["doc_hash"])
                requests.append(request)
    requests.sort(key=lambda request: request["request_id"])
    return {
        "schema": SCHEMA,
        "task": task,
        "request_count": len(requests),
        "requests": requests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact_directory = os.environ.get("SUPRA_ALGORITHM_ROOT")
    if not artifact_directory:
        parser.error("set SUPRA_ALGORITHM_ROOT to the external algorithm artifact directory")
    artifact_root = Path(artifact_directory).resolve()
    output = args.output.resolve()
    if output != artifact_root and artifact_root not in output.parents:
        raise ValueError("workload output must be below SUPRA_ALGORITHM_ROOT")
    result = export_workload(args.root, args.task)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "requests": result["request_count"]}))


if __name__ == "__main__":
    main()
