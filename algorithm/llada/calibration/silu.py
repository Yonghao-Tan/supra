"""Fit one shared BF16 SiLU table from observed gate/up distributions."""

import argparse
import json
from pathlib import Path
import torch
from numerics.bf16 import _SILU_BREAKPOINTS, _SILU_SLOPES, _SILU_INTERCEPTS
from quantization.rotation import require_algo_output
from calibration.data import SILU_SAMPLING


def statistics_shards(root):
    if (root / "exit_code").read_text().strip() != "0":
        raise ValueError("SiLU observation did not complete successfully")
    names = ("samples.jsonl", "silu_errors.json", "silu_histograms.pt")
    members = [{p.parent for p in root.glob("shard_*/" + name)} for name in names]
    if not members[0] or any((group != members[0] for group in members[1:])):
        raise ValueError(
            "SiLU samples, metadata and histograms have different shard members"
        )
    return sorted(members[0])


def load_statistics(root):
    paths = [shard / "silu_histograms.pt" for shard in statistics_shards(root)]
    counts = torch.zeros(65536, dtype=torch.int64)
    weights = torch.zeros(65536, dtype=torch.float64)
    by_layer = {}
    for path in paths:
        observation = json.loads((path.parent / "silu_errors.json").read_text())
        if observation.get("sampling") != SILU_SAMPLING:
            raise ValueError("SiLU fitting requires channel-safe sampling")
        records = torch.load(path, map_location="cpu", weights_only=True)
        if set(records) != set(range(32)):
            raise ValueError("SiLU distributions must cover all 32 layers")
        for layer, row in records.items():
            if (
                row["gate_counts"].shape != counts.shape
                or row["up_squared_sum"].shape != weights.shape
                or (not torch.isfinite(row["up_squared_sum"]).all())
                or (row["up_squared_sum"] < 0).any()
                or (row["gate_counts"].sum() != row["sampled_elements"])
            ):
                raise ValueError("invalid SiLU histogram")
            counts += row["gate_counts"]
            weights += row["up_squared_sum"]
            by_layer[layer] = (
                by_layer.get(layer, torch.zeros_like(weights)) + row["up_squared_sum"]
            )
    return (counts, weights, by_layer)


def neighbors(value, radius=4):
    rounded = torch.tensor(value, dtype=torch.float32).to(torch.bfloat16)
    raw = int(rounded.view(torch.int16)) & 65535
    words = torch.arange(max(0, raw - radius), min(65535, raw + radius) + 1).to(
        torch.int16
    )
    values = words.view(torch.bfloat16).float()
    return values[torch.isfinite(values)].tolist()


def segment_output(x, slope, intercept):
    return (
        ((x.float() * slope).to(torch.bfloat16).float() + intercept)
        .to(torch.bfloat16)
        .double()
    )


def fit_table(x, target, weights, *, monotone_positive_branch=False):
    candidates = []
    for index, (lo, hi) in enumerate(
        zip(_SILU_BREAKPOINTS[:-1], _SILU_BREAKPOINTS[1:])
    ):
        mask = (x > -8) & (x >= lo) & (x < hi) & (weights > 0)
        (xx, yy, ww) = (x[mask], target[mask], weights[mask])
        (old_slope, old_intercept) = (_SILU_SLOPES[index], _SILU_INTERCEPTS[index])

        def error(slope, intercept):
            return float(
                ((segment_output(xx, slope, intercept) - yy).square() * ww).sum()
            )

        choices = [(old_slope, old_intercept, error(old_slope, old_intercept))]
        if len(xx) >= 2 and ww.sum() > 0:
            if index in (7, 8):
                fitted_slope = float((ww * xx * yy).sum() / (ww * xx.square()).sum())
                fitted_intercept = 0.0
                intercept_candidates = [0.0]
            else:
                total = ww.sum()
                (mx, my) = ((ww * xx).sum() / total, (ww * yy).sum() / total)
                variance = (ww * (xx - mx).square()).sum()
                fitted_slope = float((ww * (xx - mx) * (yy - my)).sum() / variance)
                fitted_intercept = float(my - fitted_slope * mx)
                intercept_candidates = neighbors(fitted_intercept)
            for trial_slope in neighbors(fitted_slope):
                for trial_intercept in intercept_candidates:
                    choices.append(
                        (
                            trial_slope,
                            trial_intercept,
                            error(trial_slope, trial_intercept),
                        )
                    )
        candidates.append(choices)
    selected = [min(range(len(rows)), key=lambda i: rows[i][2]) for rows in candidates]
    if monotone_positive_branch:
        (costs, parents) = (None, [])
        for index, rows in enumerate(candidates):
            values = torch.tensor(rows, dtype=torch.float64)
            current = values[:, 2].clone()
            if index:
                previous = torch.tensor(candidates[index - 1], dtype=torch.float64)
                transitions = costs[:, None].expand(-1, len(rows)).clone()
                boundary = _SILU_BREAKPOINTS[index]
                if boundary > -1:
                    right_x = torch.tensor(boundary, dtype=torch.bfloat16)
                    left_x = torch.nextafter(
                        right_x, torch.tensor(float("-inf"), dtype=torch.bfloat16)
                    )
                    left = segment_output(
                        left_x, previous[:, 0].float(), previous[:, 1].float()
                    )
                    right = segment_output(
                        right_x, values[:, 0].float(), values[:, 1].float()
                    )
                    transitions[left[:, None] > right[None, :]] = float("inf")
                (best, parent) = transitions.min(dim=0)
                current += best
                parents.append(parent)
            if _SILU_BREAKPOINTS[index] >= -1:
                current[values[:, 0] < 0] = float("inf")
            if index == len(candidates) - 1:
                tail = torch.tensor(8.0, dtype=torch.bfloat16)
                last_x = torch.nextafter(
                    tail, torch.tensor(float("-inf"), dtype=torch.bfloat16)
                )
                last = segment_output(
                    last_x, values[:, 0].float(), values[:, 1].float()
                )
                current[last > 8.0] = float("inf")
            costs = current
        if not torch.isfinite(costs).any():
            raise ValueError("no monotone SiLU candidate table")
        selected[-1] = int(costs.argmin())
        for index in range(len(candidates) - 2, -1, -1):
            selected[index] = int(parents[index][selected[index + 1]])
    (slopes, intercepts, measurements) = ([], [], [])
    for index, (rows, choice) in enumerate(zip(candidates, selected)):
        (slope, intercept, error) = rows[choice]
        slopes.append(slope)
        intercepts.append(intercept)
        measurements.append(dict(segment=index, old_error=rows[0][2], new_error=error))
    return (slopes, intercepts, measurements)


def table_error(x, target, weights, slopes, intercepts):
    total = 0.0
    for index, (lo, hi) in enumerate(
        zip(_SILU_BREAKPOINTS[:-1], _SILU_BREAKPOINTS[1:])
    ):
        mask = (x > -8) & (x >= lo) & (x < hi) & (weights > 0)
        total += float(
            (
                (
                    segment_output(x[mask], slopes[index], intercepts[index])
                    - target[mask]
                ).square()
                * weights[mask]
            ).sum()
        )
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--check-root", type=Path, required=True)
    parser.add_argument("--secondary-calibration-root", type=Path)
    parser.add_argument("--secondary-check-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-only-calibration", action="store_true")
    parser.add_argument(
        "--monotone-positive-branch",
        action="store_true",
        help="Require nondecreasing BF16 outputs for gate >= -1, including segment boundaries",
    )
    args = parser.parse_args()
    if (args.secondary_calibration_root is None) != (args.secondary_check_root is None):
        parser.error("secondary calibration and check roots must be supplied together")
    output = require_algo_output(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    calibration_roots = [args.calibration_root]
    check_roots = [args.check_root]
    if args.secondary_calibration_root is not None:
        calibration_roots.append(args.secondary_calibration_root)
        check_roots.append(args.secondary_check_root)
    request_sets = [set(), set()]
    sampling_schedules = set()
    for root in calibration_roots + check_roots:
        shards = statistics_shards(root)
        records = [
            json.loads(line)
            for path in (shard / "samples.jsonl" for shard in shards)
            for line in path.open()
        ]
        for shard in shards:
            observation = json.loads((shard / "silu_errors.json").read_text())
            schedule = observation["sampling"]
            sampling_schedules.add(schedule)
            for line in (shard / "samples.jsonl").read_text().splitlines():
                if (
                    json.loads(line)["silu_sampling"]
                    != schedule
                ):
                    raise ValueError(
                        "SiLU request and histogram sampling schedules differ"
                    )
        identities = {record["sample_id"] for record in records}
        if args.train_only_calibration:
            expected_split = "train" if root in calibration_roots else "validation"
            if any(
                (
                    r.get("source_split") != expected_split
                    or r.get("train_only_calibration") is not True
                    for r in records
                )
            ):
                raise ValueError(
                    "train-only SiLU rejects test or unverified observations"
                )
            for path in (shard / "silu_errors.json" for shard in shards):
                metadata = json.loads(path.read_text())
                if (
                    metadata.get("train_only_calibration") is not True
                    or metadata.get("source_split") != expected_split
                ):
                    raise ValueError(
                        "SiLU histogram provenance differs from its request records"
                    )
        if not identities or len(identities) != len(records):
            raise ValueError("missing or duplicate SiLU observation requests")
        group = request_sets[0 if root in calibration_roots else 1]
        if group.intersection(identities):
            raise ValueError("duplicate requests across task roots")
        group.update(identities)
    if request_sets[0] & request_sets[1]:
        raise ValueError("calibration and check requests overlap")
    if len(sampling_schedules) != 1:
        raise ValueError("SiLU observation roots use different sampling schedules")
    x = torch.arange(65536).to(torch.int16).view(torch.bfloat16).double()
    finite = torch.isfinite(x)
    target = torch.nn.functional.silu(x.float()).to(torch.bfloat16).double()

    def combine(roots):
        counts = torch.zeros(65536, dtype=torch.int64)
        weights = torch.zeros(65536, dtype=torch.float64)
        (layers, tasks) = ({}, [])
        for root in roots:
            (cc, ww, ll) = load_statistics(root)
            if cc[~finite].sum():
                raise ValueError("nonfinite observed gates")
            old_error = table_error(x, target, ww, _SILU_SLOPES, _SILU_INTERCEPTS)
            if len(roots) > 1 and (
                not torch.isfinite(torch.tensor(old_error)) or old_error <= 0
            ):
                raise ValueError("joint SiLU needs positive finite task baseline error")
            divisor = old_error if len(roots) > 1 else 1.0
            counts += cc
            weights += ww / divisor
            for layer, mass in ll.items():
                layers[layer] = (
                    layers.get(layer, torch.zeros_like(weights)) + mass / divisor
                )
            tasks.append((root, ww, old_error))
        return (counts, weights, layers, tasks)

    (counts, weights, _, _) = combine(calibration_roots)
    (check_counts, check_weights, check_layers, check_tasks) = combine(check_roots)
    (slopes, intercepts, fit) = fit_table(
        x, target, weights, monotone_positive_branch=args.monotone_positive_branch
    )
    checks = []
    for layer, mass in sorted(check_layers.items()):
        checks.append(
            dict(
                layer=layer,
                old_error=table_error(x, target, mass, _SILU_SLOPES, _SILU_INTERCEPTS),
                new_error=table_error(x, target, mass, slopes, intercepts),
            )
        )
    result = dict(
        schema_version="shared-silu-pwl-calibration/v1",
        train_only_calibration=args.train_only_calibration,
        sampling=next(iter(sampling_schedules)),
        task_combination="sum task error / original-table task error"
        if len(calibration_roots) > 1
        else "single task",
        secondary_calibration_root=str(args.secondary_calibration_root)
        if args.secondary_calibration_root
        else None,
        secondary_check_root=str(args.secondary_check_root)
        if args.secondary_check_root
        else None,
        calibration_root=str(args.calibration_root.resolve()),
        check_root=str(args.check_root.resolve()),
        scope="Shared sixteen segments, existing breakpoints/tails; BF16 multiply then BF16 add",
        objective="sum up^2*(BF16-PWL(gate)-BF16-native-SiLU(gate))^2; excludes final multiply rounding and unchanged tails",
        monotone_positive_branch=args.monotone_positive_branch,
        calibration_sampled_elements=int(counts.sum()),
        check_sampled_elements=int(check_counts.sum()),
        calibration_requests=len(request_sets[0]),
        check_requests=len(request_sets[1]),
        breakpoints=list(_SILU_BREAKPOINTS),
        slopes=slopes,
        intercepts=intercepts,
        fit=fit,
        check_by_layer=checks,
        check_by_task=[
            dict(
                root=str(root),
                old_error=old_error,
                new_error=table_error(x, target, mass, slopes, intercepts),
            )
            for (root, mass, old_error) in check_tasks
        ],
        check_old_error=table_error(
            x, target, check_weights, _SILU_SLOPES, _SILU_INTERCEPTS
        ),
        check_new_error=table_error(x, target, check_weights, slopes, intercepts),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
