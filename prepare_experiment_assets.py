"""Create and verify immutable prompt and patient-selection assets.

This utility is deliberately offline: loading ``run_full.py`` only evaluates its
module-level definitions, and no provider call is made.  The committed prompt
fingerprints should be created once, before refactoring the runner, then checked
after every change::

    python prepare_experiment_assets.py --write
    python prepare_experiment_assets.py

Patient files preserve the seeded draw order.  Consequently the 100-patient
file is literally the first 100 lines of the 300-patient file, while both files
contain exactly the same RID sets selected by the legacy shuffle/slice logic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence


SEED = 20260920
DEFAULT_RUNNER = Path("run_full.py")
DEFAULT_CARDS = Path("data/cards.json")
DEFAULT_FINGERPRINTS = Path("prompt_fingerprints.json")
DEFAULT_PATIENT_DIR = Path("patients")
PATIENT_COUNTS = (100, 300)
COMBINED_SEPARATOR = "\0"


def sha1_bytes(value: bytes) -> str:
    return hashlib.sha1(value).hexdigest()


def sha1_text(value: str) -> str:
    return sha1_bytes(value.encode("utf-8"))


def sha1_file(path: Path) -> str:
    return sha1_bytes(path.read_bytes())


def load_runner(path: Path) -> ModuleType:
    """Load the runner as a module without executing its ``__main__`` block."""

    spec = importlib.util.spec_from_file_location("_fingerprint_run_full", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("SYS", "CONDS", "build"):
        if not hasattr(module, name):
            raise AttributeError(f"{path} does not define required name {name}")
    return module


def load_cards(path: Path) -> list[dict[str, Any]]:
    cards = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cards, list) or not cards:
        raise ValueError(f"{path} must contain a non-empty JSON array")
    if not all(isinstance(card, dict) and "RID" in card for card in cards):
        raise ValueError(f"every entry in {path} must be an object with RID")
    return cards


def _fingerprint_invariants(payload: dict[str, Any]) -> dict[str, Any]:
    """Return only values that must survive runner refactors unchanged."""

    return {
        "algorithm": payload["algorithm"],
        "combined_format": payload["combined_format"],
        "selection_seed": payload["selection_seed"],
        "immutable_source": payload["immutable_source"],
        "first_patient": payload["first_patient"],
        "sys": payload["sys"],
        "conditions": payload["conditions"],
    }


def build_prompt_fingerprints(
    runner: ModuleType,
    cards: Sequence[dict[str, Any]],
    *,
    runner_path: Path,
    cards_path: Path,
) -> dict[str, Any]:
    """Fingerprint SYS and every condition's build output for cards[0].card12."""

    first = cards[0]
    if "card12" not in first or not isinstance(first["card12"], str):
        raise ValueError("the first card must contain a string card12 field")
    if not isinstance(runner.SYS, str):
        raise TypeError("run_full.SYS must be a string")
    if not isinstance(runner.CONDS, dict) or not runner.CONDS:
        raise TypeError("run_full.CONDS must be a non-empty dictionary")

    card12 = first["card12"]
    runner_source = runner_path.read_text(encoding="utf-8")
    source_start_marker = "SYS ="
    source_end_marker = '    return p + "\\n\\nProvide your assessment now."'
    source_start = runner_source.find(source_start_marker)
    source_end_start = runner_source.find(source_end_marker, source_start)
    if source_start < 0 or source_end_start < 0:
        raise ValueError("cannot locate the immutable SYS-through-build source block")
    source_end = source_end_start + len(source_end_marker)
    immutable_source = runner_source[source_start:source_end]
    conditions: dict[str, dict[str, Any]] = {}
    for condition, config in runner.CONDS.items():
        prompt = runner.build(card12, config)
        if not isinstance(prompt, str):
            raise TypeError(f"build() returned non-string for {condition}")
        conditions[condition] = {
            "build_sha1": sha1_text(prompt),
            "sys_nul_build_sha1": sha1_text(
                runner.SYS + COMBINED_SEPARATOR + prompt
            ),
            "build_chars": len(prompt),
        }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": "SHA1 over UTF-8 bytes",
        "combined_format": "SYS + U+0000 + build(card12, CONDS[cond])",
        "selection_seed": SEED,
        "immutable_source": {
            "scope": "run_full.py from SYS assignment through build() return",
            "sha1": sha1_text(immutable_source),
            "chars": len(immutable_source),
        },
        "source": {
            "runner": runner_path.as_posix(),
            "runner_file_sha1_at_baseline": sha1_file(runner_path),
            "cards": cards_path.as_posix(),
        },
        "first_patient": {
            "array_index": 0,
            "rid": first["RID"],
            "card_field": "card12",
            "card12_sha1": sha1_text(card12),
            "card12_chars": len(card12),
        },
        "sys": {"sha1": sha1_text(runner.SYS), "chars": len(runner.SYS)},
        "conditions": conditions,
    }
    canonical = json.dumps(
        _fingerprint_invariants(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload["aggregate_sha1"] = sha1_text(canonical)
    return payload


def compare_prompt_fingerprints(
    baseline: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    """Return human-readable invariant mismatches; an empty list means equal."""

    errors: list[str] = []
    for key in (
        "algorithm", "combined_format", "selection_seed", "immutable_source",
        "first_patient", "sys",
    ):
        if baseline.get(key) != current.get(key):
            errors.append(f"{key} differs")

    expected_conditions = baseline.get("conditions", {})
    actual_conditions = current.get("conditions", {})
    expected_names = list(expected_conditions)
    actual_names = list(actual_conditions)
    if expected_names != actual_names:
        errors.append(
            "condition names/order differ: "
            f"expected {expected_names!r}, got {actual_names!r}"
        )
    for name in expected_names:
        if name in actual_conditions and expected_conditions[name] != actual_conditions[name]:
            errors.append(f"condition {name!r} fingerprint differs")

    if baseline.get("aggregate_sha1") != current.get("aggregate_sha1"):
        errors.append("aggregate_sha1 differs")
    return errors


def seeded_rids(cards: Sequence[dict[str, Any]]) -> list[Any]:
    """Return the legacy sorted-RID population in its seeded draw order."""

    rids = [card["RID"] for card in cards]
    if len(rids) != len(set(rids)):
        raise ValueError("cards contain duplicate RID values")
    rids = sorted(rids)
    random.Random(SEED).shuffle(rids)
    return rids


def patient_lists(cards: Sequence[dict[str, Any]]) -> dict[int, list[Any]]:
    drawn = seeded_rids(cards)
    if len(drawn) < max(PATIENT_COUNTS):
        raise ValueError(
            f"need at least {max(PATIENT_COUNTS)} cards, found {len(drawn)}"
        )
    selected = {count: drawn[:count] for count in PATIENT_COUNTS}
    if selected[300][:100] != selected[100]:
        raise AssertionError("n100 selection is not the prefix of n300")
    return selected


def patient_file_content(rids: Iterable[Any]) -> str:
    return "".join(f"{rid}\n" for rid in rids)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        target.write(value)


def write_assets(
    fingerprints: dict[str, Any],
    selected: dict[int, list[Any]],
    *,
    fingerprints_path: Path,
    patient_dir: Path,
    force: bool,
) -> None:
    targets = [fingerprints_path] + [
        patient_dir / f"patients_n{count}.txt" for count in PATIENT_COUNTS
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite immutable assets: "
            + ", ".join(existing)
            + " (use --force only when intentionally establishing a new baseline)"
        )

    write_text(
        fingerprints_path,
        json.dumps(fingerprints, ensure_ascii=False, indent=2) + "\n",
    )
    for count, rids in selected.items():
        write_text(patient_dir / f"patients_n{count}.txt", patient_file_content(rids))


def parse_patient_file(path: Path) -> list[int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        return [int(line.strip()) for line in lines if line.strip()]
    except ValueError as exc:
        raise ValueError(f"{path} contains a non-integer RID") from exc


def verify_assets(
    current: dict[str, Any],
    selected: dict[int, list[Any]],
    *,
    fingerprints_path: Path,
    patient_dir: Path,
) -> None:
    baseline = json.loads(fingerprints_path.read_text(encoding="utf-8"))
    errors = compare_prompt_fingerprints(baseline, current)
    for count, expected in selected.items():
        path = patient_dir / f"patients_n{count}.txt"
        actual = parse_patient_file(path)
        if actual != expected:
            errors.append(f"{path} does not match the seed-{SEED} selection")

    n100 = parse_patient_file(patient_dir / "patients_n100.txt")
    n300 = parse_patient_file(patient_dir / "patients_n300.txt")
    if n300[:100] != n100:
        errors.append("patients_n100.txt is not the first 100 lines of patients_n300.txt")
    if errors:
        raise AssertionError("asset verification failed:\n- " + "\n- ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    parser.add_argument(
        "--fingerprints", type=Path, default=DEFAULT_FINGERPRINTS
    )
    parser.add_argument("--patient-dir", type=Path, default=DEFAULT_PATIENT_DIR)
    parser.add_argument(
        "--write",
        action="store_true",
        help="establish the baseline assets (default behavior is read-only verification)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow --write to replace existing assets",
    )
    args = parser.parse_args()
    if args.force and not args.write:
        parser.error("--force is valid only with --write")

    runner = load_runner(args.runner)
    cards = load_cards(args.cards)
    fingerprints = build_prompt_fingerprints(
        runner,
        cards,
        runner_path=args.runner,
        cards_path=args.cards,
    )
    selected = patient_lists(cards)

    if args.write:
        write_assets(
            fingerprints,
            selected,
            fingerprints_path=args.fingerprints,
            patient_dir=args.patient_dir,
            force=args.force,
        )
        action = "wrote"
    else:
        verify_assets(
            fingerprints,
            selected,
            fingerprints_path=args.fingerprints,
            patient_dir=args.patient_dir,
        )
        action = "verified"

    print(
        f"{action} {len(fingerprints['conditions'])} prompt fingerprints; "
        f"SYS={fingerprints['sys']['sha1']} "
        f"aggregate={fingerprints['aggregate_sha1']}"
    )
    for count in PATIENT_COUNTS:
        path = args.patient_dir / f"patients_n{count}.txt"
        print(f"{path}: {count} RIDs, sha1={sha1_file(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
