"""Configuration-driven orchestration for :mod:`run_full`.

This module deliberately contains no prompt text.  ``run_full.py`` owns the
immutable experiment definitions and passes its module object to ``main``.
All provider calls are delegated to ``provider_adapters``.
"""

from __future__ import annotations

import argparse
import copy
import errno
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

import yaml

from provider_adapters import build_request_preview, call_provider


SEED = 20260920
DEFAULT_CARDS = Path("data/cards.json")
DEFAULT_PROVIDER_CONFIG = Path("providers.yaml")
DEFAULT_FINGERPRINTS = Path("prompt_fingerprints.json")
THOUGHT_SCENARIOS = (0, 300, 1000, 2500)
PERMANENT_PROVIDER_TYPES = {"openai_compat", "anthropic", "gemini"}
_TRAJECTORIES = {"stable", "mild_decline", "marked_decline"}
_CONFIDENCES = {"low", "moderate", "high"}
_META_LOCK = threading.Lock()


def _credential_field(key: Any) -> bool:
    """Return whether *key* conventionally contains an inline credential.

    Token *counts* such as ``max_tokens`` deliberately do not match.  The
    environment-variable *name* in ``api_key_env`` is configuration, not a
    credential, and must remain available in reproducibility metadata.
    """

    normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
    if normalized == "apikeyenv":
        return False
    return (
        normalized
        in {
            "authorization",
            "auth",
            "apikey",
            "xapikey",
            "password",
            "passwd",
            "secret",
            "token",
            "credential",
            "credentials",
            "privatekey",
        }
        or normalized.endswith("apikey")
        or normalized.endswith("password")
        or normalized.endswith("secret")
        or normalized.endswith("token")
        or normalized.endswith("credential")
        or normalized.endswith("credentials")
        or normalized.endswith("privatekey")
    )


def _credential_paths(value: Any, path: str = "") -> list[str]:
    """Find credential-shaped field names recursively in configuration."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if _credential_field(key):
                found.append(child)
            else:
                found.extend(_credential_paths(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_credential_paths(item, f"{path}[{index}]"))
    return found


def _canonical_path(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def _output_lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


def _pid_is_running(pid: int) -> bool:
    """Return whether *pid* still identifies a live process.

    Permission errors are treated as "alive" so lock recovery always fails
    closed when process ownership cannot be established.
    """

    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        # CPython on Windows reports an absent PID as WinError 87; POSIX uses
        # ESRCH.  Unknown errors are conservatively treated as a live owner.
        if exc.errno == errno.ESRCH or getattr(exc, "winerror", None) == 87:
            return False
        return True
    return True


def _remove_stale_output_lock(lock_path: Path) -> bool:
    """Remove a well-formed lock whose recorded owner is no longer running."""

    try:
        payload = _read_json(lock_path)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    pid = payload.get("pid")
    token = payload.get("token")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if not isinstance(token, str) or not token or _pid_is_running(pid):
        return False

    # Re-read immediately before unlinking and only remove the exact lock that
    # was inspected.  A newly-created or malformed lock remains fail-closed.
    try:
        current = _read_json(lock_path)
        if not isinstance(current, dict):
            return False
        if current.get("pid") != pid or current.get("token") != token:
            return False
        lock_path.unlink()
    except FileNotFoundError:
        return True
    except Exception:
        return False
    return True


@contextmanager
def output_locks(paths: Iterable[Path]):
    """Exclusively lock output files for the lifetime of a run.

    The lock is deliberately a sibling of the JSONL rather than an in-memory
    mutex: two separately launched Python processes must not both append to the
    same result file.  A well-formed lock left by a dead process is recovered
    automatically; live, malformed, or unverifiable locks remain fail-closed.
    """

    unique = sorted({_canonical_path(path): Path(path) for path in paths}.values(), key=_canonical_path)
    acquired: list[tuple[Path, str]] = []
    try:
        for output in unique:
            output.parent.mkdir(parents=True, exist_ok=True)
            lock_path = _output_lock_path(output)
            token = f"{os.getpid()}-{time.time_ns()}-{os.urandom(8).hex()}"
            for attempt in range(2):
                descriptor: int | None = None
                try:
                    descriptor = os.open(
                        lock_path,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                        0o600,
                    )
                    payload = {
                        "token": token,
                        "pid": os.getpid(),
                        "created_at": now_iso(),
                        "output": str(output),
                    }
                    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
                        descriptor = None
                        json.dump(payload, target, ensure_ascii=False)
                        target.write("\n")
                    acquired.append((lock_path, token))
                    break
                except FileExistsError as exc:
                    if attempt == 0 and _remove_stale_output_lock(lock_path):
                        print(f"[resume] removed stale output lock: {lock_path}")
                        continue
                    detail = ""
                    try:
                        detail = lock_path.read_text(encoding="utf-8").strip()[:1000]
                    except Exception:
                        pass
                    suffix = f"; lock contents: {detail}" if detail else ""
                    raise RuntimeError(
                        f"output is locked by another process: {output} "
                        f"({lock_path}){suffix}"
                    ) from exc
                except Exception:
                    if descriptor is not None:
                        os.close(descriptor)
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    raise
        yield
    finally:
        for lock_path, token in reversed(acquired):
            try:
                payload = _read_json(lock_path)
            except FileNotFoundError:
                continue
            except Exception:
                # Never remove a lock that can no longer be proved to be ours.
                continue
            if isinstance(payload, dict) and payload.get("token") == token:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass


def sha1_bytes(value: bytes) -> str:
    return hashlib.sha1(value).hexdigest()


def sha1_text(value: str) -> str:
    return sha1_bytes(value.encode("utf-8"))


def sha1_file(path: str | Path) -> str:
    return sha1_bytes(Path(path).read_bytes())


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def valid_assessment(
    risk: Any,
    trajectory: Any,
    evidence: Any,
    would_change: Any,
    confidence: Any,
) -> bool:
    return (
        isinstance(risk, int)
        and not isinstance(risk, bool)
        and 0 <= risk <= 100
        and isinstance(trajectory, str)
        and trajectory in _TRAJECTORIES
        and isinstance(evidence, list)
        and len(evidence) == 3
        and all(isinstance(item, str) and item.strip() for item in evidence)
        and isinstance(would_change, str)
        and bool(would_change.strip())
        and isinstance(confidence, str)
        and confidence in _CONFIDENCES
    )


def validate_output(output: Any) -> dict[str, Any]:
    if not isinstance(output, dict) or not valid_assessment(
        output.get("risk_24m"),
        output.get("trajectory"),
        output.get("evidence"),
        output.get("would_change_my_mind"),
        output.get("confidence"),
    ):
        raise ValueError("response does not match the required assessment schema")
    return output


def successful_record(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and record.get("err") in (None, "")
        and valid_assessment(
            record.get("risk"),
            record.get("traj"),
            record.get("evidence"),
            record.get("would_change"),
            record.get("conf"),
        )
    )


def _safe_component(value: Any) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value).strip())
    text = text.rstrip(" .")
    return text or "unnamed"


def _thinking_enabled(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value.casefold() not in {"", "disabled", "false", "none", "off"}
    if isinstance(value, Mapping):
        return (
            str(value.get("type", "")).casefold()
            not in {"disabled", "false", "none", "off"}
            and value.get("enabled") is not False
        )
    return bool(value)


def _thinking_label(value: Any) -> str:
    if not _thinking_enabled(value):
        return "off"
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return str(value.get("effort") or value.get("type") or "on")
    return "on"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_cards(path: Path) -> list[dict[str, Any]]:
    cards = _read_json(path)
    if not isinstance(cards, list) or not cards:
        raise ValueError(f"{path} must contain a non-empty JSON array")
    if not all(isinstance(card, dict) and "RID" in card for card in cards):
        raise ValueError(f"every card in {path} must contain RID")
    if len({card["RID"] for card in cards}) != len(cards):
        raise ValueError(f"{path} contains duplicate RID values")
    return cards


def verify_prompt_fingerprints(
    definitions: Any,
    cards: Sequence[dict[str, Any]],
    cards_path: Path,
    fingerprints_path: Path,
) -> dict[str, Any]:
    """Fail closed if any immutable prompt component changed."""

    import prepare_experiment_assets as assets

    baseline = _read_json(fingerprints_path)
    runner_path = Path(definitions.__file__).resolve()
    current = assets.build_prompt_fingerprints(
        definitions,
        cards,
        runner_path=runner_path,
        cards_path=cards_path,
    )
    errors = assets.compare_prompt_fingerprints(baseline, current)
    if errors:
        raise RuntimeError(
            "prompt fingerprint verification failed; refusing to run:\n- "
            + "\n- ".join(errors)
        )
    return baseline


def _coerce_arm_entries(raw: Any) -> dict[str, dict[str, Any]]:
    if isinstance(raw, dict):
        result: dict[str, dict[str, Any]] = {}
        for key, value in raw.items():
            if not isinstance(value, dict):
                raise ValueError(f"arm {key!r} must be a mapping")
            item = copy.deepcopy(value)
            item.setdefault("name", key)
            result[str(key)] = item
        return result
    if isinstance(raw, list):
        result = {}
        for value in raw:
            if not isinstance(value, dict) or not value.get("name"):
                raise ValueError("every arm list entry must contain name")
            name = str(value["name"])
            if name in result:
                raise ValueError(f"duplicate arm name {name!r}")
            result[name] = copy.deepcopy(value)
        return result
    raise ValueError("providers.yaml arms must be a mapping or list")


def _validate_arm(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "name",
        "provider",
        "vendor",
        "endpoint",
        "api_key_env",
        "model_id",
        "send_temperature",
        "temperature",
        "json_mode",
        "extra_body",
        "thinking",
        "price_in",
        "price_out",
        "default_workers",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"arm {name!r} is missing fields: {', '.join(missing)}")
    arm = copy.deepcopy(dict(value))
    if arm["name"] != name:
        raise ValueError(f"arm key {name!r} disagrees with name={arm['name']!r}")
    if arm["provider"] not in PERMANENT_PROVIDER_TYPES:
        raise ValueError(
            f"arm {name!r}: provider must be one of {sorted(PERMANENT_PROVIDER_TYPES)}"
        )
    for field in ("name", "vendor", "endpoint", "api_key_env", "model_id"):
        if not isinstance(arm[field], str) or not arm[field].strip():
            raise ValueError(f"arm {name!r}: {field} must be a non-empty string")
    endpoint = arm["endpoint"]
    try:
        parsed_endpoint = urlsplit(endpoint)
        endpoint_port = parsed_endpoint.port
    except ValueError as exc:
        raise ValueError(f"arm {name!r}: endpoint is not a valid URL") from exc
    if parsed_endpoint.scheme.casefold() != "https" or not parsed_endpoint.hostname:
        raise ValueError(f"arm {name!r}: endpoint must be an https URL")
    if parsed_endpoint.username is not None or parsed_endpoint.password is not None:
        raise ValueError(f"arm {name!r}: endpoint must not contain userinfo")
    if "?" in endpoint or "#" in endpoint:
        raise ValueError(f"arm {name!r}: endpoint must not contain query or fragment")
    if endpoint_port is not None and not 1 <= endpoint_port <= 65535:
        raise ValueError(f"arm {name!r}: endpoint port is invalid")
    if not isinstance(arm["extra_body"], dict):
        raise ValueError(f"arm {name!r}: extra_body must be a mapping")
    credential_paths = _credential_paths(arm)
    if credential_paths:
        raise ValueError(
            f"arm {name!r}: inline credential field(s) are forbidden; use "
            f"api_key_env instead: {', '.join(credential_paths)}"
        )
    if not isinstance(arm["send_temperature"], bool):
        raise ValueError(f"arm {name!r}: send_temperature must be boolean")
    if "send_max_tokens" in arm and not isinstance(arm["send_max_tokens"], bool):
        raise ValueError(f"arm {name!r}: send_max_tokens must be boolean")
    if not isinstance(arm["json_mode"], bool):
        raise ValueError(f"arm {name!r}: json_mode must be boolean")
    if isinstance(arm["default_workers"], bool) or int(arm["default_workers"]) <= 0:
        raise ValueError(f"arm {name!r}: default_workers must be positive")
    arm["default_workers"] = int(arm["default_workers"])
    arm["price_in"] = float(arm["price_in"])
    arm["price_out"] = float(arm["price_out"])
    arm["max_tokens"] = int(arm.get("max_tokens", 500))
    if arm["max_tokens"] <= 0:
        raise ValueError(f"arm {name!r}: max_tokens must be positive")
    if (
        not math.isfinite(arm["price_in"])
        or not math.isfinite(arm["price_out"])
        or arm["price_in"] < 0
        or arm["price_out"] < 0
    ):
        raise ValueError(f"arm {name!r}: prices cannot be negative")
    if arm["send_temperature"]:
        if (
            isinstance(arm["temperature"], bool)
            or not isinstance(arm["temperature"], (int, float))
            or not math.isfinite(float(arm["temperature"]))
        ):
            raise ValueError(f"arm {name!r}: sent temperature must be finite")
    if arm["provider"] == "gemini":
        if arm["send_temperature"]:
            raise ValueError(f"arm {name!r}: Gemini must not send temperature")
        if arm["extra_body"].get("store") is not False:
            raise ValueError(f"arm {name!r}: Gemini extra_body.store must be false")
    if (
        arm["provider"] == "anthropic"
        and _thinking_enabled(arm.get("thinking"))
        and arm["send_temperature"]
    ):
        raise ValueError(
            f"arm {name!r}: Anthropic thinking and temperature cannot be enabled together"
        )
    if arm["provider"] == "anthropic" and arm["json_mode"]:
        raise ValueError(f"arm {name!r}: Anthropic Messages has no JSON mode")
    # Build the exact request skeleton now so --plan fails locally on unsafe
    # extra_body conflicts instead of discovering them after paid work starts.
    build_request_preview(arm)
    return arm


def load_provider_config(path: Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    entries = _coerce_arm_entries(payload.get("arms", payload))
    return {name: _validate_arm(name, value) for name, value in entries.items()}


def _legacy_provider_type(vendor: str) -> str:
    lowered = vendor.casefold()
    if lowered in {"anthropic", "claude"}:
        return "anthropic"
    if lowered in {"gemini", "google"}:
        return "gemini"
    return "openai_compat"


def _legacy_endpoint(vendor: str, provider_type: str, model_id: str) -> str:
    if os.environ.get("LLM_API"):
        return os.environ["LLM_API"]
    if provider_type == "anthropic":
        return "https://api.anthropic.com/v1/messages"
    if provider_type == "gemini":
        return "https://generativelanguage.googleapis.com/v1beta/interactions"
    if vendor.casefold() == "deepseek":
        return "https://api.deepseek.com/chat/completions"
    raise ValueError("legacy mode for this provider requires LLM_API")


def legacy_arm(args: argparse.Namespace) -> dict[str, Any]:
    vendor = args.provider or os.environ.get("LLM_PROVIDER", "deepseek")
    provider_type = _legacy_provider_type(vendor)
    minimal_openai_request = vendor.casefold() == "yunzhuhub"
    model_id = args.model or os.environ.get("LLM_MODEL", "deepseek-chat")
    if minimal_openai_request:
        incompatible = [
            flag
            for flag, value in (
                ("--thinking", args.thinking),
                ("--temperature", args.temperature),
                ("--max-tokens", args.max_tokens),
            )
            if value is not None
        ]
        if incompatible:
            raise ValueError(
                "legacy yunzhuhub uses a minimal model+messages request and "
                "cannot accept: " + ", ".join(incompatible)
            )
    thinking_arg = args.thinking
    thinking: Any = thinking_arg
    extra: dict[str, Any] = {}
    if provider_type == "openai_compat" and thinking_arg:
        if thinking_arg in {"low", "high", "max"}:
            extra["thinking"] = {"type": "enabled"}
            extra["reasoning_effort"] = thinking_arg
            thinking = {"type": "enabled", "effort": thinking_arg}
        else:
            extra["thinking"] = {"type": thinking_arg}
            thinking = {"type": thinking_arg}
    elif provider_type == "gemini":
        extra["store"] = False
    elif provider_type == "anthropic" and isinstance(thinking_arg, str):
        if thinking_arg in {"disabled", "none", "off"}:
            thinking = {"type": "disabled"}
        else:
            thinking = {"type": "adaptive"}
            if thinking_arg not in {"adaptive", "enabled"}:
                extra["output_config"] = {"effort": thinking_arg}
    send_temperature = provider_type != "gemini" and not minimal_openai_request
    temperature = 0.0 if args.temperature is None else args.temperature
    if _thinking_enabled(thinking):
        send_temperature = False
    arm_name = model_id
    suffixes: list[str] = []
    if minimal_openai_request:
        suffixes.append("minimal")
    if _thinking_enabled(thinking):
        suffixes.append("thinking-" + _safe_component(_thinking_label(thinking)))
    if send_temperature and temperature != 0.0:
        suffixes.append("temp-" + _safe_component(format(temperature, "g")))
    if suffixes:
        arm_name += "@" + "_".join(suffixes)
    arm = {
        "name": arm_name,
        "provider": provider_type,
        "vendor": vendor,
        "endpoint": _legacy_endpoint(vendor, provider_type, model_id),
        "api_key_env": "LLM_KEY",
        "model_id": model_id,
        "send_temperature": send_temperature,
        "temperature": temperature,
        "json_mode": provider_type != "anthropic" and not minimal_openai_request,
        "extra_body": extra,
        "thinking": thinking,
        "price_in": 0.0,
        "price_out": 0.0,
        "default_workers": args.workers or 8,
        "max_tokens": args.max_tokens or 500,
        "legacy": True,
    }
    if minimal_openai_request:
        arm["send_max_tokens"] = False
    return _validate_arm(arm_name, arm)


def _apply_cli_overrides(arm: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    arm = copy.deepcopy(arm)
    if args.model:
        arm["model_id"] = args.model
    if args.provider:
        arm["vendor"] = args.provider
    if args.temperature is not None:
        arm["temperature"] = args.temperature
    if args.thinking is not None:
        arm["thinking"] = args.thinking
        if arm["provider"] == "openai_compat":
            if args.thinking in {"low", "high", "max"}:
                arm["extra_body"]["thinking"] = {"type": "enabled"}
                arm["extra_body"]["reasoning_effort"] = args.thinking
            else:
                arm["extra_body"]["thinking"] = {"type": args.thinking}
                arm["extra_body"].pop("reasoning_effort", None)
        if arm["provider"] == "anthropic":
            arm["thinking"] = (
                {"type": "adaptive"}
                if args.thinking == "adaptive"
                else (
                    {"type": "disabled"}
                    if args.thinking == "disabled"
                    else args.thinking
                )
            )
            if args.thinking != "disabled":
                arm["send_temperature"] = False
    if args.max_tokens is not None:
        arm["max_tokens"] = args.max_tokens
    if arm["provider"] == "gemini":
        arm["send_temperature"] = False
        arm["extra_body"]["store"] = False
    return _validate_arm(arm["name"], arm)


def select_arms(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    requested: list[str] = []
    if args.arm:
        requested = [args.arm]
    elif args.arms:
        requested = [part.strip() for part in args.arms.split(",") if part.strip()]
    if not requested:
        arm = legacy_arm(args)
        return {arm["name"]: arm}
    incompatible = [
        flag
        for flag, value in (
            ("--provider", args.provider),
            ("--model", args.model),
            ("--thinking", args.thinking),
            ("--temperature", args.temperature),
            ("--max-tokens", args.max_tokens),
        )
        if value is not None
    ]
    if incompatible:
        raise ValueError(
            "configured --arm/--arms cannot be combined with request overrides: "
            + ", ".join(incompatible)
        )
    configured = load_provider_config(args.provider_config)
    if requested == ["all"]:
        requested = list(configured)
    if len(requested) != len(set(requested)):
        raise ValueError("--arms contains duplicate arm names")
    unknown = [name for name in requested if name not in configured]
    if unknown:
        raise ValueError(
            f"unknown arm(s): {', '.join(unknown)}; available: {', '.join(configured)}"
        )
    return {name: _apply_cli_overrides(configured[name], args) for name in requested}


def parse_rid_file(path: Path) -> list[int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        rids = [int(line.strip()) for line in lines if line.strip()]
    except ValueError as exc:
        raise ValueError(f"{path} contains a non-integer RID") from exc
    if not rids:
        raise ValueError(f"{path} is empty")
    if len(rids) != len(set(rids)):
        raise ValueError(f"{path} contains duplicate RIDs")
    return rids


def validate_patient_prefix_files(
    directory: Path,
    cards_by_rid: Mapping[int, dict[str, Any]] | None = None,
) -> None:
    n100 = directory / "patients_n100.txt"
    n300 = directory / "patients_n300.txt"
    missing = [str(path) for path in (n100, n300) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "canonical patient files are required: " + ", ".join(missing)
        )
    first = parse_rid_file(n100)
    larger = parse_rid_file(n300)
    if len(first) != 100 or len(larger) != 300 or larger[:100] != first:
        raise ValueError("patients_n100.txt must be the first 100 rows of patients_n300.txt")
    if cards_by_rid is not None:
        population = sorted(cards_by_rid)
        random.Random(SEED).shuffle(population)
        if first != population[:100] or larger != population[:300]:
            raise ValueError(
                f"canonical patient files do not match the seed-{SEED} legacy draw"
            )


def select_patients(
    cards_by_rid: Mapping[int, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[int], str, str | None, random.Random]:
    population = sorted(cards_by_rid)
    rnd = random.Random(SEED)
    rnd.shuffle(population)  # Preserve the legacy RNG state and draw logic.
    rid_source: str | None = None
    if args.rid_file:
        file_order = parse_rid_file(args.rid_file)
        unknown = sorted(set(file_order) - set(cards_by_rid))
        if unknown:
            raise ValueError(f"RID file contains IDs absent from cards: {unknown[:10]}")
        rids = sorted(file_order)  # Legacy --n sorted the selected sample for expansion.
        rid_hash = sha1_file(args.rid_file)
        rid_source = str(args.rid_file)
    elif args.n:
        if args.n < 1 or args.n > len(population):
            raise ValueError(f"--n must be in 1..{len(population)}")
        rids = sorted(population[: args.n])
        rid_hash = sha1_text("".join(f"{rid}\n" for rid in population[: args.n]))
    else:
        rids = population
        rid_hash = sha1_text("".join(f"{rid}\n" for rid in rids))
    return rids, rid_hash, rid_source, rnd


def resolve_conditions(definitions: Any, expression: str) -> list[str]:
    conds = definitions.SETS.get(expression, expression.split(","))
    if len(conds) != len(set(conds)):
        raise ValueError("condition list contains duplicates")
    missing = [cond for cond in conds if cond not in definitions.CONDS]
    if missing:
        raise ValueError(f"unknown conditions: {', '.join(missing)}")
    return list(conds)


def build_task_schedule(
    arm_names: Sequence[str],
    rids: Sequence[int],
    conds: Sequence[str],
    repeats: int,
    rnd: random.Random,
    *,
    sentinel_n: int,
    sentinel_every: int,
    core_conditions: Sequence[str],
    smoke: int | None = None,
) -> tuple[list[tuple[str, int, str, int]], list[tuple[str, int, str, int]]]:
    """Return ``(selected_schedule, complete_schedule)``.

    ``complete_schedule`` is always the full run and powers cost projection.
    For smoke mode, ``selected_schedule`` contains exactly N calls per arm and
    deliberately omits sentinel tasks.
    """

    if repeats < 1:
        raise ValueError("--repeats must be positive")
    main = [
        (arm, rid, cond, rep)
        for arm in arm_names
        for rid in rids
        for cond in conds
        for rep in range(repeats)
    ]
    rnd.shuffle(main)
    sent_rids = sorted(rids)[: max(0, sentinel_n)]
    sentinel_pairs = [
        (rid, cond) for rid in sent_rids for cond in core_conditions
    ]
    # The original protocol inserts one full sentinel group after every 1,500
    # calls on the global time axis.  In multi-arm runs that group contains
    # every arm, but the interval itself remains global rather than per-arm.
    interval = max(1, sentinel_every)
    complete: list[tuple[str, int, str, int]] = []
    index = 0
    block = 2
    while index < len(main):
        complete.extend(main[index : index + interval])
        # Keep the same patient-condition adjacent across arms, and rotate the
        # first arm at each checkpoint so one arm is not systematically earlier.
        shift = (block - 2) % max(len(arm_names), 1)
        arm_order = list(arm_names[shift:]) + list(arm_names[:shift])
        complete.extend(
            (arm, rid, cond, -block)
            for rid, cond in sentinel_pairs
            for arm in arm_order
        )
        index += interval
        block += 1

    if smoke is None:
        return complete, complete
    if smoke < 1:
        raise ValueError("--smoke must be positive")
    selected: list[tuple[str, int, str, int]] = []
    counts: Counter[str] = Counter()
    for task in main:
        arm = task[0]
        if counts[arm] < smoke:
            selected.append(task)
            counts[arm] += 1
        if all(counts[name] >= smoke for name in arm_names):
            break
    if any(counts[name] < smoke for name in arm_names):
        raise ValueError("--smoke exceeds the number of available tasks")
    return selected, complete


def _conditions_label(expression: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.-]+", expression):
        return expression
    return "custom_" + sha1_text(expression)[:8]


def output_path_for(
    arm: Mapping[str, Any],
    args: argparse.Namespace,
    patient_count: int,
) -> Path:
    root = Path("runs")
    if args.smoke is not None:
        root /= "_smoke"
    folder = root / _safe_component(arm["vendor"]) / _safe_component(arm["name"])
    filename = (
        f"n{patient_count}_r{args.repeats}_"
        f"{_conditions_label(args.conds)}.jsonl"
    )
    return folder / filename


def _same_temperature(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def _record_identity_error(
    row: Mapping[str, Any], arm: Mapping[str, Any], month: int
) -> str | None:
    arm_name = str(arm["name"])
    row_arm = row.get("arm", row.get("model"))
    if row_arm != arm_name:
        return f"arm={row_arm!r}, expected {arm_name!r}"
    if row.get("model") != arm["model_id"]:
        return f"model={row.get('model')!r}, expected {arm['model_id']!r}"
    if row.get("provider") != arm["vendor"]:
        return f"provider={row.get('provider')!r}, expected {arm['vendor']!r}"
    expected_temperature = arm["temperature"] if arm["send_temperature"] else None
    if not _same_temperature(row.get("temperature"), expected_temperature):
        return (
            f"temperature={row.get('temperature')!r}, "
            f"expected {expected_temperature!r}"
        )
    try:
        row_month = int(row.get("month", -1))
    except (TypeError, ValueError):
        row_month = -1
    if row_month != int(month):
        return f"month={row.get('month')!r}, expected {month!r}"
    if "provider_type" in row and row.get("provider_type") != arm["provider"]:
        return (
            f"provider_type={row.get('provider_type')!r}, "
            f"expected {arm['provider']!r}"
        )
    if "model_id" in row and row.get("model_id") != arm["model_id"]:
        return f"model_id={row.get('model_id')!r}, expected {arm['model_id']!r}"
    if "thinking" in row:
        if row.get("thinking") != arm.get("thinking"):
            return "thinking configuration differs"
    elif _thinking_enabled(arm.get("thinking")):
        return "existing row has no thinking identity for a thinking-enabled arm"
    return None


def load_seen(
    path: Path, arm: Mapping[str, Any], month: int
) -> set[tuple[str, int, str, int]]:
    arm_name = str(arm["name"])
    seen: set[tuple[str, int, str, int]] = set()
    if not path.exists():
        return seen
    if path.stat().st_size:
        with path.open("rb") as binary:
            binary.seek(-1, os.SEEK_END)
            if binary.read(1) != b"\n":
                raise ValueError(
                    f"refusing to append to {path}: final JSONL line is incomplete"
                )
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"refusing to append to {path}: invalid JSON on line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"refusing to append to {path}: line {line_number} is not an object"
                )
            mismatch = _record_identity_error(row, arm, month)
            if mismatch:
                raise ValueError(
                    f"refusing to mix run identities in {path} line {line_number}: "
                    + mismatch
                )
            try:
                if successful_record(row):
                    seen.add((arm_name, int(row["RID"]), row["cond"], int(row["rep"])))
            except (KeyError, TypeError, ValueError):
                continue
    return seen


def _schedule_sha1(tasks: Sequence[tuple[str, int, str, int]]) -> str:
    digest = hashlib.sha1()
    for task in tasks:
        digest.update(
            (json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _experiment_identity(
    *,
    arm_name: str,
    arm_names: Sequence[str],
    args: argparse.Namespace,
    schedule: Sequence[tuple[str, int, str, int]],
    complete_schedule: Sequence[tuple[str, int, str, int]],
    rid_hash: str,
    cards_sha1: str,
    fingerprints_sha1: str,
    fingerprint_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    selected_for_arm = [task for task in schedule if task[0] == arm_name]
    complete_for_arm = [task for task in complete_schedule if task[0] == arm_name]
    return {
        "schema_version": 1,
        "seed": SEED,
        "arm": arm_name,
        "arms": list(arm_names),
        "rid_file_sha1": rid_hash,
        "cards_file_sha1": cards_sha1,
        "fingerprints_file_sha1": fingerprints_sha1,
        "prompt_aggregate_sha1": fingerprint_baseline.get("aggregate_sha1"),
        "immutable_source_sha1": fingerprint_baseline.get("immutable_source", {}).get(
            "sha1"
        ),
        "conditions_expression": args.conds,
        "repeats": args.repeats,
        "month": args.month,
        "sentinel_n": args.sentinel_n,
        "sentinel_every": args.sentinel_every,
        "smoke_calls_per_arm": args.smoke,
        "selected_schedule_sha1": _schedule_sha1(schedule),
        "complete_schedule_sha1": _schedule_sha1(complete_schedule),
        "arm_selected_schedule_sha1": _schedule_sha1(selected_for_arm),
        "arm_complete_schedule_sha1": _schedule_sha1(complete_for_arm),
    }


def _metadata_identities_for_output(meta_path: Path, output: Path) -> list[Any]:
    if not meta_path.exists():
        return []
    try:
        payload = _read_json(meta_path)
    except Exception as exc:
        raise ValueError(f"cannot verify unreadable metadata: {meta_path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("runs", {}), dict):
        raise ValueError(f"cannot verify malformed metadata: {meta_path}")
    target = _canonical_path(output)
    identities: list[Any] = []
    for run in payload.get("runs", {}).values():
        if not isinstance(run, dict) or not isinstance(run.get("output"), str):
            continue
        if _canonical_path(run["output"]) == target:
            identities.append(run.get("experiment_identity"))
    return identities


def _validate_complete_legacy_output(
    path: Path,
    arm: Mapping[str, Any],
    args: argparse.Namespace,
    expected_tasks: set[tuple[str, int, str, int]],
    cards_by_rid: Mapping[int, dict[str, Any]],
    definitions: Any,
    rid_hash: str,
) -> None:
    """Allow a metadata-free historical file only when it is fully complete."""

    successful: set[tuple[str, int, str, int]] = set()
    prompt_hashes: dict[tuple[int, str], str] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("row is not an object")
                mismatch = _record_identity_error(row, arm, args.month)
                if mismatch:
                    raise ValueError(mismatch)
                key = (str(arm["name"]), int(row["RID"]), str(row["cond"]), int(row["rep"]))
                if key not in expected_tasks:
                    raise ValueError(f"task {key!r} is not in the current schedule")
                if "rid_file_sha1" in row and row.get("rid_file_sha1") != rid_hash:
                    raise ValueError("RID-list fingerprint differs")
                prompt_key = (key[1], key[2])
                if prompt_key not in prompt_hashes:
                    prompt = definitions.build(
                        cards_by_rid[key[1]][f"card{args.month}"],
                        definitions.CONDS[key[2]],
                    )
                    prompt_hashes[prompt_key] = sha1_text(prompt)
                if row.get("prompt_sha1") != prompt_hashes[prompt_key]:
                    raise ValueError("prompt fingerprint differs")
                if successful_record(row):
                    successful.add(key)
            except Exception as exc:
                raise ValueError(
                    f"cannot safely adopt metadata-free output {path}, line {line_number}: {exc}"
                ) from exc
    if successful != expected_tasks:
        missing = len(expected_tasks - successful)
        raise ValueError(
            f"refusing to resume metadata-free incomplete output {path}: "
            f"{missing} scheduled successful calls are missing"
        )


def _validate_output_experiment_identity(
    path: Path,
    arm: Mapping[str, Any],
    identity: Mapping[str, Any],
    args: argparse.Namespace,
    complete_schedule: Sequence[tuple[str, int, str, int]],
    cards_by_rid: Mapping[int, dict[str, Any]],
    definitions: Any,
    rid_hash: str,
) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    meta_path = path.parent / "run_meta.json"
    recorded = _metadata_identities_for_output(meta_path, path)
    if recorded:
        if any(value != identity for value in recorded):
            raise ValueError(
                f"refusing to append: experiment identity differs for {path}"
            )
        return
    expected = {task for task in complete_schedule if task[0] == arm["name"]}
    _validate_complete_legacy_output(
        path, arm, args, expected, cards_by_rid, definitions, rid_hash
    )


def _redact_config(arm: Mapping[str, Any]) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                if _credential_field(key):
                    result[key] = "<redacted>"
                else:
                    result[key] = clean(item)
            return result
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value

    return clean(copy.deepcopy(dict(arm)))


def request_preview(arm: Mapping[str, Any]) -> dict[str, Any]:
    return build_request_preview(arm)


def _update_run_meta(path: Path, arm: Mapping[str, Any], run_id: str, run: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _META_LOCK:
        if path.exists():
            try:
                payload = _read_json(path)
            except Exception as exc:
                raise ValueError(f"refusing to overwrite unreadable metadata: {path}") from exc
        else:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError(f"refusing to overwrite non-object metadata: {path}")
        safe_arm = _redact_config(arm)
        if payload.get("arm") not in (None, arm["name"]):
            raise ValueError(f"metadata arm identity differs in {path}")
        if payload.get("arm_config") not in (None, safe_arm):
            raise ValueError(f"metadata arm configuration differs in {path}")
        payload["schema_version"] = 1
        payload["arm"] = arm["name"]
        payload["arm_config"] = safe_arm
        runs = payload.setdefault("runs", {})
        if not isinstance(runs, dict):
            raise ValueError(f"metadata runs field is not an object in {path}")
        # Defense in depth: configuration validation rejects inline secrets,
        # while metadata is independently scrubbed in case a future field or
        # legacy caller supplies one.
        runs[run_id] = _redact_config(run)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)


def _price(total_input: int, total_output: int, arm: Mapping[str, Any]) -> float:
    return (
        total_input * float(arm["price_in"])
        + total_output * float(arm["price_out"])
    ) / 1_000_000


def print_plan(
    schedules: Sequence[tuple[str, int, str, int]],
    complete: Sequence[tuple[str, int, str, int]],
    arms: Mapping[str, Mapping[str, Any]],
    patient_count: int,
    cond_count: int,
    repeats: int,
    workers: int,
    rid_hash: str,
    smoke: int | None,
) -> None:
    selected_counts = Counter(task[0] for task in schedules)
    full_counts = Counter(task[0] for task in complete)
    print(f"患者 {patient_count}  条件 {cond_count}  重复 {repeats}")
    print(f"病人名单 SHA1 {rid_hash}")
    print(f"全局并发 {workers}  arms {', '.join(arms)}")
    for name, arm in arms.items():
        if smoke is not None:
            print(f"\n[{name}] smoke 调用 {selected_counts[name]} / 全轮 {full_counts[name]}")
        else:
            print(f"\n[{name}] 调用 {selected_counts[name]}")
        calls = full_counts[name]
        for thought_tokens in THOUGHT_SCENARIOS:
            prompt_tokens = calls * 750
            completion_tokens = calls * (220 + thought_tokens)
            cost = _price(prompt_tokens, completion_tokens, arm)
            print(
                f"  思考{thought_tokens:4d}/call: 输入{prompt_tokens/1e6:6.1f}M "
                f"输出{completion_tokens/1e6:6.1f}M  估算 ${cost:,.2f}"
            )


def _script_hashes(definitions: Any) -> dict[str, str]:
    paths = [Path(definitions.__file__).resolve(), Path(__file__).resolve()]
    try:
        import provider_adapters

        paths.append(Path(provider_adapters.__file__).resolve())
    except Exception:
        pass
    return {path.name: sha1_file(path) for path in paths}


def validate_cli_args(args: argparse.Namespace) -> None:
    checks = (
        (args.repeats >= 1, "--repeats must be positive"),
        (args.month >= 0, "--month cannot be negative"),
        (args.n >= 0, "--n cannot be negative"),
        (args.sentinel_n >= 0, "--sentinel-n cannot be negative"),
        (args.sentinel_every >= 1, "--sentinel-every must be positive"),
        (args.retries >= 1, "--retries must be positive"),
        (
            math.isfinite(args.timeout) and args.timeout > 0,
            "--timeout must be finite and positive",
        ),
        (
            args.smoke is None or args.smoke >= 1,
            "--smoke must be positive",
        ),
        (
            args.max_tokens is None or args.max_tokens >= 1,
            "--max-tokens must be positive",
        ),
        (
            args.temperature is None or math.isfinite(args.temperature),
            "--temperature must be finite",
        ),
    )
    for valid, message in checks:
        if not valid:
            raise ValueError(message)


def validate_cards_for_month(
    cards_by_rid: Mapping[int, dict[str, Any]], rids: Sequence[int], month: int
) -> None:
    field = f"card{month}"
    invalid = [rid for rid in rids if not isinstance(cards_by_rid[rid].get(field), str)]
    if invalid:
        raise ValueError(
            f"selected patients are missing a string {field}: {invalid[:10]}"
        )


def execute(
    definitions: Any,
    args: argparse.Namespace,
    cards_by_rid: Mapping[int, dict[str, Any]],
    arms: Mapping[str, dict[str, Any]],
    schedule: Sequence[tuple[str, int, str, int]],
    complete_schedule: Sequence[tuple[str, int, str, int]],
    paths: Mapping[str, Path],
    rid_hash: str,
    rid_source: str | None,
    fingerprint_baseline: Mapping[str, Any],
    workers: int,
) -> int:
    with output_locks(paths.values()):
        return _execute_locked(
            definitions,
            args,
            cards_by_rid,
            arms,
            schedule,
            complete_schedule,
            paths,
            rid_hash,
            rid_source,
            fingerprint_baseline,
            workers,
        )


def _execute_locked(
    definitions: Any,
    args: argparse.Namespace,
    cards_by_rid: Mapping[int, dict[str, Any]],
    arms: Mapping[str, dict[str, Any]],
    schedule: Sequence[tuple[str, int, str, int]],
    complete_schedule: Sequence[tuple[str, int, str, int]],
    paths: Mapping[str, Path],
    rid_hash: str,
    rid_source: str | None,
    fingerprint_baseline: Mapping[str, Any],
    workers: int,
) -> int:
    run_id = sha1_text(
        "|".join(arms) + f"|{args.conds}|{args.repeats}|{time.time()}"
    )[:10]
    started = now_iso()
    prompt_summary = {
        "file": str(args.fingerprints),
        "file_sha1": sha1_file(args.fingerprints),
        "aggregate_sha1": fingerprint_baseline.get("aggregate_sha1"),
        "selection_seed": fingerprint_baseline.get("selection_seed"),
        "immutable_source_sha1": fingerprint_baseline.get(
            "immutable_source", {}
        ).get("sha1"),
        "sys_sha1": fingerprint_baseline.get("sys", {}).get("sha1"),
        "conditions": {
            name: value.get("build_sha1")
            for name, value in fingerprint_baseline.get("conditions", {}).items()
        },
    }
    script_hashes = _script_hashes(definitions)
    complete_counts = Counter(task[0] for task in complete_schedule)
    cards_sha1 = sha1_file(args.cards)
    fingerprints_sha1 = sha1_file(args.fingerprints)
    experiment_identities = {
        name: _experiment_identity(
            arm_name=name,
            arm_names=list(arms),
            args=args,
            schedule=schedule,
            complete_schedule=complete_schedule,
            rid_hash=rid_hash,
            cards_sha1=cards_sha1,
            fingerprints_sha1=fingerprints_sha1,
            fingerprint_baseline=fingerprint_baseline,
        )
        for name in arms
    }
    for name, arm in arms.items():
        _validate_output_experiment_identity(
            paths[name],
            arm,
            experiment_identities[name],
            args,
            complete_schedule,
            cards_by_rid,
            definitions,
            rid_hash,
        )

    seen = {
        name: load_seen(paths[name], arm, args.month)
        for name, arm in arms.items()
    }
    todo = [task for task in schedule if (task[0], task[1], task[2], task[3]) not in seen[task[0]]]
    for name in arms:
        print(
            f"[{name}] 输出 {paths[name]}  已有 {len(seen[name])}  "
            f"待跑 {sum(task[0] == name for task in todo)}"
        )

    keys: dict[str, str] = {}
    todo_arms = {task[0] for task in todo}
    for name, arm in arms.items():
        if name not in todo_arms:
            continue
        key = os.environ.get(str(arm["api_key_env"]))
        if not key:
            raise RuntimeError(
                f"arm {name!r} requires environment variable {arm['api_key_env']}"
            )
        keys[name] = key

    condition_order = list(dict.fromkeys(task[2] for task in complete_schedule))
    patient_count = len({task[1] for task in complete_schedule if task[3] >= 0})
    provider_config_sha1 = (
        sha1_file(args.provider_config) if args.provider_config.is_file() else None
    )
    canonical_patient_hashes = {
        path.name: sha1_file(path)
        for path in (
            args.patient_dir / "patients_n100.txt",
            args.patient_dir / "patients_n300.txt",
        )
    }

    base_meta: dict[str, dict[str, Any]] = {}
    for name, arm in arms.items():
        run = {
            "run_id": run_id,
            "status": "running",
            "output": str(paths[name]),
            "started_at": started,
            "ended_at": None,
            "request_parameters": request_preview(arm),
            "rid_file": rid_source,
            "rid_file_sha1": rid_hash,
            "prompt_fingerprints": prompt_summary,
            "cards_file": str(args.cards),
            "cards_file_sha1": sha1_file(args.cards),
            "provider_config_file": str(args.provider_config),
            "provider_config_sha1": provider_config_sha1,
            "canonical_patient_files": canonical_patient_hashes,
            "experiment_identity": experiment_identities[name],
            "script_sha1": script_hashes.get(Path(definitions.__file__).name),
            "script_sha1s": script_hashes,
            "experiment": {
                "seed": SEED,
                "patients": patient_count,
                "conditions_expression": args.conds,
                "conditions": condition_order,
                "repeats": args.repeats,
                "month": args.month,
                "sentinel_n": args.sentinel_n,
                "sentinel_every": args.sentinel_every,
                "smoke_calls_per_arm": args.smoke,
                "global_workers": workers,
                "arm_worker_limit": (
                    workers
                    if len(arms) == 1 and args.workers is not None
                    else min(workers, int(arm["default_workers"]))
                ),
                "retries": args.retries,
                "timeout_seconds": args.timeout,
            },
            "planned_calls": sum(task[0] == name for task in schedule),
            "full_run_calls": complete_counts[name],
            "resume_existing_successes": len(seen[name]),
        }
        base_meta[name] = run
        _update_run_meta(paths[name].parent / "run_meta.json", arm, run_id, run)

    handles: dict[str, Any] = {}
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        handles[name] = path.open("a", encoding="utf-8", newline="\n")

    stats: dict[str, dict[str, Any]] = {
        name: {
            "completed": 0,
            "failures": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "visible_output_tokens": 0,
            "thought_tokens": 0,
            "billing_usage_calls": 0,
            "usage_reported_calls": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "visible_output_tokens": 0,
                "thought_tokens": 0,
            },
            "errors": [],
        }
        for name in arms
    }
    global_stats = {"completed": 0, "failures": 0, "errors": []}
    arm_worker_limits = {
        name: (
            workers
            if len(arms) == 1 and args.workers is not None
            else min(workers, int(arm["default_workers"]))
        )
        for name, arm in arms.items()
    }
    aborted_reason: str | None = None
    t_start = time.time()

    def call_task(task: tuple[str, int, str, int]) -> tuple[dict[str, Any], str]:
        arm_name, rid, cond, rep = task
        arm = arms[arm_name]
        cfg = definitions.CONDS[cond]
        prompt = definitions.build(cards_by_rid[rid][f"card{args.month}"], cfg)
        output, usage, latency_ms, attempts, err = call_provider(
            arm,
            keys[arm_name],
            definitions.SYS,
            prompt,
            validate_output,
            retries=args.retries,
            timeout=args.timeout,
        )
        usage = dict(usage or {})
        usage_raw = usage.pop("usage_raw", {})
        record = {
            "run_id": run_id,
            "ts": now_iso(),
            "provider": arm["vendor"],
            "model": arm["model_id"],
            "temperature": arm["temperature"] if arm["send_temperature"] else None,
            "RID": rid,
            "month": args.month,
            "cond": cond,
            "rep": rep,
            "is_sentinel": rep < 0,
            "cond_family": cfg.get("family"),
            "belief_level": cfg.get("level"),
            "source": cfg.get("header"),
            "firewall": bool(cfg.get("firewall")),
            "dx24": cards_by_rid[rid].get("dx24"),
            "dx36": cards_by_rid[rid].get("dx36"),
            "target_dx": cards_by_rid[rid].get("dx36"),
            "prompt_sha1": sha1_text(prompt),
            "prompt_chars": len(prompt),
            "risk": (output or {}).get("risk_24m"),
            "traj": (output or {}).get("trajectory"),
            "conf": (output or {}).get("confidence"),
            "evidence": (output or {}).get("evidence"),
            "would_change": (output or {}).get("would_change_my_mind"),
            "usage": usage,
            "latency_ms": latency_ms,
            "attempts": attempts,
            "err": err,
            # New fields are additive; old field names above remain untouched.
            "arm": arm_name,
            "provider_type": arm["provider"],
            "model_id": arm["model_id"],
            "thinking": arm["thinking"],
            "sent_temperature": bool(arm["send_temperature"]),
            "usage_raw": usage_raw,
            "rid_file_sha1": rid_hash,
        }
        return record, arm_name

    def record_result(record: dict[str, Any], arm_name: str) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        handles[arm_name].write(line)
        handles[arm_name].flush()
        arm_stats = stats[arm_name]
        arm_stats["completed"] += 1
        global_stats["completed"] += 1
        failed = bool(record.get("err"))
        if failed:
            arm_stats["failures"] += 1
            global_stats["failures"] += 1
            if len(arm_stats["errors"]) < 3:
                arm_stats["errors"].append(record["err"])
            if len(global_stats["errors"]) < 3:
                global_stats["errors"].append(record["err"])
        usage = record.get("usage") or {}
        usage_reported = usage.get("usage_reported")
        if usage_reported is None:
            usage_reported = bool(record.get("usage_raw"))
        if (
            usage_reported
            and usage.get("usage_complete", True)
            and usage.get("prompt_tokens") is not None
            and usage.get("completion_tokens") is not None
        ):
            arm_stats["billing_usage_calls"] += 1
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "visible_output_tokens",
            "thought_tokens",
        ):
            value = usage.get(field)
            if value is not None:
                arm_stats[field] += int(value)
                arm_stats["usage_reported_calls"][field] += 1

    global_guard_checked = False
    arm_guard_checked = {name: False for name in arms}

    def guard_failure_reason() -> str | None:
        nonlocal global_guard_checked
        if not global_guard_checked and global_stats["completed"] >= 30:
            global_guard_checked = True
            if global_stats["failures"] > 15:
                return (
                    f"前30次调用全局失败 {global_stats['failures']} 次，超过50%"
                )
        for name, arm_stats in stats.items():
            if not arm_guard_checked[name] and arm_stats["completed"] >= 30:
                arm_guard_checked[name] = True
                if arm_stats["failures"] > 15:
                    return (
                        f"arm {name} 前30次调用失败 {arm_stats['failures']} 次，超过50%"
                    )
        return None

    pending_tasks = deque(todo)
    futures: dict[Any, tuple[str, int, str, int]] = {}
    in_flight: Counter[str] = Counter()

    def submit_until_full(pool: ThreadPoolExecutor) -> None:
        if aborted_reason is not None:
            return
        while len(futures) < workers and pending_tasks:
            submitted = False
            for _ in range(len(pending_tasks)):
                task = pending_tasks.popleft()
                arm_name = task[0]
                global_warmup_full = (
                    not global_guard_checked
                    and global_stats["completed"] + len(futures) >= 30
                )
                arm_warmup_full = (
                    not arm_guard_checked[arm_name]
                    and stats[arm_name]["completed"] + in_flight[arm_name] >= 30
                )
                if (
                    in_flight[arm_name] >= arm_worker_limits[arm_name]
                    or global_warmup_full
                    or arm_warmup_full
                ):
                    pending_tasks.append(task)
                    continue
                future = pool.submit(call_task, task)
                futures[future] = task
                in_flight[arm_name] += 1
                submitted = True
                if len(futures) >= workers:
                    break
            if not submitted:
                return

    crashed_error: str | None = None
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            submit_until_full(pool)
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    task = futures.pop(future)
                    in_flight[task[0]] -= 1
                    try:
                        record, arm_name = future.result()
                    except Exception as exc:  # Defensive: adapters normally return errors.
                        arm_name, rid, cond, rep = task
                        cfg = definitions.CONDS[cond]
                        prompt = None
                        try:
                            card_text = cards_by_rid[rid].get(f"card{args.month}")
                            if isinstance(card_text, str):
                                prompt = definitions.build(card_text, cfg)
                        except Exception:
                            prompt = None
                        record = {
                            "run_id": run_id,
                            "ts": now_iso(),
                            "provider": arms[arm_name]["vendor"],
                            "model": arms[arm_name]["model_id"],
                            "temperature": (
                                arms[arm_name]["temperature"]
                                if arms[arm_name]["send_temperature"]
                                else None
                            ),
                            "RID": rid,
                            "month": args.month,
                            "cond": cond,
                            "rep": rep,
                            "is_sentinel": rep < 0,
                            "cond_family": cfg.get("family"),
                            "belief_level": cfg.get("level"),
                            "source": cfg.get("header"),
                            "firewall": bool(cfg.get("firewall")),
                            "dx24": cards_by_rid[rid].get("dx24"),
                            "dx36": cards_by_rid[rid].get("dx36"),
                            "target_dx": cards_by_rid[rid].get("dx36"),
                            "prompt_sha1": sha1_text(prompt) if prompt is not None else None,
                            "prompt_chars": len(prompt) if prompt is not None else None,
                            "risk": None,
                            "traj": None,
                            "conf": None,
                            "evidence": None,
                            "would_change": None,
                            "usage": {
                                "prompt_tokens": None,
                                "completion_tokens": None,
                                "visible_output_tokens": None,
                                "thought_tokens": None,
                                "usage_reported": False,
                            },
                            "latency_ms": 0,
                            "attempts": 0,
                            "err": f"runner exception: {type(exc).__name__}: {exc}",
                            "arm": arm_name,
                            "provider_type": arms[arm_name]["provider"],
                            "model_id": arms[arm_name]["model_id"],
                            "thinking": arms[arm_name]["thinking"],
                            "sent_temperature": bool(arms[arm_name]["send_temperature"]),
                            "usage_raw": {},
                            "rid_file_sha1": rid_hash,
                        }
                    record_result(record, arm_name)
                    if aborted_reason is None:
                        aborted_reason = guard_failure_reason()
                        if aborted_reason:
                            print(
                                f"自动中止：{aborted_reason}",
                                file=sys.stderr,
                                flush=True,
                            )
                            errors = global_stats["errors"]
                            if not errors:
                                errors = [
                                    err
                                    for arm_stats in stats.values()
                                    for err in arm_stats["errors"]
                                ][:3]
                            for index, error in enumerate(errors[:3], 1):
                                print(
                                    f"错误原文 {index}: {error}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            # Calls already running may finish, but queued work
                            # is cancelled and no new calls are submitted.
                            for queued, queued_task in list(futures.items()):
                                if queued.cancel():
                                    futures.pop(queued)
                                    in_flight[queued_task[0]] -= 1
                if global_stats["completed"] and global_stats["completed"] % 200 == 0:
                    elapsed = time.time() - t_start
                    rate = global_stats["completed"] / max(elapsed, 1e-9)
                    print(
                        f"  {global_stats['completed']}/{len(todo)}  {rate:.1f}/s  "
                        f"失败{global_stats['failures']}  "
                        f"剩余~{(len(todo)-global_stats['completed'])/max(rate,1e-9)/3600:.2f}h",
                        flush=True,
                    )
                submit_until_full(pool)
    except Exception as exc:
        crashed_error = f"{type(exc).__name__}: {exc}"
        aborted_reason = "runner crashed: " + crashed_error
        print(aborted_reason, file=sys.stderr, flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    ended = now_iso()
    for name, arm in arms.items():
        arm_stats = stats[name]
        completed_count = arm_stats["completed"]
        observed_cost = _price(
            arm_stats["prompt_tokens"], arm_stats["completion_tokens"], arm
        )
        billing_calls = arm_stats["billing_usage_calls"]
        cost_complete = completed_count == 0 or billing_calls == completed_count
        actual_cost = observed_cost if cost_complete else None
        smoke_failed = args.smoke is not None and arm_stats["failures"] > 0
        run = dict(base_meta[name])
        run.update(
            {
                "status": (
                    "crashed" if crashed_error
                    else (
                        "aborted" if aborted_reason
                        else ("failed" if smoke_failed else "complete")
                    )
                ),
                "ended_at": ended,
                "calls_completed": completed_count,
                "failures": arm_stats["failures"],
                "average_tokens": {
                    field: (
                        arm_stats[field]
                        / arm_stats["usage_reported_calls"][field]
                        if arm_stats["usage_reported_calls"][field]
                        else None
                    )
                    for field in (
                        "prompt_tokens",
                        "completion_tokens",
                        "visible_output_tokens",
                        "thought_tokens",
                    )
                },
                "token_totals": {
                    field: (
                        arm_stats[field]
                        if arm_stats["usage_reported_calls"][field]
                        else None
                    )
                    for field in (
                        "prompt_tokens",
                        "completion_tokens",
                        "visible_output_tokens",
                        "thought_tokens",
                    )
                },
                "usage_reported_calls": dict(arm_stats["usage_reported_calls"]),
                "billing_usage_calls": billing_calls,
                "actual_cost_usd": actual_cost,
                "observed_cost_usd": observed_cost,
                "cost_complete": cost_complete,
                "abort_reason": aborted_reason,
                "errors": list(arm_stats["errors"]),
            }
        )
        if args.smoke is not None and billing_calls:
            mean_input = arm_stats["prompt_tokens"] / billing_calls
            mean_output = arm_stats["completion_tokens"] / billing_calls
            projected = _price(
                int(mean_input * complete_counts[name]),
                int(mean_output * complete_counts[name]),
                arm,
            )
            run["projected_full_run_cost_usd"] = projected
            run["projection_usage_coverage"] = billing_calls / completed_count
            print(
                f"[{name}] smoke 实测均值 输入{mean_input:.0f} / 输出{mean_output:.0f} token；"
                f"整轮 {complete_counts[name]} 次预计 ${projected:,.2f}"
            )
        elif args.smoke is not None:
            run["projected_full_run_cost_usd"] = None
            run["projection_usage_coverage"] = 0.0
            print(f"[{name}] smoke 未获得可计费 token，无法推算整轮费用")
        _update_run_meta(paths[name].parent / "run_meta.json", arm, run_id, run)

    elapsed_hours = (time.time() - t_start) / 3600
    print(
        f"完成 {global_stats['completed']}  失败 {global_stats['failures']}  "
        f"用时 {elapsed_hours:.2f}h"
    )
    smoke_failed = args.smoke is not None and global_stats["failures"] > 0
    return 2 if aborted_reason or smoke_failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    parser.add_argument("--provider-config", type=Path, default=DEFAULT_PROVIDER_CONFIG)
    parser.add_argument("--patient-dir", type=Path, default=Path("patients"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--arm")
    group.add_argument("--arms", help="comma-separated arm names, or 'all'")
    parser.add_argument("--rid-file", type=Path)
    parser.add_argument("--out", type=Path)
    # Legacy flags remain accepted only when --arm/--arms is omitted.
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--thinking")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--conds", default="main")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--month", type=int, default=12)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--sentinel-n", type=int, default=8)
    parser.add_argument("--sentinel-every", type=int, default=1500)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--fingerprints", type=Path, default=DEFAULT_FINGERPRINTS)
    parser.add_argument("--smoke", type=int, metavar="N")
    parser.add_argument("--plan", action="store_true")
    return parser


def main(definitions: Any, argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli_args(args)
    cards = _read_cards(args.cards)
    cards_by_rid = {int(card["RID"]): card for card in cards}
    validate_patient_prefix_files(args.patient_dir, cards_by_rid)
    fingerprint_baseline = verify_prompt_fingerprints(
        definitions, cards, args.cards, args.fingerprints
    )
    arms = select_arms(args)
    if args.out and len(arms) != 1:
        raise ValueError("--out can override only a single --arm")
    rids, rid_hash, rid_source, rnd = select_patients(cards_by_rid, args)
    validate_cards_for_month(cards_by_rid, rids, args.month)
    conds = resolve_conditions(definitions, args.conds)
    schedule, complete_schedule = build_task_schedule(
        list(arms),
        rids,
        conds,
        args.repeats,
        rnd,
        sentinel_n=args.sentinel_n,
        sentinel_every=args.sentinel_every,
        core_conditions=definitions.SETS["core"],
        smoke=args.smoke,
    )
    if args.workers is not None:
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        workers = args.workers
    else:
        workers = sum(int(arm["default_workers"]) for arm in arms.values())
    paths = {
        name: (args.out if args.out is not None else output_path_for(arm, args, len(rids)))
        for name, arm in arms.items()
    }
    normalized_paths = [
        os.path.normcase(str(path.resolve(strict=False))) for path in paths.values()
    ]
    if len(normalized_paths) != len(set(normalized_paths)):
        raise ValueError("two arm names resolve to the same output path after sanitizing")
    print_plan(
        schedule,
        complete_schedule,
        arms,
        len(rids),
        len(conds),
        args.repeats,
        workers,
        rid_hash,
        args.smoke,
    )
    if args.plan:
        return 0
    return execute(
        definitions,
        args,
        cards_by_rid,
        arms,
        schedule,
        complete_schedule,
        paths,
        rid_hash,
        rid_source,
        fingerprint_baseline,
        workers,
    )
