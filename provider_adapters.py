"""Provider-specific HTTP adapters with one small, testable interface.

The transport uses the Python standard library and, when installed, certifi's
CA bundle.  Callers pass the API key separately from the serializable provider
configuration and may inject ``opener`` and ``sleep_fn`` in tests, so importing
or exercising this module never requires network access.

Every call returns the same five-tuple::

    (parsed_object, normalized_usage, latency_ms, attempts, error)

``completion_tokens`` is the billable generated-token count (including known
thinking/reasoning tokens).  ``visible_output_tokens`` and ``thought_tokens``
split that value whenever the provider reports enough information.  The
provider's unmodified usage object is retained under ``usage_raw``.
"""

from __future__ import annotations

import copy
import functools
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any


AdapterResult = tuple[dict[str, Any] | None, dict[str, Any], int, int, str | None]
Validator = Callable[[dict[str, Any]], Any]

_IMMEDIATE_HTTP_CODES = {400, 401, 403, 404}
_RETRYABLE_HTTP_CODES = {408, 409, 425, 429}
_SENSITIVE_QUERY_NAMES = {
    "access_token",
    "api-key",
    "api_key",
    "apikey",
    "key",
    "token",
}
_RESERVED_EXTRA_BODY_KEYS = {
    # Identity and prompt-bearing fields.
    "model",
    "messages",
    "system",
    "system_instruction",
    "systemInstruction",
    "input",
    "contents",
    # Adapter-controlled generation and response invariants.
    "max_tokens",
    "max_output_tokens",
    "temperature",
    "response_format",
    "store",
    "stream",
}


class AdapterConfigurationError(ValueError):
    """A local configuration error that retrying cannot repair."""


def extract_first_json_object(text: str) -> dict[str, Any]:
    """Return the first complete JSON object embedded in *text*.

    ``json.JSONDecoder.raw_decode`` correctly handles braces and escaped quotes
    inside strings.  Starting at each opening brace also tolerates Markdown
    fences and prose before or after the object.
    """

    if not isinstance(text, str):
        raise TypeError("model output must be text")
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("response did not contain a complete JSON object")


def validate_risk_24m(value: dict[str, Any]) -> dict[str, Any]:
    """Minimal default validation shared by all providers."""

    risk = value.get("risk_24m")
    if not isinstance(risk, int) or isinstance(risk, bool) or not 0 <= risk <= 100:
        raise ValueError("risk_24m must be an integer from 0 to 100")
    return value


def _as_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float) and value.is_integer():
        return max(int(value), 0)
    return None


def _path_int(value: Mapping[str, Any], *path: str) -> int | None:
    current: Any = value
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return _as_nonnegative_int(current)


def _first_int(value: Mapping[str, Any], paths: list[tuple[str, ...]]) -> int | None:
    for path in paths:
        found = _path_int(value, *path)
        if found is not None:
            return found
    return None


def _usage_result(
    raw: Any,
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    visible_output_tokens: int | None,
    thought_tokens: int | None,
    usage_reported: bool | None = None,
) -> dict[str, Any]:
    reported = isinstance(raw, Mapping) if usage_reported is None else usage_reported
    raw_usage = copy.deepcopy(raw) if reported else {}
    if not reported:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "visible_output_tokens": None,
            "thought_tokens": None,
            "usage_reported": False,
            "request_attempts": 0,
            "usage_response_count": 0,
            "usage_complete": True,
            "usage_raw": {},
        }

    prompt = prompt_tokens
    thought = thought_tokens
    completion = completion_tokens
    visible = visible_output_tokens

    if completion is None and visible is not None:
        completion = visible + thought if thought is not None else visible
    if visible is None and completion is not None and thought is not None:
        visible = max(completion - thought, 0)

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "visible_output_tokens": visible,
        "thought_tokens": thought,
        "usage_reported": True,
        "usage_raw": raw_usage,
    }


def normalize_openai_usage(
    raw: Any, *, usage_reported: bool | None = None
) -> dict[str, Any]:
    usage = dict(raw) if isinstance(raw, Mapping) else {}
    prompt = _first_int(
        usage,
        [("prompt_tokens",), ("input_tokens",), ("promptTokenCount",)],
    )
    completion = _first_int(
        usage,
        [("completion_tokens",), ("output_tokens",), ("candidatesTokenCount",)],
    )
    thought = _first_int(
        usage,
        [
            ("completion_tokens_details", "reasoning_tokens"),
            ("output_tokens_details", "reasoning_tokens"),
            ("reasoning_tokens",),
            ("thinking_tokens",),
            ("thought_tokens",),
        ],
    )
    visible = _first_int(
        usage,
        [("visible_output_tokens",), ("visible_completion_tokens",)],
    )
    return _usage_result(
        raw,
        prompt_tokens=prompt,
        completion_tokens=completion,
        visible_output_tokens=visible,
        thought_tokens=thought,
        usage_reported=usage_reported,
    )


def normalize_anthropic_usage(
    raw: Any, *, usage_reported: bool | None = None
) -> dict[str, Any]:
    usage = dict(raw) if isinstance(raw, Mapping) else {}
    prompt = _first_int(usage, [("input_tokens",), ("prompt_tokens",)])
    completion = _first_int(usage, [("output_tokens",), ("completion_tokens",)])
    thought = _first_int(
        usage,
        [
            ("output_tokens_details", "thinking_tokens"),
            ("output_tokens_details", "reasoning_tokens"),
            ("thinking_tokens",),
            ("thought_tokens",),
        ],
    )
    visible = _first_int(
        usage,
        [("visible_output_tokens",), ("visible_completion_tokens",)],
    )
    return _usage_result(
        raw,
        prompt_tokens=prompt,
        completion_tokens=completion,
        visible_output_tokens=visible,
        thought_tokens=thought,
        usage_reported=usage_reported,
    )


def normalize_gemini_usage(
    raw: Any, *, usage_reported: bool | None = None
) -> dict[str, Any]:
    usage = dict(raw) if isinstance(raw, Mapping) else {}
    prompt = _first_int(
        usage,
        [
            ("total_input_tokens",),
            ("promptTokenCount",),
            ("prompt_token_count",),
            ("input_tokens",),
        ],
    )
    visible = _first_int(
        usage,
        [
            ("total_output_tokens",),
            ("candidatesTokenCount",),
            ("responseTokenCount",),
            ("output_tokens",),
        ],
    )
    thought = _first_int(
        usage,
        [
            ("total_thought_tokens",),
            ("thoughtsTokenCount",),
            ("thought_tokens",),
            ("thinking_tokens",),
        ],
    )
    # Gemini reports response and thought tokens separately; both are billed as
    # generated output.  If a future response supplies only a total, derive the
    # generated portion without double-counting prompt tokens.
    completion = None
    if visible is not None or thought is not None:
        completion = (visible or 0) + (thought or 0)
    else:
        total = _first_int(usage, [("total_tokens",), ("totalTokenCount",)])
        if total is not None:
            completion = max(total - (prompt or 0), 0)
    return _usage_result(
        raw,
        prompt_tokens=prompt,
        completion_tokens=completion,
        visible_output_tokens=visible,
        thought_tokens=thought,
        usage_reported=usage_reported,
    )


def normalize_usage(provider_type: str, raw: Any) -> dict[str, Any]:
    """Normalize a provider usage object without making a request."""

    if provider_type == "openai_compat":
        return normalize_openai_usage(raw)
    if provider_type == "anthropic":
        return normalize_anthropic_usage(raw)
    if provider_type == "gemini":
        return normalize_gemini_usage(raw)
    raise AdapterConfigurationError(f"unsupported provider type: {provider_type!r}")


def _provider_type(config: Mapping[str, Any]) -> str:
    """Resolve the protocol type while allowing a separate vendor label."""

    explicit = config.get("provider_type", config.get("type"))
    if isinstance(explicit, str) and explicit:
        return explicit
    return str(config.get("provider", ""))


def _thinking_enabled(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "disabled", "false", "none", "off"}
    if isinstance(value, Mapping):
        kind = str(value.get("type", "")).lower()
        enabled = value.get("enabled")
        return kind not in {"disabled", "none", "off"} and enabled is not False
    return bool(value)


def _contains_mapping_key(value: Any, key: str) -> bool:
    if isinstance(value, Mapping):
        return key in value or any(_contains_mapping_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_mapping_key(child, key) for child in value)
    return False


def _copy_extra_body(
    config: Mapping[str, Any], *, provider_type: str
) -> dict[str, Any]:
    extra = config.get("extra_body") or {}
    if not isinstance(extra, Mapping):
        raise AdapterConfigurationError("extra_body must be a mapping")
    copied = copy.deepcopy(dict(extra))
    conflicts = sorted(_RESERVED_EXTRA_BODY_KEYS.intersection(copied))
    # OpenAI-compatible vendors intentionally expose their thinking/reasoning
    # switches through extra_body.  Those fields are not prompt-bearing and
    # are therefore allowed to pass through unchanged.
    if provider_type == "openai_compat":
        conflicts = [name for name in conflicts if name not in {"thinking"}]
    if provider_type == "gemini" and copied.get("store") is False:
        conflicts = [name for name in conflicts if name != "store"]
    if provider_type in {"anthropic", "gemini"} and "thinking" in copied:
        conflicts = sorted(set(conflicts) | {"thinking"})
    if provider_type == "gemini" and (
        "generation_config" in copied or "generationConfig" in copied
    ):
        conflicts = sorted(
            set(conflicts)
            | ({"generation_config"} if "generation_config" in copied else set())
            | ({"generationConfig"} if "generationConfig" in copied else set())
        )
    if provider_type == "gemini" and _contains_mapping_key(copied, "temperature"):
        conflicts = sorted(set(conflicts) | {"temperature"})
    if conflicts:
        names = ", ".join(conflicts)
        raise AdapterConfigurationError(
            f"extra_body cannot override adapter-controlled field(s): {names}"
        )
    return copied


def _require_text(config: Mapping[str, Any], name: str) -> str:
    value = config.get(name)
    if not isinstance(value, str) or not value.strip():
        raise AdapterConfigurationError(f"provider config requires non-empty {name!r}")
    return value


def _format_endpoint(config: Mapping[str, Any]) -> str:
    endpoint = _require_text(config, "endpoint")
    model = _require_text(config, "model_id")
    quoted_model = urllib.parse.quote(model, safe="-._~")
    try:
        return endpoint.format(model=quoted_model, model_id=quoted_model)
    except (KeyError, ValueError) as exc:
        raise AdapterConfigurationError(f"invalid endpoint template: {exc}") from exc


def _openai_body(
    config: Mapping[str, Any], system_prompt: str, user_prompt: str
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": _require_text(config, "model_id"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    # OpenAI-compatible gateways do not all accept the same optional
    # generation controls.  Default to the historical request shape, while
    # allowing a deliberately minimal model+messages request when configured.
    if config.get("send_max_tokens", True):
        body["max_tokens"] = int(config.get("max_tokens", 500))
    if config.get("send_temperature", False):
        body["temperature"] = config.get("temperature", 0.0)
    if config.get("json_mode", False):
        body["response_format"] = {"type": "json_object"}
    body.update(_copy_extra_body(config, provider_type="openai_compat"))
    return body


def _anthropic_body(
    config: Mapping[str, Any], system_prompt: str, user_prompt: str
) -> dict[str, Any]:
    thinking = copy.deepcopy(config.get("thinking"))
    if isinstance(thinking, str):
        normalized_kind = thinking.strip().lower()
        if normalized_kind in {"", "none", "off", "false"}:
            thinking = None
        elif normalized_kind in {"adaptive", "disabled"}:
            thinking = {"type": normalized_kind}
    elif thinking is False:
        thinking = None
    body: dict[str, Any] = {
        "model": _require_text(config, "model_id"),
        "max_tokens": int(config.get("max_tokens", 500)),
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if thinking is not None:
        body["thinking"] = thinking
    if config.get("send_temperature", False) and not _thinking_enabled(thinking):
        body["temperature"] = config.get("temperature", 0.0)
    body.update(_copy_extra_body(config, provider_type="anthropic"))
    # Anthropic rejects temperature whenever thinking is enabled, including if
    # it was accidentally supplied through extra_body.
    if _thinking_enabled(body.get("thinking")):
        body.pop("temperature", None)
    return body


def _gemini_interactions_body(
    config: Mapping[str, Any], system_prompt: str, user_prompt: str
) -> dict[str, Any]:
    generation: dict[str, Any] = {
        "max_output_tokens": int(config.get("max_tokens", 500))
    }
    thinking = config.get("thinking")
    if isinstance(thinking, Mapping):
        generation.update(copy.deepcopy(dict(thinking)))
    elif thinking not in (None, False, ""):
        generation["thinking_level"] = thinking

    body: dict[str, Any] = {
        "model": _require_text(config, "model_id"),
        "system_instruction": system_prompt,
        "input": user_prompt,
        "generation_config": generation,
        "store": False,
    }
    if config.get("json_mode", False):
        body["response_format"] = {
            "type": "text",
            "mime_type": "application/json",
        }
    body.update(_copy_extra_body(config, provider_type="gemini"))
    body["store"] = False
    _remove_key_recursive(body, "temperature")
    return body


def _gemini_generate_content_body(
    config: Mapping[str, Any], system_prompt: str, user_prompt: str
) -> dict[str, Any]:
    generation: dict[str, Any] = {
        "maxOutputTokens": int(config.get("max_tokens", 500))
    }
    if config.get("json_mode", False):
        generation["responseMimeType"] = "application/json"
    thinking = config.get("thinking")
    if isinstance(thinking, Mapping):
        level = thinking.get("thinking_level", thinking.get("thinkingLevel"))
    else:
        level = thinking
    if level not in (None, False, ""):
        generation["thinkingConfig"] = {"thinkingLevel": level}

    body: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": generation,
        "store": False,
    }
    body.update(_copy_extra_body(config, provider_type="gemini"))
    body["store"] = False
    _remove_key_recursive(body, "temperature")
    return body


def _is_gemini_generate_content(config: Mapping[str, Any]) -> bool:
    style = str(config.get("api_style", "")).lower()
    return style in {"generate_content", "generatecontent"} or "generatecontent" in str(
        config.get("endpoint", "")
    ).lower()


def _gemini_body(
    config: Mapping[str, Any], system_prompt: str, user_prompt: str
) -> dict[str, Any]:
    if _is_gemini_generate_content(config):
        return _gemini_generate_content_body(config, system_prompt, user_prompt)
    return _gemini_interactions_body(config, system_prompt, user_prompt)


def _remove_key_recursive(value: Any, forbidden_key: str) -> None:
    """Remove a forbidden request parameter at any nesting depth in place."""

    if isinstance(value, dict):
        value.pop(forbidden_key, None)
        for child in value.values():
            _remove_key_recursive(child, forbidden_key)
    elif isinstance(value, list):
        for child in value:
            _remove_key_recursive(child, forbidden_key)


def config_to_request_params(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return serializable request parameters with prompts and secrets omitted."""

    provider_type = _provider_type(config)
    placeholder = "__PROMPT_OMITTED__"
    if provider_type == "openai_compat":
        body = _openai_body(config, placeholder, placeholder)
        body.pop("messages", None)
    elif provider_type == "anthropic":
        body = _anthropic_body(config, placeholder, placeholder)
        body.pop("system", None)
        body.pop("messages", None)
    elif provider_type == "gemini":
        body = _gemini_body(config, placeholder, placeholder)
        for key in ("system_instruction", "input", "systemInstruction", "contents"):
            body.pop(key, None)
    else:
        raise AdapterConfigurationError(f"unsupported provider type: {provider_type!r}")
    return body


def _redact_endpoint(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    redacted = [
        (name, "<redacted>" if name.lower() in _SENSITIVE_QUERY_NAMES else value)
        for name, value in query
    ]
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(redacted), parsed.fragment)
    )


def build_request_preview(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build a metadata-safe request preview for ``run_meta.json``."""

    provider_type = _provider_type(config)
    credential = f"<from {config.get('api_key_env', 'configured environment')}>"
    headers = {"Content-Type": "application/json"}
    if provider_type == "openai_compat":
        headers["Authorization"] = "Bearer " + credential
    elif provider_type == "anthropic":
        headers["x-api-key"] = credential
        headers["anthropic-version"] = str(
            config.get("anthropic_version", "2023-06-01")
        )
        if config.get("anthropic_beta"):
            headers["anthropic-beta"] = str(config["anthropic_beta"])
    elif provider_type == "gemini":
        headers["x-goog-api-key"] = credential
    return {
        "provider_type": provider_type,
        "method": "POST",
        "endpoint": _redact_endpoint(_format_endpoint(config)),
        "headers": headers,
        "request_params": config_to_request_params(config),
    }


def _json_request(endpoint: str, api_key: str, body: Mapping[str, Any], provider: str):
    if not isinstance(api_key, str) or not api_key:
        raise AdapterConfigurationError("api_key must be passed separately and cannot be empty")
    headers = {"Content-Type": "application/json"}
    if provider == "openai_compat":
        headers["Authorization"] = f"Bearer {api_key}"
    elif provider == "anthropic":
        headers["x-api-key"] = api_key
    elif provider == "gemini":
        headers["x-goog-api-key"] = api_key
    else:
        raise AdapterConfigurationError(f"unsupported provider type: {provider!r}")
    return urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )


@functools.lru_cache(maxsize=1)
def _certifi_ssl_context() -> ssl.SSLContext | None:
    """Build a verified TLS context without consulting the Windows cert store.

    Some Windows/Conda combinations fail while decoding a malformed certificate
    from the OS store before an HTTPS request is sent.  An explicit certifi CA
    file avoids that code path.  Certifi remains optional so environments that
    rely on the platform trust store retain the previous behavior.
    """

    try:
        import certifi
    except ImportError:
        return None
    try:
        return ssl.create_default_context(cafile=certifi.where())
    except (OSError, ssl.SSLError, ValueError):
        return None


def _response_bytes(opener: Any, request: urllib.request.Request, timeout: float) -> bytes:
    if opener is None:
        context = _certifi_ssl_context()
        if context is None:
            response = urllib.request.urlopen(request, timeout=timeout)
        else:
            response = urllib.request.urlopen(request, timeout=timeout, context=context)
    else:
        open_fn = getattr(opener, "open", opener)
        response = open_fn(request, timeout=timeout)
    try:
        payload = response.read()
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    if isinstance(payload, str):
        return payload.encode("utf-8")
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("HTTP response .read() must return bytes or str")
    return bytes(payload)


def _http_error_text(error: urllib.error.HTTPError) -> str:
    detail = ""
    try:
        raw = error.read()
        if isinstance(raw, bytes):
            detail = raw.decode("utf-8", errors="replace").strip()
        elif isinstance(raw, str):
            detail = raw.strip()
    except Exception:
        detail = ""
    base = f"HTTP {error.code}: {error.reason or type(error).__name__}"
    return f"{base}: {detail[:2000]}" if detail else base


def _retry_http_status(code: int) -> bool:
    if code in _IMMEDIATE_HTTP_CODES:
        return False
    if code in _RETRYABLE_HTTP_CODES or 500 <= code <= 599:
        return True
    # Other 4xx statuses are generally permanent request/auth failures.
    if 400 <= code <= 499:
        return False
    return True


def _validated(value: dict[str, Any], validator: Validator | None) -> dict[str, Any]:
    check = validator or validate_risk_24m
    result = check(value)
    if result is False:
        raise ValueError("response validator rejected the JSON object")
    return value


def _reported_usage(
    response: Mapping[str, Any], keys: tuple[str, ...]
) -> tuple[Any, bool]:
    for key in keys:
        if key in response:
            return response[key], True
    return None, False


def _aggregate_usage(
    normalized_attempts: list[dict[str, Any]], raw_attempts: list[Any]
) -> dict[str, Any]:
    if not normalized_attempts:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "visible_output_tokens": None,
            "thought_tokens": None,
            "usage_reported": False,
            "usage_raw": {},
        }

    result: dict[str, Any] = {}
    for field in (
        "prompt_tokens",
        "completion_tokens",
        "visible_output_tokens",
        "thought_tokens",
    ):
        values = [attempt.get(field) for attempt in normalized_attempts]
        if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            result[field] = sum(values)
        else:
            result[field] = None
    result["usage_reported"] = True
    result["usage_raw"] = (
        copy.deepcopy(raw_attempts[0])
        if len(raw_attempts) == 1
        else {"attempts": copy.deepcopy(raw_attempts)}
    )
    return result


def _execute(
    request_factory: Callable[[], urllib.request.Request],
    response_parser: Callable[[dict[str, Any]], dict[str, Any]],
    usage_normalizer: Callable[..., dict[str, Any]],
    usage_keys: tuple[str, ...],
    validator: Validator | None,
    *,
    retries: int,
    opener: Any,
    sleep_fn: Callable[[float], Any],
    timeout: float,
) -> AdapterResult:
    attempts_allowed = max(int(retries), 1)
    started = time.perf_counter()
    normalized_attempts: list[dict[str, Any]] = []
    raw_attempts: list[Any] = []
    last_usage = _aggregate_usage(normalized_attempts, raw_attempts)
    last_error: str | None = None

    def usage_with_coverage(request_attempts: int) -> dict[str, Any]:
        result = copy.deepcopy(last_usage)
        result["request_attempts"] = request_attempts
        result["usage_response_count"] = len(raw_attempts)
        result["usage_complete"] = len(raw_attempts) == request_attempts
        return result

    for attempt in range(1, attempts_allowed + 1):
        try:
            request = request_factory()
            raw = _response_bytes(opener, request, timeout)
            response = json.loads(raw.decode("utf-8"))
            if not isinstance(response, dict):
                raise ValueError("provider response must be a JSON object")
            usage_raw, usage_was_reported = _reported_usage(response, usage_keys)
            if usage_was_reported:
                raw_attempts.append(copy.deepcopy(usage_raw))
                normalized_attempts.append(
                    usage_normalizer(usage_raw, usage_reported=True)
                )
                last_usage = _aggregate_usage(normalized_attempts, raw_attempts)
            parsed = response_parser(response)
            parsed = _validated(parsed, validator)
            latency_ms = int((time.perf_counter() - started) * 1000)
            return parsed, usage_with_coverage(attempt), latency_ms, attempt, None
        except AdapterConfigurationError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            latency_ms = int((time.perf_counter() - started) * 1000)
            actual_attempts = attempt - 1
            return (
                None,
                usage_with_coverage(actual_attempts),
                latency_ms,
                actual_attempts,
                last_error,
            )
        except urllib.error.HTTPError as exc:
            last_error = _http_error_text(exc)
            if not _retry_http_status(exc.code) or attempt == attempts_allowed:
                latency_ms = int((time.perf_counter() - started) * 1000)
                return (
                    None, usage_with_coverage(attempt), latency_ms, attempt, last_error
                )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:2000]}"
            if attempt == attempts_allowed:
                latency_ms = int((time.perf_counter() - started) * 1000)
                return (
                    None, usage_with_coverage(attempt), latency_ms, attempt, last_error
                )

        sleep_fn(min(2 ** (attempt - 1), 30))

    # The loop always returns on its final iteration; this keeps type checkers
    # and unusual integer subclasses from obscuring the function contract.
    latency_ms = int((time.perf_counter() - started) * 1000)
    return (
        None,
        usage_with_coverage(attempts_allowed),
        latency_ms,
        attempts_allowed,
        last_error,
    )


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    raise ValueError("provider response did not contain assistant text")


def _parse_openai_response(response: dict[str, Any]) -> dict[str, Any]:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("OpenAI-compatible response is missing choices[0].message.content") from exc
    return extract_first_json_object(_message_text(content))


def _parse_anthropic_response(response: dict[str, Any]) -> dict[str, Any]:
    content = response.get("content")
    if not isinstance(content, list):
        raise ValueError("Anthropic response is missing content blocks")
    text = "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, Mapping)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )
    if not text:
        raise ValueError("Anthropic response did not contain a text block")
    return extract_first_json_object(text)


def _parse_gemini_response(response: dict[str, Any]) -> dict[str, Any]:
    text_parts: list[str] = []

    # Interactions API response.
    if isinstance(response.get("output_text"), str):
        text_parts.append(response["output_text"])
    # Some Interactions API revisions/gateways expose ``outputs`` rather than
    # the resource-shaped ``steps`` array.  Walk the output envelopes, but do
    # not collect thought/reasoning blocks as visible answer text.
    outputs = response.get("outputs")
    if not text_parts and isinstance(outputs, list):
        def append_output_text(value: Any) -> None:
            if isinstance(value, str):
                text_parts.append(value)
                return
            if isinstance(value, list):
                for item in value:
                    append_output_text(item)
                return
            if not isinstance(value, Mapping):
                return
            kind = str(value.get("type", "")).lower()
            if "thought" in kind or "reason" in kind:
                return
            text = value.get("text")
            if isinstance(text, str):
                text_parts.append(text)
            for key in ("content", "parts", "output"):
                if key in value:
                    append_output_text(value[key])

        append_output_text(outputs)
    steps = response.get("steps")
    if not text_parts and isinstance(steps, list):
        for step in steps:
            if not isinstance(step, Mapping) or step.get("type") != "model_output":
                continue
            content = step.get("content")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, Mapping)
                        and block.get("type") == "text"
                        and isinstance(block.get("text"), str)
                    ):
                        text_parts.append(block["text"])

    # Legacy generateContent response.
    if not text_parts:
        candidates = response.get("candidates")
        if isinstance(candidates, list) and candidates:
            candidate = candidates[0]
            content = candidate.get("content", {}) if isinstance(candidate, Mapping) else {}
            parts = content.get("parts", []) if isinstance(content, Mapping) else []
            for part in parts if isinstance(parts, list) else []:
                if (
                    isinstance(part, Mapping)
                    and not part.get("thought", False)
                    and isinstance(part.get("text"), str)
                ):
                    text_parts.append(part["text"])

    if not text_parts:
        raise ValueError("Gemini response did not contain visible model text")
    return extract_first_json_object("".join(text_parts))


def call_openai_compat(
    config: Mapping[str, Any],
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    validator: Validator | None = None,
    *,
    retries: int = 5,
    opener: Any = None,
    sleep_fn: Callable[[float], Any] = time.sleep,
    timeout: float = 180,
) -> AdapterResult:
    """Call a DeepSeek/Moonshot/DashScope OpenAI-compatible endpoint."""

    return _execute(
        lambda: _json_request(
            _format_endpoint(config),
            api_key,
            _openai_body(config, system_prompt, user_prompt),
            "openai_compat",
        ),
        _parse_openai_response,
        normalize_openai_usage,
        ("usage",),
        validator,
        retries=retries,
        opener=opener,
        sleep_fn=sleep_fn,
        timeout=timeout,
    )


def call_anthropic(
    config: Mapping[str, Any],
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    validator: Validator | None = None,
    *,
    retries: int = 5,
    opener: Any = None,
    sleep_fn: Callable[[float], Any] = time.sleep,
    timeout: float = 180,
) -> AdapterResult:
    """Call Anthropic's Messages API."""

    def request_factory():
        request = _json_request(
            _format_endpoint(config),
            api_key,
            _anthropic_body(config, system_prompt, user_prompt),
            "anthropic",
        )
        request.add_header(
            "anthropic-version", str(config.get("anthropic_version", "2023-06-01"))
        )
        beta = config.get("anthropic_beta")
        if beta:
            request.add_header("anthropic-beta", str(beta))
        return request

    return _execute(
        request_factory,
        _parse_anthropic_response,
        normalize_anthropic_usage,
        ("usage",),
        validator,
        retries=retries,
        opener=opener,
        sleep_fn=sleep_fn,
        timeout=timeout,
    )


def call_gemini(
    config: Mapping[str, Any],
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    validator: Validator | None = None,
    *,
    retries: int = 5,
    opener: Any = None,
    sleep_fn: Callable[[float], Any] = time.sleep,
    timeout: float = 180,
) -> AdapterResult:
    """Call Gemini Interactions (or a configured generateContent endpoint).

    ``store`` is forced to ``False`` and temperature is removed even if either
    is supplied in ``extra_body``.  These experiment invariants therefore
    cannot be weakened accidentally by provider configuration.
    """

    return _execute(
        lambda: _json_request(
            _format_endpoint(config),
            api_key,
            _gemini_body(config, system_prompt, user_prompt),
            "gemini",
        ),
        _parse_gemini_response,
        normalize_gemini_usage,
        ("usage", "usageMetadata"),
        validator,
        retries=retries,
        opener=opener,
        sleep_fn=sleep_fn,
        timeout=timeout,
    )


def call_provider(
    config: Mapping[str, Any],
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    validator: Validator | None = None,
    **kwargs: Any,
) -> AdapterResult:
    """Dispatch to the adapter selected by ``provider``/``provider_type``."""

    provider_type = _provider_type(config)
    adapters = {
        "openai_compat": call_openai_compat,
        "anthropic": call_anthropic,
        "gemini": call_gemini,
    }
    adapter = adapters.get(provider_type)
    if adapter is None:
        usage = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "visible_output_tokens": None,
            "thought_tokens": None,
            "usage_reported": False,
            "request_attempts": 0,
            "usage_response_count": 0,
            "usage_complete": True,
            "usage_raw": {},
        }
        return (
            None,
            usage,
            0,
            0,
            f"AdapterConfigurationError: unsupported provider type: {provider_type!r}",
        )
    return adapter(
        config,
        api_key,
        system_prompt,
        user_prompt,
        validator,
        **kwargs,
    )


__all__ = [
    "AdapterConfigurationError",
    "AdapterResult",
    "build_request_preview",
    "call_anthropic",
    "call_gemini",
    "call_openai_compat",
    "call_provider",
    "config_to_request_params",
    "extract_first_json_object",
    "normalize_anthropic_usage",
    "normalize_gemini_usage",
    "normalize_openai_usage",
    "normalize_usage",
    "validate_risk_24m",
]
