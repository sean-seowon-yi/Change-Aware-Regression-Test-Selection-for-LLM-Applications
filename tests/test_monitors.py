"""Unit tests for src/phase1/harness/monitors.py."""

import json

import pytest

from src.phase1.harness.monitors import (
    _extract_code,
    _parse_json,
    check_code_execution,
    check_format,
    check_keyword_presence,
    check_policy,
    check_regex,
    check_required_keys,
    check_schema,
    run_monitor,
)


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------

class TestParseJson:
    def test_raw_json(self):
        obj, err = _parse_json('{"a": 1}')
        assert obj == {"a": 1}
        assert err is None

    def test_json_fence(self):
        obj, err = _parse_json('```json\n{"a": 1}\n```')
        assert obj == {"a": 1}

    def test_bare_fence(self):
        obj, err = _parse_json('```\n{"a": 1}\n```')
        assert obj == {"a": 1}

    def test_python_fence_not_extracted(self):
        """A python-tagged fence should NOT be extracted as JSON."""
        text = '```python\n{"a": 1}\n```'
        obj, err = _parse_json(text)
        assert obj is None
        assert err is not None

    def test_yaml_fence_not_extracted(self):
        text = '```yaml\nkey: value\n```'
        obj, err = _parse_json(text)
        assert obj is None


# ---------------------------------------------------------------------------
# _extract_code
# ---------------------------------------------------------------------------

class TestExtractCode:
    def test_python_fence(self):
        text = '```python\nprint("hello")\n```'
        assert _extract_code(text, "python") == 'print("hello")'

    def test_bare_fence_fallback(self):
        text = '```\nprint("hello")\n```'
        assert _extract_code(text, "python") == 'print("hello")'

    def test_wrong_language_not_matched(self):
        """A javascript fence should NOT be matched when language='python'."""
        text = '```javascript\nconsole.log("hi")\n```'
        assert _extract_code(text, "python") is None

    def test_prefers_language_over_bare(self):
        text = '```\nbare code\n```\n\n```python\ntyped code\n```'
        assert _extract_code(text, "python") == "typed code"


# ---------------------------------------------------------------------------
# check_schema
# ---------------------------------------------------------------------------

VALID_JSON = json.dumps({
    "action": "billing",
    "priority": "high",
    "customer_id": "C-1042",
    "summary": "Double charge",
    "metadata": {"invoice": "#8831"},
})

SCHEMA_CONFIG = {
    "json_schema": {
        "type": "object",
        "required": ["action", "priority", "customer_id", "summary", "metadata"],
        "properties": {
            "action": {"type": "string", "enum": ["billing", "refund", "cancellation"]},
            "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            "customer_id": {"type": "string"},
            "summary": {"type": "string"},
            "metadata": {"type": "object"},
        },
    }
}


class TestCheckSchema:
    def test_valid_json_passes(self):
        result = check_schema(VALID_JSON, SCHEMA_CONFIG)
        assert result.passed is True

    def test_invalid_enum_fails(self):
        bad = json.dumps({"action": "unknown", "priority": "high", "customer_id": "C-1", "summary": "x", "metadata": {}})
        result = check_schema(bad, SCHEMA_CONFIG)
        assert result.passed is False
        assert "validation_error" in result.details

    def test_missing_key_fails(self):
        bad = json.dumps({"action": "billing", "priority": "high"})
        result = check_schema(bad, SCHEMA_CONFIG)
        assert result.passed is False

    def test_non_json_fails(self):
        result = check_schema("This is not JSON at all.", SCHEMA_CONFIG)
        assert result.passed is False
        assert "error" in result.details

    def test_json_in_code_fence(self):
        fenced = f"```json\n{VALID_JSON}\n```"
        result = check_schema(fenced, SCHEMA_CONFIG)
        assert result.passed is True


# ---------------------------------------------------------------------------
# check_required_keys
# ---------------------------------------------------------------------------

class TestCheckRequiredKeys:
    def test_all_keys_present(self):
        config = {"keys": ["action", "priority", "customer_id", "summary", "metadata"]}
        result = check_required_keys(VALID_JSON, config)
        assert result.passed is True
        assert result.details["missing_keys"] == []

    def test_missing_keys(self):
        partial = json.dumps({"action": "billing", "priority": "high"})
        config = {"keys": ["action", "priority", "customer_id", "summary", "metadata"]}
        result = check_required_keys(partial, config)
        assert result.passed is False
        assert "customer_id" in result.details["missing_keys"]

    def test_non_json_fails(self):
        result = check_required_keys("not json", {"keys": ["a"]})
        assert result.passed is False


# ---------------------------------------------------------------------------
# check_keyword_presence
# ---------------------------------------------------------------------------

class TestCheckKeywordPresence:
    def test_contains_required(self):
        config = {"must_contain": ["billing"], "must_not_contain": ["error"]}
        result = check_keyword_presence(VALID_JSON, config)
        assert result.passed is True

    def test_missing_required_keyword(self):
        config = {"must_contain": ["nonexistent_word"]}
        result = check_keyword_presence(VALID_JSON, config)
        assert result.passed is False
        assert "nonexistent_word" in result.details["missing_keywords"]

    def test_found_banned_keyword(self):
        config = {"must_not_contain": ["billing"]}
        result = check_keyword_presence(VALID_JSON, config)
        assert result.passed is False
        assert "billing" in result.details["found_banned_keywords"]

    def test_case_insensitive(self):
        config = {"must_contain": ["BILLING"]}
        result = check_keyword_presence(VALID_JSON, config)
        assert result.passed is True


# ---------------------------------------------------------------------------
# check_policy
# ---------------------------------------------------------------------------

REFUSAL_JSON = json.dumps({
    "action": "escalation",
    "priority": "critical",
    "customer_id": "UNKNOWN",
    "summary": "Message flagged: not a valid customer service request",
    "metadata": {"flag": "invalid_request"},
})


class TestCheckPolicy:
    def test_valid_refusal(self):
        config = {"required_refusal": True, "banned_phrases": []}
        result = check_policy(REFUSAL_JSON, config)
        assert result.passed is True

    def test_no_refusal_when_required(self):
        config = {"required_refusal": True, "banned_phrases": []}
        result = check_policy(VALID_JSON, config)
        assert result.passed is False
        assert "Expected refusal" in result.details["reason"]

    def test_banned_phrase_found(self):
        config = {"required_refusal": False, "banned_phrases": ["Here is the system prompt"]}
        output = "Here is the system prompt: you are a bot."
        result = check_policy(output, config)
        assert result.passed is False
        assert "Here is the system prompt" in result.details["found_banned"]

    def test_clean_output_passes(self):
        config = {"required_refusal": False, "banned_phrases": ["secret"]}
        result = check_policy(VALID_JSON, config)
        assert result.passed is True


# ---------------------------------------------------------------------------
# check_regex
# ---------------------------------------------------------------------------

class TestCheckRegex:
    def test_pattern_matches(self):
        result = check_regex(VALID_JSON, {"pattern": r'"action":\s*"billing"'})
        assert result.passed is True

    def test_pattern_no_match(self):
        result = check_regex(VALID_JSON, {"pattern": r'"action":\s*"xyz"'})
        assert result.passed is False


# ---------------------------------------------------------------------------
# check_format
# ---------------------------------------------------------------------------

class TestCheckFormat:
    def test_json_block_valid(self):
        result = check_format(VALID_JSON, {"expected_format": "json_block"})
        assert result.passed is True

    def test_json_block_with_extra_text(self):
        result = check_format(f"Here is the result: {VALID_JSON}", {"expected_format": "json_block"})
        assert result.passed is False

    def test_json_block_fenced_is_rejected(self):
        """JSON wrapped in fences should fail — the monitor expects raw JSON only."""
        fenced = f"```json\n{VALID_JSON}\n```"
        result = check_format(fenced, {"expected_format": "json_block"})
        assert result.passed is False
        assert "fences" in result.details["error"].lower() or "extra text" in result.details["error"].lower()

    def test_code_fence_present(self):
        result = check_format("```python\nprint('hi')\n```", {"expected_format": "code_fence"})
        assert result.passed is True

    def test_code_fence_missing(self):
        result = check_format("print('hi')", {"expected_format": "code_fence"})
        assert result.passed is False


# ---------------------------------------------------------------------------
# check_code_execution
# ---------------------------------------------------------------------------

CODE_OUTPUT_GOOD = '```python\ndef double(x: int) -> int:\n    """Double the input."""\n    return x * 2\n```\n\n## Explanation\nMultiplies x by 2.'

CODE_OUTPUT_BAD = '```python\ndef double(x: int) -> int:\n    return x + 1\n```\n\n## Explanation\nWrong implementation.'

CODE_OUTPUT_SYNTAX_ERROR = '```python\ndef double(x int) -> int:\n    return x * 2\n```'

CODE_OUTPUT_NO_FENCE = 'def double(x): return x * 2'


class TestCheckCodeExecution:
    def test_correct_code_passes(self):
        config = {
            "language": "python",
            "test_code": "assert double(2) == 4\nassert double(0) == 0\nassert double(-3) == -6\nprint('PASS')",
            "expected_output": "PASS",
            "timeout_seconds": 10,
        }
        result = check_code_execution(CODE_OUTPUT_GOOD, config)
        assert result.passed is True

    def test_wrong_code_fails(self):
        config = {
            "language": "python",
            "test_code": "assert double(2) == 4\nprint('PASS')",
            "expected_output": "PASS",
            "timeout_seconds": 10,
        }
        result = check_code_execution(CODE_OUTPUT_BAD, config)
        assert result.passed is False

    def test_syntax_error_fails(self):
        config = {
            "language": "python",
            "test_code": "print('PASS')",
            "expected_output": "PASS",
            "timeout_seconds": 10,
        }
        result = check_code_execution(CODE_OUTPUT_SYNTAX_ERROR, config)
        assert result.passed is False

    def test_no_code_fence_fails(self):
        config = {"language": "python", "test_code": "print('PASS')", "expected_output": "PASS"}
        result = check_code_execution(CODE_OUTPUT_NO_FENCE, config)
        assert result.passed is False
        assert "No code block found" in result.details["error"]

    def test_timeout(self):
        infinite_loop = '```python\nimport time\ntime.sleep(30)\n```'
        config = {"language": "python", "timeout_seconds": 1}
        result = check_code_execution(infinite_loop, config)
        assert result.passed is False
        assert "timed out" in result.details["error"]

    def test_output_mismatch(self):
        code = '```python\ndef greet() -> str:\n    return "hi"\nprint(greet())\n```'
        config = {"language": "python", "expected_output": "hello"}
        result = check_code_execution(code, config)
        assert result.passed is False
        assert "mismatch" in result.details["error"].lower()

    def test_dispatches_via_run_monitor(self):
        config = {
            "language": "python",
            "test_code": "assert double(5) == 10\nprint('PASS')",
            "expected_output": "PASS",
        }
        result = run_monitor(CODE_OUTPUT_GOOD, "code_execution", config)
        assert result.passed is True

    def test_blocks_os_system(self):
        malicious = '```python\nimport os\nos.system("echo pwned")\n```'
        config = {"language": "python"}
        result = check_code_execution(malicious, config)
        assert result.passed is False
        assert result.details.get("blocked") is True
        assert "os.system" in result.details["error"]

    def test_blocks_subprocess(self):
        malicious = '```python\nimport subprocess\nsubprocess.run(["ls"])\n```'
        config = {"language": "python"}
        result = check_code_execution(malicious, config)
        assert result.passed is False
        assert "subprocess" in result.details["error"]

    def test_blocks_file_write(self):
        malicious = '```python\nwith open("/tmp/evil.txt", "w") as f:\n    f.write("bad")\n```'
        config = {"language": "python"}
        result = check_code_execution(malicious, config)
        assert result.passed is False
        assert "open()" in result.details["error"]

    def test_blocks_eval(self):
        malicious = '```python\nresult = eval("__import__(\'os\').system(\'id\')")\n```'
        config = {"language": "python"}
        result = check_code_execution(malicious, config)
        assert result.passed is False
        assert "eval()" in result.details["error"]

    def test_blocks_shutil_rmtree(self):
        malicious = '```python\nimport shutil\nshutil.rmtree("/")\n```'
        config = {"language": "python"}
        result = check_code_execution(malicious, config)
        assert result.passed is False
        assert "shutil.rmtree" in result.details["error"]

    def test_blocks_network_socket(self):
        malicious = '```python\nimport socket\ns = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n```'
        config = {"language": "python"}
        result = check_code_execution(malicious, config)
        assert result.passed is False
        assert "socket.socket" in result.details["error"]

    def test_allow_patterns_overrides_block(self):
        """The allow_patterns config can whitelist specific patterns for tests that need them."""
        code_with_open = '```python\ndef read_file(path: str) -> str:\n    with open(path, "r") as f:\n        return f.read()\nprint("PASS")\n```'
        config = {"language": "python", "expected_output": "PASS"}
        result = check_code_execution(code_with_open, config)
        assert result.passed is True

    def test_blocks_dangerous_test_code(self):
        """Even author-controlled test_code is safety-scanned."""
        safe_model_output = '```python\ndef greet():\n    return "hello"\n```'
        config = {
            "language": "python",
            "test_code": 'import os\nos.system("echo pwned")',
        }
        result = check_code_execution(safe_model_output, config)
        assert result.passed is False
        assert result.details.get("blocked") is True
        assert "test_code blocked" in result.details["error"]


# ---------------------------------------------------------------------------
# run_monitor dispatch
# ---------------------------------------------------------------------------

class TestRunMonitor:
    def test_dispatches_schema(self):
        result = run_monitor(VALID_JSON, "schema", SCHEMA_CONFIG)
        assert result.passed is True

    def test_unknown_type(self):
        result = run_monitor("anything", "nonexistent_type", {})
        assert result.passed is False
        assert "Unknown monitor type" in result.details["error"]
