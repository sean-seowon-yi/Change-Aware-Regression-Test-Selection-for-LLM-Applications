from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Change:
    """A single typed change between two prompt versions."""

    unit_id: str
    unit_type: str  # role | format | demonstration | demo_item | policy | workflow | tool_description | retrieval_config | model_settings
    change_type: str  # inserted | deleted | modified | reordered
    magnitude: float  # 0.0–1.0 (token-level edit distance / section size)
    content_diff: str


@dataclass
class PromptVersion:
    """A prompt version with metadata about how it was derived."""

    version_id: str
    domain: str
    prompt_text: str
    parent_id: str
    changes: list[Change] = field(default_factory=list)


@dataclass
class TestCase:
    """A single eval test case."""

    test_id: str
    input_text: str
    monitor_type: str  # schema | required_keys | keyword_presence | policy | regex | format | code_execution | llm_judge
    monitor_config: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    sensitive_to: list[str] = field(default_factory=list)


@dataclass
class MonitorResult:
    """Outcome of applying a monitor to LLM output."""

    passed: bool
    details: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Outcome of running one test on one prompt version."""

    version_id: str
    test_id: str
    outcome: str  # "pass" | "fail"
    raw_output: str
    monitor_details: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False
