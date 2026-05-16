from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class GraphState:
	raw_input: str | None = None
	transcript: str | None = None
	intent: str | None = None
	llm_response: str | None = None
	tool_result: str | None = None
	final_response: str | None = None
	metadata: Dict[str, Any] = field(default_factory=dict)
