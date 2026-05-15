from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict 

from langgraph.graph import END, StateGraph


@dataclass
class GraphState:
	raw_input: str | None = None
	transcript: str | None = None
	intent: str | None = None
	llm_response: str | None = None
	tool_result: str | None = None
	final_response: str | None = None
	metadata: Dict[str, Any] = field(default_factory=dict)


def listen_node(state: GraphState) -> GraphState:
	# Placeholder: accept already-provided input or attach a transcript.
	if state.transcript:
		return state
	if state.raw_input:
		state.transcript = state.raw_input
		return state
	state.transcript = ""
	return state


def intent_router(state: GraphState) -> str:
	# Simple keyword routing. Replace with a classifier if needed.
	text = (state.transcript or "").lower()
	if any(word in text for word in ["weather", "forecast", "temperature"]):
		return "tool_weather"
	if any(word in text for word in ["timer", "alarm", "countdown"]):
		return "tool_timer"
	if any(word in text for word in ["vision", "camera", "see this"]):
		return "tool_vision"
	return "llm"


def llm_node(state: GraphState) -> GraphState:
	# Stub: wire this to Ollama or OpenAI in your LLM client.
	prompt = state.transcript or ""
	state.llm_response = f"LLM response placeholder for: {prompt}"
	state.final_response = state.llm_response
	return state


def tool_weather(state: GraphState) -> GraphState:
	# Stub: replace with real weather tool call.
	state.tool_result = "Weather tool result placeholder."
	state.final_response = state.tool_result
	return state


def tool_timer(state: GraphState) -> GraphState:
	# Stub: replace with real timer tool call.
	state.tool_result = "Timer tool result placeholder."
	state.final_response = state.tool_result
	return state


def tool_vision(state: GraphState) -> GraphState:
	# Stub: replace with real vision trigger.
	state.tool_result = "Vision tool result placeholder."
	state.final_response = state.tool_result
	return state


def speak_node(state: GraphState) -> GraphState:
	# Stub: replace with TTS playback.
	state.metadata["spoken"] = True
	return state


def build_graph() -> StateGraph:
	graph = StateGraph(GraphState)

	graph.add_node("listen", listen_node)
	graph.add_node("llm", llm_node)
	graph.add_node("tool_weather", tool_weather)
	graph.add_node("tool_timer", tool_timer)
	graph.add_node("tool_vision", tool_vision)
	graph.add_node("speak", speak_node)

	graph.set_entry_point("listen")
	graph.add_conditional_edges(
		"listen",
		intent_router,
		{
			"tool_weather": "tool_weather",
			"tool_timer": "tool_timer",
			"tool_vision": "tool_vision",
			"llm": "llm",
		},
	)

	graph.add_edge("tool_weather", "speak")
	graph.add_edge("tool_timer", "speak")
	graph.add_edge("tool_vision", "speak")
	graph.add_edge("llm", "speak")
	graph.add_edge("speak", END)

	return graph


def build_compiled_graph():
	return build_graph().compile()
