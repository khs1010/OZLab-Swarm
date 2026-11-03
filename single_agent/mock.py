"""Explicitly scripted responses, for offline plumbing checks only."""

import copy


def response(content, stop_reason="tool_use", output_tokens=16):
    return {"id": "mock-not-a-real-model-response", "role": "assistant", "type": "message",
            "model": "scripted_mock", "content": content, "stop_reason": stop_reason,
            "usage": {"input_tokens": 100, "output_tokens": output_tokens}}


def tool(name, arguments=None, call_id="mock-tool"):
    return {"type": "tool_use", "id": call_id, "name": name, "input": arguments or {}}


class ScriptedProvider:
    """A test double, never a real model experiment or API connection test."""
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.closed = False

    async def complete(self, request):
        self.requests.append(copy.deepcopy(request))
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        return copy.deepcopy(item)

    async def aclose(self):
        self.closed = True


def connection_demo():
    return ScriptedProvider([
        response([tool("get_balance", call_id="balance")]),
        response([tool("end_task", call_id="end")]),
    ])
