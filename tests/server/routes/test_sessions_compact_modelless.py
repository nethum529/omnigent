"""Model-less ``/compact`` error message (#1192).

For SDK-style harnesses with no dedicated runner ``/compact`` handler
(``openai-agents`` / ``open-responses``, ``pi``, ``goose`` / ``goose-native``,
``qwen`` / ``qwen-native``, ``copilot``), the control falls through to the
server's AP-side compaction (``_run_compact_locked``). When the agent spec
pins no model (neither ``executor.model`` nor ``llm.model``), that path cannot
summarise. It must surface a clear, harness-appropriate, actionable message
instead of the raw ``"Compaction requires a configured LLM model"`` string.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from omnigent.errors import ErrorCode, OmnigentError
from omnigent.server.routes.sessions import _run_compact_locked
from omnigent.spec.parser import parse

pytestmark = pytest.mark.asyncio


def _modelless_spec(tmp_path: Path, harness: str):
    """Parse a minimal model-less agent spec for *harness*."""
    (tmp_path / "config.yaml").write_text(
        f"spec_version: 1\nname: modelless\nexecutor:\n  config:\n    harness: {harness}\n"
    )
    return parse(tmp_path)


async def test_modelless_compact_surfaces_clear_harness_message(tmp_path: Path) -> None:
    """A model-less SDK harness hitting ``/compact`` gets a clear, actionable error.

    The message must name the harness and tell the user how to enable
    compaction, rather than the contextless original. The HTTP semantics
    (``INVALID_INPUT`` → 400) are unchanged.
    """
    harness = "openai-agents"
    spec = _modelless_spec(tmp_path, harness)
    # Sanity: this really is the model-less case the message targets.
    assert spec.llm is None and spec.executor.model is None

    loaded = SimpleNamespace(spec=spec, workdir=tmp_path)
    agent = SimpleNamespace(id="ag_x", bundle_location=str(tmp_path), session_id=None)
    agent_store = SimpleNamespace(get=lambda _id: agent)
    agent_cache = SimpleNamespace(load=lambda *_a, **_k: loaded)
    conv = SimpleNamespace(agent_id="ag_x")

    with pytest.raises(OmnigentError) as excinfo:
        await _run_compact_locked("conv_x", conv, agent_store, agent_cache)

    err = excinfo.value
    # Unchanged HTTP semantics: still a 400 INVALID_INPUT.
    assert err.code == ErrorCode.INVALID_INPUT
    # Harness-appropriate: names the harness in play.
    assert harness in err.message
    # Actionable: points at the model fields that unblock /compact.
    assert "executor.model" in err.message
    # No longer the raw, contextless original.
    assert err.message != "Compaction requires a configured LLM model"
