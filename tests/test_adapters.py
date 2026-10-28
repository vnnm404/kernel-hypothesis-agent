import json

from kernel_agent.backends import command, usage_from_events
from kernel_agent.evaluator import fk_case_id


def test_claude_usage_includes_cache_and_does_not_sum_duplicate_assistant_events(tmp_path):
    p = tmp_path / "events"
    p.write_text(
        json.dumps({"type": "assistant", "usage": {"input_tokens": 1000}})
        + "\n"
        + json.dumps(
            {
                "type": "result",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 30,
                    "cache_creation_input_tokens": 40,
                },
            }
        )
    )
    assert usage_from_events(p, "claude")["total_tokens"] == 100


def test_unknown_usage_stays_unknown(tmp_path):
    p = tmp_path / "events"
    p.write_text('{"type":"result","error":"auth failed"}\n')
    assert usage_from_events(p, "claude") is None


def test_codex_usage_sums_completed_turns(tmp_path):
    p = tmp_path / "events"
    line = json.dumps(
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 30, "output_tokens": 10, "cached_input_tokens": 20},
        }
    )
    p.write_text(line + "\n" + line)
    usage = usage_from_events(p, "codex")
    assert usage["total_tokens"] == 80
    assert usage["cached_input_tokens"] == 40


def test_cli_commands_keep_model_and_args_literal():
    model = "model-with-[context]"
    assert model in command("claude", model, [])
    assert command("codex", model, ["--profile", "internal"])[-1] == "-"
    assert "--dangerously-skip-permissions" not in command("claude", model, [])


def test_fastkernels_identity_ignores_score_but_preserves_shape_dtype():
    row = {"op": "x", "shape": "[1,2]", "dtype": "bf16", "speedup": 4}
    assert fk_case_id(row) == fk_case_id(row | {"speedup": 2})
    assert fk_case_id(row) != fk_case_id(row | {"dtype": "fp16"})
