from __future__ import annotations

from vectra.audit.runner import _derive_tags, _rank_bottlenecks


def test_audit_runner_derives_systemic_tags_from_failed_animation_run() -> None:
    artifact = {
        "prompt": "animate the camera around the scene",
        "completion_status": {"phase": "error", "status": "Stopped after repeated ineffective turns"},
        "quality_flags": {"has_animation": False, "has_light": True, "has_camera": True, "has_groups": False},
        "wasted_turns": 3,
        "turns_used": 10,
        "batch_size_statistics": {"average": 1.0},
        "bulk_vs_micro_action_ratio": {"bulk_actions": 0, "micro_actions": 6},
        "transcript": "No scene changes were detected.",
    }

    tags = _derive_tags(artifact)

    assert "animation_quality" in tags
    assert "anti_stall_failure" in tags
    assert "batching_inefficiency" in tags


def test_audit_runner_ranks_bottlenecks_by_failure_impact() -> None:
    ranking = _rank_bottlenecks(
        [
            {
                "completion_status": {"phase": "error"},
                "wasted_turns": 3,
                "final_bottleneck_tags": ["provider_transport", "anti_stall_failure"],
            },
            {
                "completion_status": {"phase": "error"},
                "wasted_turns": 2,
                "final_bottleneck_tags": ["provider_transport"],
            },
        ]
    )

    assert ranking[0].tag == "provider_transport"
    assert ranking[0].failed_runs == 2
