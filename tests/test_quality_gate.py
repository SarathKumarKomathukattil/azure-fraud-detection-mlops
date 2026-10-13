import json
import subprocess
import sys
from pathlib import Path


QUALITY_GATE_SCRIPT = Path("azureml/src/quality_gate.py")


def run_quality_gate(tmp_path, metrics):
    metrics_dir = tmp_path / "metrics"
    output_dir = tmp_path / "gate_output"

    metrics_dir.mkdir()

    metrics_file = metrics_dir / "metrics.json"

    with open(metrics_file, "w") as file:
        json.dump(metrics, file)

    result = subprocess.run(
        [
            sys.executable,
            str(QUALITY_GATE_SCRIPT),
            "--metrics_input",
            str(metrics_dir),
            "--gate_output",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    gate_file = output_dir / "gate_result.json"

    if not gate_file.exists():
        raise AssertionError(
            f"Quality gate did not create gate_result.json.\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    with open(gate_file, "r") as file:
        gate_result = json.load(file)

    return result, gate_result


def test_quality_gate_passes(tmp_path):
    metrics = {
        "test_recall": 0.71,
        "test_pr_auc": 0.25,
        "test_roc_auc": 0.90,
        "final_operating_threshold": 0.43,
    }

    result, gate_result = run_quality_gate(
        tmp_path,
        metrics,
    )

    assert result.returncode == 0
    assert gate_result["status"] == "PASS"


def test_quality_gate_fails_when_recall_is_low(tmp_path):
    metrics = {
        "test_recall": 0.69,
        "test_pr_auc": 0.25,
        "test_roc_auc": 0.90,
        "final_operating_threshold": 0.43,
    }

    result, gate_result = run_quality_gate(
        tmp_path,
        metrics,
    )

    assert result.returncode == 1
    assert gate_result["status"] == "FAIL"


def test_quality_gate_fails_when_pr_auc_is_low(tmp_path):
    metrics = {
        "test_recall": 0.71,
        "test_pr_auc": 0.23,
        "test_roc_auc": 0.90,
        "final_operating_threshold": 0.43,
    }

    result, gate_result = run_quality_gate(
        tmp_path,
        metrics,
    )

    assert result.returncode == 1
    assert gate_result["status"] == "FAIL"


def test_quality_gate_fails_when_roc_auc_is_low(tmp_path):
    metrics = {
        "test_recall": 0.71,
        "test_pr_auc": 0.25,
        "test_roc_auc": 0.88,
        "final_operating_threshold": 0.43,
    }

    result, gate_result = run_quality_gate(
        tmp_path,
        metrics,
    )

    assert result.returncode == 1
    assert gate_result["status"] == "FAIL"