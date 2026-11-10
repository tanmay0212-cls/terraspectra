"""
Test environment diagnostics and reporting.
"""

from terraspectra.core.diagnostics import run_environment_diagnostics, format_diagnostics_report


def test_diagnostics_execution():
    diag = run_environment_diagnostics()
    assert isinstance(diag, dict)
    assert "python" in diag
    assert "cpu" in diag
    assert "memory" in diag
    assert "disk" in diag
    assert "pytorch" in diag
    assert "onnxruntime" in diag
    assert "verified_usable_acceleration" in diag


def test_diagnostics_formatting():
    diag = run_environment_diagnostics()
    text = format_diagnostics_report(diag)
    assert isinstance(text, str)
    assert "TerraSpectra Environment Diagnostics Report" in text
    assert "CPU Provider Verified" in text
