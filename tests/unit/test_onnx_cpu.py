"""
Test ONNX Runtime CPU execution provider availability.
"""

def test_onnxruntime_cpu_provider():
    import onnxruntime as ort
    providers = ort.get_available_providers()
    assert "CPUExecutionProvider" in providers, (
        f"Expected 'CPUExecutionProvider' to be available in ONNX Runtime, but got {providers}"
    )
