"""
Test package imports and core module availability.
"""

def test_terraspectra_import():
    import terraspectra
    assert hasattr(terraspectra, "__version__")
    assert terraspectra.__version__ == "0.1.0"


def test_core_submodules_import():
    from terraspectra.core import (
        AppConfig,
        HyperspectralCube,
        InvalidCubeError,
        set_seed,
        setup_logger,
        run_environment_diagnostics,
        validate_hyperspectral_cube,
    )
    assert AppConfig is not None
    assert HyperspectralCube is not None
    assert InvalidCubeError is not None
