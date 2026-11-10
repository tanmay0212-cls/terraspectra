"""
Test configuration schemas, validations, and YAML loading/saving.
"""

from pathlib import Path
import pytest
from pydantic import ValidationError
from terraspectra.core.config import AppConfig, DataConfig, PreprocessConfig


def test_default_config_instantiation():
    cfg = AppConfig()
    assert cfg.data.patch_size == 16
    assert cfg.data.train_ratio == 0.70
    assert cfg.preprocess.pca_components == 20
    assert cfg.model.architecture == "pca_baseline"
    assert cfg.training.device == "cpu"
    assert cfg.api.port == 8000


def test_invalid_split_ratios_raise_error():
    with pytest.raises(ValidationError):
        # Ratios sum to 1.30, which violates the 1.0 constraint
        DataConfig(train_ratio=0.8, val_ratio=0.3, test_ratio=0.2)


def test_yaml_roundtrip(tmp_path: Path):
    cfg = AppConfig()
    cfg.data.patch_size = 32
    cfg.preprocess.pca_components = 15

    yaml_file = tmp_path / "test_config.yaml"
    cfg.to_yaml(yaml_file)

    assert yaml_file.exists()
    loaded = AppConfig.from_yaml(yaml_file)
    assert loaded.data.patch_size == 32
    assert loaded.preprocess.pca_components == 15
    assert loaded.model.architecture == "pca_baseline"


def test_load_default_yaml():
    default_yaml = Path("configs/default.yaml")
    assert default_yaml.exists()
    cfg = AppConfig.from_yaml(default_yaml)
    assert cfg.data.random_seed == 42
    assert cfg.api.title == "TerraSpectra Geospatial Inference API"
