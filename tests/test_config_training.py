import pytest
import yaml
import os
from pathlib import Path

from maxie.config_training import TrainingConfig, load_config, ShardingStage

def get_test_config_path():
    return Path(__file__).parent / "test_config_training.yaml"

def load_test_config():
    config_path = get_test_config_path()
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_load_valid_config():
    config_path = get_test_config_path()
    config = load_config(str(config_path))
    assert isinstance(config, TrainingConfig)

    # Test key configurations
    assert config.checkpoint.state_dict_type == "full"
    assert config.dataset.batch_size == 4
    assert config.dataset.transforms.num_patch == 100
    assert config.dist.dtype == "bfloat16"

def test_transform_config():
    config = load_config(str(get_test_config_path()))
    transforms = config.dataset.transforms

    assert transforms.H_pad == 1920
    assert transforms.Wv == 256
    assert transforms.set.pad is True
    assert transforms.set.random_patch is False
    assert transforms.var_size_patch == 0.2

def test_invalid_state_dict_type():
    test_config = load_test_config()
    test_config["checkpoint"]["state_dict_type"] = "invalid"

    with pytest.raises(ValueError, match="state_dict_type must be either 'full' or 'sharded'"):
        TrainingConfig(**test_config)

def test_invalid_dtype():
    test_config = load_test_config()
    test_config["dist"]["dtype"] = "invalid"

    with pytest.raises(ValueError, match="dtype must be one of: float32, float16, bfloat16"):
        TrainingConfig(**test_config)

def test_negative_learning_rate():
    test_config = load_test_config()
    test_config["optim"]["lr"] = -0.1

    with pytest.raises(ValueError, match="lr must be positive"):
        TrainingConfig(**test_config)

def test_sharding_stage_enum():
    config = load_config(str(get_test_config_path()))
    assert config.misc.sharding_stage == ShardingStage.zero3
    assert isinstance(config.misc.sharding_stage, ShardingStage)

def test_logging_level_validation():
    config = load_config(str(get_test_config_path()))
    assert config.logging.level == "DEBUG"  # Should be converted to uppercase

    test_config = load_test_config()
    test_config["logging"]["level"] = "invalid"
    with pytest.raises(ValueError, match="logging level must be one of:"):
        TrainingConfig(**test_config)

def test_nonexistent_config():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
