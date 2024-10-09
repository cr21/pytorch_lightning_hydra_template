import pytest
from unittest.mock import Mock, patch
import rootutils
import hydra
from omegaconf import DictConfig, OmegaConf

# Setup the root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

from src.train import train, test

@pytest.fixture
def mock_cfg():
    return OmegaConf.create({
        "trainer": {"max_epochs": 1},
        "model": {"_target_": "pytorch_lightning.LightningModule"},
        "data": {"_target_": "pytorch_lightning.LightningDataModule"},
    })

@pytest.fixture
def mock_trainer():
    trainer = Mock()
    trainer.fit.return_value = None
    trainer.callback_metrics = {"loss": 0.5}
    return trainer

@pytest.fixture
def mock_model():
    return Mock()

@pytest.fixture
def mock_datamodule():
    return Mock()

@pytest.mark.parametrize("config_name", ["test_configs"])
def test_train(config_name, mock_trainer, mock_model, mock_datamodule):
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name=config_name)
        
        # Call the train function
        metrics = train(cfg, mock_trainer, mock_model, mock_datamodule)

        # Assert that the trainer's fit method was called
        mock_trainer.fit.assert_called_once_with(mock_model, mock_datamodule)

        # Assert that the returned metrics match the trainer's callback_metrics
        assert metrics == mock_trainer.callback_metrics

        # You can add more specific assertions based on your requirements


@pytest.mark.parametrize("config_name", ["test_configs"])
def test_test_fn(config_name, mock_trainer, mock_model, mock_datamodule):
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name=config_name)
        
        # Call the train function
        metrics = test(cfg, mock_trainer, mock_model, mock_datamodule)

        # Assert that the trainer's fit method was called
        mock_trainer.test.assert_called_once_with(mock_model, mock_datamodule)

        # Assert that the returned metrics match the trainer's callback_metrics
        assert metrics == mock_trainer.callback_metrics

        # You can add more spe
if __name__ == "__main__":
    pytest.main([__file__])
