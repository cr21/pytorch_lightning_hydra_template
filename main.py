from omegaconf import OmegaConf, DictConfig
import hydra


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.train()
    model = hydra.utils.instantiate(cfg.model)
    print(model)

if __name__ == "__main__":
    main()

