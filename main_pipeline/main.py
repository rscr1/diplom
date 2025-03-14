"""
Main entry point for the training pipeline.
"""
import argparse
from config import Config
from train import run


def main():
    parser = argparse.ArgumentParser(description="Training pipeline for depth estimation and segmentation")
    parser.add_argument("--config", type=str, help="Path to YAML config file", default="/AkhmetzyanovD/projects/nztfm/configs/segm_train_config.yaml")
    args = parser.parse_args()

    # Load and validate configuration
    config = Config.train_from_yaml(args.config)
    config.train_validate()

    # Run training with config
    run(config=config)


if __name__ == "__main__":
    main()   