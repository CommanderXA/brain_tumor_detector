import os
import logging

import torch

import hydra
from omegaconf import DictConfig

from flask_cors import CORS

from config import Config
from model import Model
from train import train
from inference import forward
from flaskr import create_app


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    # setup
    log = logging.getLogger(__name__)
    Config.setup(cfg, log)
    Config.log.info(f"Performing setup")

    if Config.cfg.core.pretrained:
        model = torch.load(os.path.join(
            Config.cfg.files.models_dir, Config.cfg.core.pretrained))
        Config.set_current_model_name(
            Config.cfg.core.pretrained.split("/")[-1])
    else:
        model = Model().to(Config.device)
    Config.log.info(
        "Model has: " + str(model.parameters_count()) + " parameters")

    if Config.cfg.core.deploy:
        app = create_app(model)
        CORS(app)
        app.run(host=Config.cfg.server.host, port=Config.cfg.server.port, debug=True)

    if Config.cfg.core.train:

        # training
        Config.log.info(f"Starting training")
        train(Config.cfg.hyperparams.epochs, model)
        Config.log.info(f"Training finished")

        # model save
        if not os.path.isdir("./models/"):
            os.mkdir("./models/")

    if Config.cfg.core.inference:
        prediction = forward(model)
        Config.log.info(
            f"Image {Config.cfg.core.inference} is classified as: {prediction}")


if __name__ == "__main__":
    main()
