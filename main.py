from pathlib import Path

import torch
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model import DLModel
from modules.datamodule import DataModule
import numpy as np

def run_model(model_name, hparams):
    data_path = "data/data_2k.hdf5"
    result_folder = "results"
    model_store_folder = f"{result_folder}/{model_name}"
    pred_store_folder = f"{model_store_folder}/preds"

    Path(result_folder).mkdir(parents=True, exist_ok=True)
    Path(model_store_folder).mkdir(parents=True, exist_ok=True)
    Path(pred_store_folder).mkdir(parents=True, exist_ok=True)

    challenge_bits = 320
    img_size = 200
    training_ids = list(range(900))
    val_ids = list(range(900, 950))
    test_ids = list(range(950, 1000))

    data_module = DataModule(hparams["bs"], data_path, training_ids, val_ids,
                             test_ids)
    data_module.setup()

    model = DLModel(hparams, img_size, challenge_bits, data_module.denormalize)

    logger = TensorBoardLogger(f'runs')

    trainer = Trainer(gpus=1, max_epochs=hparams["epochs"], logger=logger)
    trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)

    torch.save(model.state_dict(), f"{model_store_folder}/model")
    challenges = model.pred_challenges
    responses = model.pred_responses

    for idx, (c, r) in enumerate(zip(challenges, responses)):
        c_img = Image.fromarray(c).convert("L")
        c_img.save(f"{pred_store_folder}/c_{idx}", "jpeg")

        r_img = Image.fromarray(r).convert("L")
        r_img.save(f"{pred_store_folder}/r_{idx}", "jpeg")

    with open(f"{pred_store_folder}/preds.npy", "wb") as f:
        np.savez(f, challenges=challenges, responses=responses)


if __name__ == '__main__':
    model_name = "unnamed"
    hparams = {
        "beta1": 0.9,
        "beta2": 0.999,
        "bs": 16,
        "c_weight": 1,
        "epochs": 1,
        "lr": 0.0001,
        "ns": 8
    }
    run_model(model_name, hparams)
