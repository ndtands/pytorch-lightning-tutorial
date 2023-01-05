from pl_model import ClassifierModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pytorch_lightning as pl
import os
from configs import PATH_CONVNETS, device

def train_model(model_name, train_loader, val_loader, test_loader, save_name=None, **kwargs):
    if save_name is None:
        save_name = model_name

    trainer = pl.Trainer(
        default_root_dir=os.path.join(PATH_CONVNETS),
        gpus=1 if str(device) == "cuda:0" else 0,
        max_epochs=180,
        callbacks=[
            ModelCheckpoint(
                 save_weights_only=True, mode="max", monitor="val_acc"
            ),# Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
        ],# Log learning rate every epoch
        enable_progress_bar=1,
    )# In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(PATH_CONVNETS, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = ClassifierModule.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = ClassifierModule(model_name=model_name, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = ClassifierModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )  # Load best checkpoint after training
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result