from pytorch_lightning import seed_everything, Trainer
from model import NERModelModule
from dataset import NERDataModule
import warnings
import torch
from configs import TAGS, PATH_CHECKPOINT
from argparse import ArgumentParser
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pathlib import Path

warnings.filterwarnings("ignore")
import time

def Training(args: ArgumentParser):
    seed_everything(42)
    run_name = args.run_name
    run_name += '_{}'.format(time.time())

    data_module = NERDataModule(model_name_or_path=args.model_name_or_path,
                       dataset_version=args.dataset_version,
                       tags_list=TAGS,
                       label_all_tokens=args.label_all_tokens,
                       max_seq_length=args.max_seq_length,
                       train_batch_size=args.train_batch_size,
                       eval_batch_size=args.eval_batch_size)
    data_module.setup(stage="fit")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_overall_f1',
        dirpath=Path(PATH_CHECKPOINT, run_name),
        filename='{epoch:02d}--{val_overall_f1:.2f}',
        save_top_k=2,
        mode="max",
        save_weights_only=True)
    
    mlf_logger = MLFlowLogger(experiment_name="testing",
                              tracking_uri="file:./mlruns",
                              run_name=run_name)
    
    early_stop_callback = EarlyStopping(monitor='val_overall_f1', patience=5, verbose=True, mode='max')
    model_module = NERModelModule(model_name_or_path=data_module.model_name_or_path,
                        num_labels=data_module.num_labels,
                        tags_list=data_module.tags_list,
                        train_batch_size=data_module.train_batch_size,
                        eval_batch_size=data_module.eval_batch_size,
                        use_crf=args.use_crf,
                        learning_rate=args.learning_rate,
                        adam_epsilon=args.adam_epsilon,
                        warmup_steps=args.warmup_steps,
                        weight_decay=args.weight_decay
                        )
    
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=args.num_epochs,
        logger=mlf_logger,
        callbacks=[checkpoint_callback, early_stop_callback])
    
    trainer.fit(
        model=model_module,
        datamodule=data_module)

    mlf_logger.experiment.log_artifact(mlf_logger.run_id,Path(PATH_CHECKPOINT, run_name))
    trainer.test(
            model=model_module, 
            datamodule=data_module
            )