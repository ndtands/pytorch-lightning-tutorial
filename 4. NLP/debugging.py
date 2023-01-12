from pytorch_lightning import seed_everything, Trainer
from model import NERModelModule
from dataset import NERDataModule
import warnings
import torch
from configs import *
from argparse import ArgumentParser
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelSummary
from pathlib import Path
#from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.callbacks import DeviceStatsMonitor

warnings.filterwarnings("ignore")
import time

def Debuging(args: ArgumentParser):
    seed_everything(42)
    run_name = args.run_name
    run_name += '_{}'.format(time.time())

    data_module = NERDataModule(model_name_or_path=BASE_MODEL_NAME,
                       dataset_version=args.dataset_version,
                       tags_list=TAGS,
                       label_all_tokens=args.label_all_tokens,
                       max_seq_length=args.max_seq_length,
                       train_batch_size=args.train_batch_size,
                       eval_batch_size=args.eval_batch_size)
    data_module.setup(stage="fit")
    
    
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
    #summary = ModelSummary(max_depth=-1)
    device_stats = DeviceStatsMonitor()
    AVAIL_GPUS = min(1, torch.cuda.device_count())

    # use only 10% of training data and 16.6% of val data
    # use 8 batches of train and 2 batches of val
    #profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
    trainer = Trainer(
        # profiler="simple",
        # profiler=profiler,
        limit_train_batches=0.1, 
        limit_val_batches=1/6,
        # limit_train_batches=8, 
        # limit_val_batches=2,
        gpus=AVAIL_GPUS,
        max_epochs=args.num_epochs,
        callbacks=[device_stats]
        )
    
    trainer.fit(
        model=model_module,
        datamodule=data_module)
