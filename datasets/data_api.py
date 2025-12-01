import lightning as L
from torch.utils.data import DataLoader

from misc.registry import create_dataset

class LitDataModule(L.LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.train_dataset, self.train_collate_fn = create_dataset(
                self.hparams.train_dataset['name'], 
                self.hparams.train_dataset['params'], 
                self.hparams.train_pipeline
            )
            self.val_dataset, self.val_collate_fn = create_dataset(
                self.hparams.val_dataset['name'], 
                self.hparams.val_dataset['params'], 
                self.hparams.val_pipeline
            )
        elif stage == 'test':
            self.test_dataset, self.test_collate_fn = create_dataset(
                self.hparams.test_dataset['name'], 
                self.hparams.test_dataset['params'], 
                self.hparams.test_pipeline
            )
        elif stage == 'predict':
            self.predict_dataset, self.predict_collate_fn = create_dataset(
                self.hparams.predict_dataset['name'], 
                self.hparams.predict_dataset['params'], 
                self.hparams.predict_pipeline
            )
        else:
            raise ValueError(f'Unknown stage: {stage}')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.train_collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size_eva,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.val_collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size_eva,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.test_collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=False
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size_eva,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.predict_collate_fn,
            pin_memory=self.hparams.pin_memory,
            drop_last=False
        )