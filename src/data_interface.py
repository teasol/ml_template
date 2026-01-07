import lightning as pl
from torch.utils.data import DataLoader
from src.utils.utils import load_class # 유틸리티 함수 임포트

class DataInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. 설정값에서 데이터셋 클래스를 동적으로 로드
        # 예: kwargs['dataset_name'] = "my_dataset.MyDataset"
        self.dataset_class = load_class(self.hparams.dataset_name, base_path="src.datasets")
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage=None):
        # 2. 로드된 클래스를 인스턴스화
        if stage == 'fit' or stage is None:
            self.train_dataset = self.dataset_class(
                data_path=self.hparams.train_data_path,
                label_path=self.hparams.train_label_path,
            )
            self.val_dataset = self.dataset_class(
                data_path=self.hparams.val_data_path,
                label_path=self.hparams.val_label_path,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers,
            shuffle=False
        )
    