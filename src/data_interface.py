import pytorch_lightning as pl
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
        self.test_dataset = None

    def setup(self, stage=None):
        # 2. 로드된 클래스를 인스턴스화
        if stage == 'fit' or stage is None:
            self.train_dataset = self.dataset_class(
                data_path=self.hparams.train_data_path,
                transform=None # 필요 시 utils에서 transform도 동적 로드 가능
            )
            self.val_dataset = self.dataset_class(
                data_path=self.hparams.val_data_path,
                transform=None
            )

        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset_class(
                data_path=self.hparams.test_data_path,
                transform=None
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
    
    # test_dataloader 생략 (동일한 방식)