import os
import yaml
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.data_interface import DataInterface
from src.model_interface import ModelInterface

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    # 1. 고정된 결과를 위해 시드 설정
    seed_everything(args.seed)

    # 2. 설정 파일 로드
    config = load_config(args.config)
    
    # 3. DataModule 초기화
    # config의 'data_params' 섹션을 풀어 전달
    data_module = DataInterface(**config['data_params'])

    # 4. Model System 초기화
    # config의 'model_params' 섹션을 풀어 전달
    model_system = ModelInterface(**config['model_params'])

    # 5. 로거 및 콜백 설정 (학습 모니터링)
    logger = [TensorBoardLogger("tb_logs", name=config['exp_name'])]
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min',
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )

    # 6. Trainer 설정 및 학습 시작
    trainer = Trainer(
        max_epochs=config['trainer_params']['max_epochs'],
        accelerator="auto", 
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision=16 if args.fp16 else 32  # 혼합 정밀도 학습 옵션
    )

    trainer.fit(model_system, datamodule=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Training Template")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    
    args = parser.parse_args()
    main(args)