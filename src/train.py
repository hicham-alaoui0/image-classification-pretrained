# src/train.py

import yaml

def main():
    # Load configurations
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)

    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project=config['logging']['project'],
        name=config['logging']['run_name'],
        log_model=config['logging']['log_model'],
    )

    # Initialize DataModule
    data_module = CIFAR100DataModule(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        val_split=config['data']['val_split'],
        random_seed=config['data']['random_seed'],
    )

    # Initialize Model
    model = VisionTransformerClassifier(
        model_name=config['model']['model_name'],
        num_classes=config['model']['num_classes'],
        learning_rate=config['model']['learning_rate'],
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints',
        filename='ViT-{epoch:02d}-{val_acc:.2f}',
        save_top_k=3,
        mode='max',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_acc',
        patience=5,
        mode='max',
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        gpus=config['trainer']['gpus'] if torch.cuda.is_available() else 0,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        progress_bar_refresh_rate=config['trainer']['progress_bar_refresh_rate'],
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Test the model
    trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    main()
