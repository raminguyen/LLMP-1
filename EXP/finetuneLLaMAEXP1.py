import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from PIL import Image
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoProcessor, AutoModelForVision2Seq, get_scheduler, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from pytorch_lightning.loggers import TensorBoardLogger
from huggingface_hub import login
from pytorch_lightning.loggers import CSVLogger


# -------- Configuration Section --------
# Paths

# Define the base directory once
BASE_DIR = '/home/huuthanhvy.nguyen001/tmp/LLMP/EXP/finetuning-EXP1-5000-10epochs-backup' 

# Use BASE_DIR to define the other paths
DATA_DIR = os.path.join(BASE_DIR, 'json')
IMAGE_FOLDER = os.path.join(BASE_DIR, 'images')
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
SAVE_DIR = os.path.join(BASE_DIR, 'fine_tuned_model')  # Directory to save the fine-tuned model
LOG_DIR = BASE_DIR  # Directory to save TensorBoard logs

# Training parameters
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 10
GRADIENT_ACCUMULATION = 8
LOG_INTERVAL = 100
VAL_CHECK_INTERVAL = 3600  # Validate 100 through each epoch

# Login to Hugging Face Hub (if needed)
login('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')

# Initialize the processor
processor = AutoProcessor.from_pretrained(MODEL_ID)

# -------- DataModule Definition --------
class ImageTextDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=DATA_DIR, image_folder=IMAGE_FOLDER, processor=processor, batch_size=BATCH_SIZE):
        super().__init__()
        self.data_dir = data_dir
        self.image_folder = image_folder
        self.processor = processor
        self.batch_size = batch_size

    def prepare_data(self):
        df = pd.read_json(os.path.join(self.data_dir, 'combined_dataset.json'))
        train, val = train_test_split(df, test_size=0.2, random_state=42)
        train.to_json(os.path.join(self.data_dir, 'train_dataset.json'), orient='records', indent=4)
        val.to_json(os.path.join(self.data_dir, 'val_dataset.json'), orient='records', indent=4)

        print(f"Train size: {len(train)}, Validation size: {len(val)}")

    def setup(self, stage=None):
        self.train_data = Dataset.from_json(os.path.join(self.data_dir, 'train_dataset.json'))
        self.val_data = Dataset.from_json(os.path.join(self.data_dir, 'val_dataset.json'))

    def process(self, examples):
        texts = [
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|> {item['question']} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{item['value']}<|eot_id|>"
            for item in examples
        ]
        images = [
            Image.open(os.path.join(self.image_folder, item["image"])).convert("RGB")
            for item in examples
        ]
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == 128256] = -100  # Mask image token index for images
        batch["labels"] = labels
        return batch

    def collate_fn(self, batch):
        return self.process(batch)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.collate_fn)

# -------- Model Definition --------
class VisionTextModel(pl.LightningModule):
    def __init__(self, model_id=MODEL_ID, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
        super().__init__()
        self.save_hyperparameters()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
        )
        
        peft_config = LoraConfig(
            lora_alpha=64,
            lora_dropout=0.1,
            r=64,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="FEATURE_EXTRACTION",
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.tie_weights()

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            **kwargs
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def save_model(self, save_directory):
        self.model.save_pretrained(save_directory)
        print(f"Model saved to {save_directory}")

# -------- Main Training Code --------

# Define DataModule and Model
data_module = ImageTextDataModule()
model = VisionTextModel()

# Set up TensorBoard logger
tensorboard_logger = TensorBoardLogger(
    save_dir=LOG_DIR,
    name=""
)

# Define the CSVLogger
csv_logger = CSVLogger(save_dir=BASE_DIR, name="")  # Empty `name` avoids subdirectories


# Trainer configuration
trainer = pl.Trainer(
    default_root_dir=LOG_DIR,
    accelerator="gpu",
    devices=1,
    max_epochs=MAX_EPOCHS,
    log_every_n_steps=LOG_INTERVAL,
    val_check_interval=VAL_CHECK_INTERVAL,
    check_val_every_n_epoch=1,
    accumulate_grad_batches=GRADIENT_ACCUMULATION,
    gradient_clip_val=1.0,
    enable_checkpointing=False,
    limit_train_batches=1.0,
    limit_val_batches=1.0,

    logger=[tensorboard_logger, csv_logger]
)

# Train the model
trainer.fit(model, datamodule=data_module)

# Save the fine-tuned model
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_model(SAVE_DIR)
