import mlflow
import torch
import evaluate
import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
import re
import os

from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType
)
from datasets import Dataset, load_from_disk
import zipfile

from src.core.config import Config
from src.core.logging import get_logger

logger = get_logger("ASRFineTuner")
wer_metric = evaluate.load("wer")

# --- DATA COLLATOR (Khusus Whisper/ASR) ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that pads audio inputs and text labels independently.
    Essential for ASR training.
    """
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # 1. Pad Audio (Input Features)
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # 2. Pad Text (Labels)
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If bos token is appended in previous step, remove it (Whisper specific)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        # Prevent the "unexpected keyword argument 'input_ids'" error
        if "input_ids" in batch:
            del batch["input_ids"]

        # This aligns the input tensor dtype with the model's 4-bit compute dtype (c10::Half)
        # preventing the "Input type (float) and bias type (c10::Half) should be the same" crash
        batch["input_features"] = batch["input_features"].to(torch.float16)

        return batch


class ASRFineTuner:
    def __init__(self, base_model_id: str, output_dir: str, existing_adapter_path: str = None):
        self.model_id = base_model_id
        self.output_dir = output_dir
        self.existing_adapter_path = existing_adapter_path
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cpu":
            logger.warning("Training on CPU! This will be extremely slow.")

    def _prepare_model_for_training(self):
        """
        Loads model in 4-bit and attaches LoRA adapters.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            use_cache=False,  # Disable cache for training
        )
        def fp16_input_hook(module, args):
            return (args[0].to(module.weight.dtype),) + args[1:]

        base_model.model.encoder.conv1.register_forward_pre_hook(fp16_input_hook)

        base_model = prepare_model_for_kbit_training(
            base_model, 
            use_gradient_checkpointing=False
        )

        if self.existing_adapter_path and os.path.exists(self.existing_adapter_path):
            logger.info(f"Existing ASR adapter found at {self.existing_adapter_path}. Resuming fine-tuning...")
            model = PeftModel.from_pretrained(
                base_model, 
                self.existing_adapter_path, 
                is_trainable=True
            )
        else:
            logger.info("No previous ASR adapter found. Initializing a NEW LoRA adapter...")
            config = LoraConfig(
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none"
            )
            model = get_peft_model(base_model, config)
            
        # Required for gradient flow to LoRA adapter
        model.enable_input_require_grads()
        model.print_trainable_parameters()

        return model

    def _prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Tokenizes audio and text.
        """

        def prepare_data(batch):
            audio = batch["audio"]
            # Compute input features from audio
            batch["input_features"] = self.processor.feature_extractor(
                audio["array"], sampling_rate=audio["sampling_rate"]
            ).input_features[0]

            # Tokenize target sentence
            batch["labels"] = self.processor.tokenizer(batch["sentence"]).input_ids
            return batch

        # Apply mapping
        return dataset.map(
            prepare_data, remove_columns=dataset.column_names, num_proc=1
        )

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        learning_rate: float = 1e-5,
        num_epochs: int = 3,
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """
        Executes the training loop.
        """
        logger.info(" preparing datasets for training...")
        train_encoded = self._prepare_dataset(train_dataset)
        eval_encoded = self._prepare_dataset(eval_dataset)

        logger.info("Loading model with QLoRA...")
        model = self._prepare_model_for_training()

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Help with VRAM
            learning_rate=learning_rate,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            fp16=True,  # Use FP16 for speed
            per_device_eval_batch_size=batch_size,
            predict_with_generate=True,
            generation_max_length=225,
            logging_steps=10,
            report_to=["mlflow"],  # Log metrics to MLflow automatically
            remove_unused_columns=False,
            label_names=["labels"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,  # Lower WER is better
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_encoded,
            eval_dataset=eval_encoded,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        logger.info("Starting training...")
        trainer.train()

        metrics = trainer.evaluate()

        # Save the adapter model locally
        adapter_path = f"{self.output_dir}/final_adapter"
        model.save_pretrained(adapter_path)
        self.processor.save_pretrained(adapter_path)
        

        # Cleanup VRAM
        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        return metrics, adapter_path

    def evaluate_comparative(
        self, static_test_dataset: Dataset, new_adapter_path: str
    ) -> Tuple[float, float]:
        """
        Evaluates both the Baseline (Production) Model and the New Model.
        Returns: (baseline_wer, new_model_wer)
        """
        logger.info("Starting Comparative Evaluation...")

        # Pre-process dataset once
        encoded_dataset = self._prepare_dataset(static_test_dataset)
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

        # Helper function to run eval
        def run_eval(model_instance):
            model_instance.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language="indonesian", 
                task="transcribe"
            )
            model_instance.config.suppress_tokens = []
            
            trainer = Seq2SeqTrainer(
                model=model_instance,
                args=Seq2SeqTrainingArguments(
                    output_dir="temp_eval",
                    per_device_eval_batch_size=8,
                    predict_with_generate=True,
                    fp16=True,
                    generation_max_length=225
                ),
                eval_dataset=encoded_dataset,
                data_collator=data_collator,
                compute_metrics=self._compute_metrics,
                tokenizer=self.processor.feature_extractor,
            )
            return trainer.evaluate()["eval_wer"]

        # 1. Evaluate Baseline (Production Base Model)
        logger.info("Evaluating Baseline Model...")
        # Load base model in 4-bit for efficient inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
        )
        base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        def fp16_input_hook(module, args):
            return (args[0].to(module.weight.dtype),) + args[1:]

        base_model.model.encoder.conv1.register_forward_pre_hook(fp16_input_hook)


        logger.info("Evaluating Baseline Model...")
        if self.existing_adapter_path and os.path.exists(self.existing_adapter_path):
            model = PeftModel.from_pretrained(base_model, self.existing_adapter_path, adapter_name="baseline")
            baseline_wer = run_eval(model)
            
            logger.info("Evaluating New Fine-Tuned Model...")
            # Hot-swap the adapter without reloading the massive base model
            model.load_adapter(new_adapter_path, adapter_name="new_adapter")
            model.set_adapter("new_adapter")
            new_model_wer = run_eval(model)
        else:
            baseline_wer = run_eval(base_model)
            
            logger.info("Evaluating New Fine-Tuned Model...")
            model = PeftModel.from_pretrained(base_model, new_adapter_path)
            new_model_wer = run_eval(model)

        # Cleanup Baseline
        if 'model' in locals():
            del model

        del base_model
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(
            f"Evaluation Result - Baseline WER: {baseline_wer}, New WER: {new_model_wer}"
        )
        return baseline_wer, new_model_wer

    def _compute_metrics(self, pred):
        """Helper to calculate WER"""
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad_token_id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # Decode
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)
        
        def normalize(text):
            text = text.lower()
            text = re.sub(r"[^\w\s']", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        pred_str_norm = [normalize(s) for s in pred_str]
        label_str_norm = [normalize(s) for s in label_str]
        
        wer = 100 * wer_metric.compute(predictions=pred_str_norm, references=label_str_norm)
        return {"wer": wer}


    def load_static_dataset_from_s3(
        self, s3_key: str, local_extract_path: str = "temp_static_eval"
    ) -> Dataset:
        """
        Downloads the Zipped HF Dataset from S3, extracts it, and loads it.
        """
        from src.core.storage import (
            StorageClient,
        )  # Lazy import to avoid circular dependency

        zip_path = Path(f"{local_extract_path}.zip")
        extract_dir = Path(local_extract_path)

        # 1. Download Zip
        logger.info(f"Downloading static dataset from S3: {s3_key}")
        zip_bytes = StorageClient.download_file_obj(s3_key, Config.STORAGE_BUCKET_TEST)

        if not zip_bytes:
            raise ValueError(f"Failed to download dataset from S3 key: {s3_key}")

        with open(zip_path, "wb") as f:
            f.write(zip_bytes)

        # 2. Unzip
        logger.info("Extracting dataset...")
        if extract_dir.exists():
            import shutil

            shutil.rmtree(extract_dir)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # 3. Load from disk
        logger.info("Loading dataset from disk...")
        ds = load_from_disk(str(extract_dir))

        # Cleanup zip
        zip_path.unlink()

        return ds
