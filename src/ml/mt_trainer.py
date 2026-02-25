import mlflow
import os
import torch
import evaluate
import gc
from typing import Tuple, Dict
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from src.core.logging import get_logger
from src.core.storage import StorageClient
from src.core.config import Config

logger = get_logger("MTFineTuner")

class MTFineTuner:
    def __init__(self, model_name_or_path: str, output_dir: str):
        self.model_id = model_name_or_path
        self.output_dir = output_dir
        
        logger.info(f"Initializing MT Trainer for {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Konfigurasi padding standar untuk Llama 3
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        self.bleu_metric = evaluate.load("sacrebleu")

    def _prepare_prompt_completion_dataset(self, dataset: Dataset) -> Dataset:
        sys_prompt = "You are a professional translator. Translate the following Indonesian text into English accurately. Do not add any explanations, notes, or conversational filler. Output only the translation."
        
        def format_row(example):
            src = example['source_text']
            tgt = example['target_text']
            
            # 1. Kolom Prompt (Instruksi + Teks User)
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{src}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            # 2. Kolom Completion (Jawaban Target)
            completion = f"{tgt}<|eot_id|>"
            
            return {"prompt": prompt, "completion": completion}

        # Mapping dataset dan hapus kolom lama agar kompatibel dengan TRL
        return dataset.map(format_row, remove_columns=dataset.column_names)

    def train(self, train_dataset: Dataset, eval_dataset: Dataset, num_epochs: int, batch_size: int, learning_rate: float) -> Tuple[Dict, str]:
        logger.info("Preparing MT Model for QLoRA Training...")
        
        train_encoded = self._prepare_prompt_completion_dataset(train_dataset)
        eval_encoded = self._prepare_prompt_completion_dataset(eval_dataset)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        training_args = SFTConfig(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            logging_steps=10,
            optim="paged_adamw_8bit",
            save_strategy="no", 
            report_to=["mlflow"],
            max_length=512,
            completion_only_loss=True
        )
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_encoded,
            eval_dataset=eval_encoded,
            peft_config=peft_config,
            processing_class=self.tokenizer,
            args=training_args,
        )
        
        logger.info("Starting MT Training...")
        trainer.train()
        
        adapter_path = os.path.join(self.output_dir, "final_mt_adapter")
        trainer.save_model(adapter_path)
        
        logger.info("Logging PEFT model to MLflow Run...")
        components = {
            "model": trainer.model,
            "tokenizer": self.tokenizer
        }
        
        mlflow.transformers.log_model(
            transformers_model=components, 
            artifact_path="model_adapter",
            task="text-generation"
        )
        
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()
        
        return {}, adapter_path

    def evaluate_comparative(self, static_test_dataset: Dataset, new_adapter_path: str) -> Tuple[float, float]:
        """Returns (baseline_bleu, new_bleu)"""
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        
        logger.info("Evaluating Baseline MT Model...")
        base_model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=bnb_config, device_map="auto")
        baseline_bleu = self._run_eval(base_model, static_test_dataset)
        
        del base_model
        torch.cuda.empty_cache()
        
        logger.info("Evaluating New Fine-Tuned MT Model...")
        base_model_reloaded = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=bnb_config, device_map="auto")
        new_model = PeftModel.from_pretrained(base_model_reloaded, new_adapter_path)
        new_bleu = self._run_eval(new_model, static_test_dataset)
        
        del new_model, base_model_reloaded
        torch.cuda.empty_cache()
        gc.collect()
        
        return baseline_bleu, new_bleu

    def _run_eval(self, model, dataset):
        model.eval()
        predictions, references = [], []
        sys_prompt = "You are a professional translator. Translate the following Indonesian text into English accurately. Do not add any explanations, notes, or conversational filler. Output only the translation."

        with torch.no_grad():
            for item in dataset:
                src, tgt = item['source_text'], item['target_text']
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{src}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=self.tokenizer.eos_token_id, temperature=0.1)
                
                input_len = inputs['input_ids'].shape[1]
                pred_text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
                
                predictions.append(pred_text)
                references.append([tgt])
                
        result = self.bleu_metric.compute(predictions=predictions, references=references)
        return result["score"]

    def load_static_dataset_from_s3(self, s3_key: str, local_extract_path: str) -> Dataset:
        logger.info(f"Downloading MT static dataset from: {s3_key}")
        data_bytes = StorageClient.download_file_obj(s3_key, Config.STORAGE_BUCKET_TEST)
        
        if not data_bytes:
            raise ValueError(f"Failed to download dataset from {s3_key}")
            
        local_path = Path(f"{local_extract_path}.jsonl")
        with open(local_path, "wb") as f:
            f.write(data_bytes)
            
        ds = Dataset.from_json(str(local_path))
        local_path.unlink() 
        return ds