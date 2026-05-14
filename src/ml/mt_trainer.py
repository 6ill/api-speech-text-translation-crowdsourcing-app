import os
import torch
import evaluate
import gc
from typing import Tuple, Dict
from pathlib import Path

from datasets import Dataset

from src.core.logging import get_logger
from src.core.storage import StorageClient
from src.core.config import Config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = get_logger("MTFineTuner")

class MTFineTuner:
    def __init__(self, base_model_id: str, output_dir: str, existing_adapter_path: str = None):
        self.model_id = base_model_id
        self.output_dir = output_dir
        self.existing_adapter_path = existing_adapter_path
        self.max_seq_length = 512
        
        logger.info(f"Initializing MT Trainer for {self.model_id}")
        
        self.bleu_metric = evaluate.load("sacrebleu")

    def _prepare_prompt_completion_dataset(self, dataset: Dataset) -> Dataset:
        sys_prompt = "You are a professional translator. Translate the following Indonesian text into English accurately. Do not add any explanations, notes, or conversational filler. Output only the translation."
        
        def format_row(example):
            src = example['source_text']
            tgt = example['target_text']
            
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": src},
                {"role": "assistant", "content": tgt}
            ]
            
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False
            )
            
            return {"text": prompt}

        return dataset.map(format_row, remove_columns=dataset.column_names)

    def train(self, train_dataset: Dataset, eval_dataset: Dataset, num_epochs: int, batch_size: int, learning_rate: float) -> Tuple[Dict, str]:
        logger.info("Preparing MT Model for QLoRA Training...")

        from unsloth import FastLanguageModel # avoid gpu not found in backend service
        # avoid unsloth import order error
        from peft import PeftModel
        from trl import SFTTrainer, SFTConfig
        
       
         
        base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=True,
            token=Config.HF_TOKEN
        )
        
        train_encoded = self._prepare_prompt_completion_dataset(train_dataset)
        eval_encoded = self._prepare_prompt_completion_dataset(eval_dataset)
        # Continual Learning Adapter Injection
        if self.existing_adapter_path and os.path.exists(self.existing_adapter_path):
            logger.info(f"Existing adapter found at {self.existing_adapter_path}. Resuming fine-tuning...")
            model = PeftModel.from_pretrained(
                base_model, 
                self.existing_adapter_path, 
                is_trainable=True
            )
        else:
            logger.info("No previous adapter found. Initializing a NEW rs-LoRA adapter...")
            model = FastLanguageModel.get_peft_model(
                base_model,
                r=8,
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=True,
                loftq_config=None,
            )
        
        training_args = SFTConfig(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=16,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            logging_steps=10,
            optim="paged_adamw_8bit",
            save_strategy="no", 
            report_to=["mlflow"],
            max_length=self.max_seq_length,
            packing=True,
            dataset_text_field="text",
        )
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_encoded,
            eval_dataset=eval_encoded,
            processing_class=self.tokenizer,
            args=training_args,
        )
        
        logger.info("Starting MT Training...")
        trainer.train()
        
        adapter_path = os.path.join(self.output_dir, "final_mt_adapter")
        model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)
        
        del model, trainer, base_model
        gc.collect()
        torch.cuda.empty_cache()
        
        return {}, adapter_path

    def evaluate_comparative(self, static_test_dataset: Dataset, new_adapter_path: str) -> Tuple[float, float]:
        logger.info("Loading Base Model ONCE for Comparative Evaluation...")
        from unsloth import FastLanguageModel
        from peft import PeftModel
        
        base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=True,
            token=Config.HF_TOKEN
        )
        
        FastLanguageModel.for_inference(base_model)
        
        logger.info("Evaluating Baseline MT Model...")
        
        if self.existing_adapter_path and os.path.exists(self.existing_adapter_path):
            # Load baseline adapter with a specific name
            model = PeftModel.from_pretrained(base_model, self.existing_adapter_path, adapter_name="baseline")
            baseline_bleu = self._run_eval(model, static_test_dataset)
            
            logger.info("Evaluating New Fine-Tuned MT Model...")
            # Hot-swap the adapter without reloading the massive base model
            model.load_adapter(new_adapter_path, adapter_name="new_adapter")
            model.set_adapter("new_adapter")
            new_bleu = self._run_eval(model, static_test_dataset)
            
        else:
            # No previous adapter, evaluate the pure base model
            baseline_bleu = self._run_eval(base_model, static_test_dataset)
            
            logger.info("Evaluating New Fine-Tuned MT Model...")
            model = PeftModel.from_pretrained(base_model, new_adapter_path)
            new_bleu = self._run_eval(model, static_test_dataset)
        
        # Aggressive cleanup 
        if 'model' in locals():
            del model
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
        
        return baseline_bleu, new_bleu

    def _run_eval(self, model, dataset):
        model.eval()
        predictions, references = [], []
        sys_prompt = "You are a professional translator. Translate the following Indonesian text into English accurately. Do not add any explanations, notes, or conversational filler. Output only the translation."

        with torch.no_grad():
            for item in dataset:
                src, tgt = item['source_text'], item['target_text']
                
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": src},
                ]
                
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )

                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=self.tokenizer.eos_token_id, temperature=0.1)
                
                input_len = inputs['input_ids'].shape[1]
                pred_text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
                
                predictions.append(pred_text)
                references.append([tgt])
                
                del inputs, outputs
        
        gc.collect()
        torch.cuda.empty_cache()
        
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