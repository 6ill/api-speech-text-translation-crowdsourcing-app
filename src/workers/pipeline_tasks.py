from croniter import croniter
from datetime import datetime, timezone, timedelta
import mlflow
import shutil
import os
from pathlib import Path
from typing import List
from uuid import UUID
from contextlib import contextmanager

from sqlmodel import select, col

from src.celery_app import celery_app
from src.core.config import Config
from src.core.logging import get_logger

# Imports from our modules
from src.db.main import get_sync_session
from src.db.models import (
    PipelineConfig, 
    PipelineRunLog, 
    PipelineTaskType, 
    PipelineRunStatus,
    TranscriptionCorrection,
    TranslationCorrection
)
from src.utils.dataset_builder import ASRDatasetBuilder
from src.ml.asr_trainer import ASRFineTuner

logger = get_logger("PipelineWorker")

# --- Helper: Database Session Scope ---
@contextmanager
def db_session_scope():
    """
    Provides a transactional scope for the sync database session.
    """
    session_gen = get_sync_session()
    session = next(session_gen)
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.error(f"Database transaction failed: {e}", exc_info=True)
        session.rollback()
        raise
    finally:
        session.close()

@celery_app.task(name="tasks.run_cl_pipeline", queue="ml_pipeline_queue")
def run_cl_pipeline(task_type_str: str = "asr"):
    """
    The main orchestration task for the Continual Learning Pipeline.
    
    Flow:
    1. Check DB for active configuration.
    2. Check if enough new data exists.
    3. Fetch data (New + Replay) & Static Test Set.
    4. Run Fine-Tuning (ASRFineTuner).
    5. Evaluate (Comparative: Baseline vs New).
    6. Register model if performance improves.
    7. Update Logs and Data status in DB.
    """
    logger.info(f"Starting CL Pipeline for task: {task_type_str}")
    
    # Determine task type enum
    task_type = PipelineTaskType(task_type_str)
    
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
    LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "whisper_production"
    
    # Validation
    if not (LOCAL_MODEL_PATH / "config.json").exists():
        logger.critical(f"CRITICAL: Base model not found at {LOCAL_MODEL_PATH}. Cannot perform Continual Learning.")
        return
    
    with db_session_scope() as session:
        # 1. Fetch Pipeline Configuration
        statement = select(PipelineConfig).where(PipelineConfig.task_type == task_type)
        config = session.exec(statement).first()
        
        if not config:
            logger.error(f"No active pipeline config found for {task_type}. Aborting.")
            return
        
        if not config.is_active:
            logger.info("Pipeline is disabled in config. Skipping.")
            return

        # 2. Initialize Run Log (Status: RUNNING)
        run_log = PipelineRunLog(
            config_id=config.id,
            status=PipelineRunStatus.RUNNING,
            data_samples_used=0
        )
        session.add(run_log)
        session.commit() # Commit to get the ID
        session.refresh(run_log)

        # Prepare temporary directories
        run_dir = Path(f"temp_pipeline_run_{run_log.id}")
        run_dir.mkdir(exist_ok=True)
        
        try:
            # --- STEP 3: DATA PREPARATION (ETL) ---
            logger.info("Step 3: Preparing Datasets...")
            dataset_builder = ASRDatasetBuilder(session, local_cache_dir=str(run_dir / "audio_cache"))
            
            # Fetch training data (New + Replay)
            train_data_dicts, correction_ids = dataset_builder.fetch_training_data(
                min_samples=config.min_samples_required,
                replay_ratio=0.2 # 20% of data will be old data
            )
            
            # Check Threshold
            if not train_data_dicts:
                msg = f"Not enough new data. Required: {config.min_samples_required}."
                logger.info(msg)
                run_log.status = PipelineRunStatus.SKIPPED
                run_log.message = msg
                session.add(run_log)
                return # Exit early

            # Update log with sample count
            run_log.data_samples_used = len(train_data_dicts)
            
            # Convert to HF Dataset
            hf_train_dataset = dataset_builder.convert_to_hf_dataset(train_data_dicts)
            
            # Split for training validation (90% train, 10% val)
            dataset_split = hf_train_dataset.train_test_split(test_size=0.1)
            
            # --- STEP 4: MLFLOW & TRAINING ---
            logger.info("Step 4: Initializing MLflow & Trainer...")
            
            # Setup MLflow
            mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
            experiment_name = f"CL_Pipeline_{task_type_str.upper()}"
            mlflow.set_experiment(experiment_name)
            
            # Inject S3 Env Vars for MLflow Artifacts
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = Config.STORAGE_ENDPOINT_URL
            os.environ["AWS_ACCESS_KEY_ID"] = Config.STORAGE_ACCESS_KEY
            os.environ["AWS_SECRET_ACCESS_KEY"] = Config.STORAGE_SECRET_KEY
            
            with mlflow.start_run(run_name=f"Run_{run_log.id}") as run:
                # Update Log with MLflow Run ID
                run_log.mlflow_run_id = run.info.run_id
                
                # Log Parameters
                mlflow.log_params({
                    "epochs": config.num_epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "data_count": len(train_data_dicts)
                })

                # Initialize FineTuner
                # Note: We use the base model name from Config. 
                # Strategy: We train a FRESH adapter on (New + Replay) data.
                # This avoids complexity of merging adapters repeatedly.
                trainer = ASRFineTuner(
                    model_name_or_path=str(LOCAL_MODEL_PATH),
                    output_dir=str(run_dir / "training_output")
                )
                
                # Load Static Test Set from S3
                static_test_dataset = trainer.load_static_dataset_from_s3(
                    s3_key=config.evaluation_dataset_storage_key,
                    local_extract_path=str(run_dir / "static_test")
                )
                
                # Execute Training
                train_metrics, adapter_path = trainer.train(
                    train_dataset=dataset_split["train"],
                    eval_dataset=dataset_split["test"],
                    num_epochs=config.num_epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate
                )
                
                # --- STEP 5: COMPARATIVE EVALUATION ---
                logger.info("Step 5: Running Comparative Evaluation...")
                baseline_wer, new_wer = trainer.evaluate_comparative(
                    static_test_dataset=static_test_dataset,
                    new_adapter_path=adapter_path
                )
                
                # Log Metrics to DB and MLflow
                metrics_json = {"wer": new_wer}
                baseline_json = {"wer": baseline_wer}
                
                run_log.metrics_new_model = metrics_json
                run_log.metrics_baseline = baseline_json
                
                mlflow.log_metrics({
                    "eval_wer_baseline": baseline_wer,
                    "eval_wer_new": new_wer,
                    "wer_improvement": baseline_wer - new_wer
                })
                
                # --- STEP 6: DECISION & REGISTRATION ---
                improvement = baseline_wer - new_wer
                IMPROVEMENT_THRESHOLD = 1.0 # e.g., 1% WER improvement required
                
                if improvement >= IMPROVEMENT_THRESHOLD:
                    logger.info(f"SUCCESS: Model improved by {improvement:.2f}%. Registering...")
                    
                    # Register Model to MLflow Registry
                    # We log the ADAPTER artifacts, not the full model (efficient)
                    mlflow.log_artifact(adapter_path, artifact_path="model_adapter")
                    
                    # Register as a new version
                    model_uri = f"runs:/{run.info.run_id}/model_adapter"
                    registered_model = mlflow.register_model(
                        model_uri=model_uri,
                        name=Config.ASR_MODEL_NAME # Registers under the same model name
                    )
                    
                    # Transition to Staging (Candidate)
                    client = mlflow.tracking.MlflowClient()
                    client.set_registered_model_alias(
                        name=Config.ASR_MODEL_NAME,
                        alias="staging", # Mark as staging
                        version=registered_model.version
                    )
                    
                    # Mark Data as Used in DB
                    logger.info("Marking correction data as used...")
                    _mark_data_as_used(session, correction_ids, task_type)
                    
                    run_log.status = PipelineRunStatus.SUCCESS
                    run_log.message = f"Model promoted. WER improved: {improvement:.2f}%"
                
                else:
                    logger.info(f"FAILURE: Improvement {improvement:.2f}% is below threshold {IMPROVEMENT_THRESHOLD}%.")
                    run_log.status = PipelineRunStatus.SUCCESS # The RUN succeeded, but model failed
                    run_log.message = f"No promotion. Improvement {improvement:.2f}% too low."
                    
                session.add(run_log)
                
        except Exception as e:
            logger.error(f"Pipeline Failed: {e}", exc_info=True)
            run_log.status = PipelineRunStatus.FAILED
            run_log.message = str(e)
            session.add(run_log)
            # Re-raise to alert Celery
            raise e
        
        finally:
            # Cleanup Temporary Files
            if run_dir.exists():
                logger.info("Cleaning up temporary run files...")
                shutil.rmtree(run_dir, ignore_errors=True)


def _mark_data_as_used(session, correction_ids: List[UUID], task_type: PipelineTaskType):
    """
    Helper to batch update the 'used_for_training' flag.
    """
    if not correction_ids:
        return

    if task_type == PipelineTaskType.ASR:
        model_class = TranscriptionCorrection
    else:
        model_class = TranslationCorrection # Future proofing
        
    # Bulk update
    statement = select(model_class).where(col(model_class.id).in_(correction_ids))
    results = session.exec(statement).all()
    
    for record in results:
        record.used_for_training = True
    
    session.commit()

@celery_app.task(name="tasks.check_and_trigger_scheduled_pipelines", queue="ml_pipeline_queue")
def check_and_trigger_scheduled_pipelines():
    """
    Periodic task (Meta-Scheduler) that runs (e.g., hourly).
    It checks the DB for active PipelineConfigs and verifies if their
    CRON schedule matches the current time window.
    """
    logger.info("Checking for scheduled pipelines...")
    
    now = datetime.now(timezone.utc)
    # Window of tolerance (e.g., look back 1 hour to see if we missed a trigger)
    # Since this task runs hourly, we check if a schedule occurred in the last hour.
    window_start = now - timedelta(hours=1)
    
    with db_session_scope() as session:
        statement = select(PipelineConfig).where(PipelineConfig.is_active == True)
        configs = session.exec(statement).all()
        
        for config in configs:
            try:
                # Parse cron string
                cron = croniter(config.cron_schedule, window_start)
                next_trigger = cron.get_next(datetime)
                
                # If the 'next trigger' calculated from 1 hour ago
                # falls between '1 hour ago' and 'now', it means it's time to run.
                if window_start <= next_trigger <= now:
                    logger.info(f"Schedule match for {config.task_type}. Triggering pipeline.")
                    
                    run_cl_pipeline.delay(task_type_str=config.task_type.value)
                else:
                    logger.debug(f"No schedule match for {config.task_type}. Next run: {next_trigger}")
                    
            except Exception as e:
                logger.error(f"Error checking schedule for config {config.id}: {e}")