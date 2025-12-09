"""
Fine-tuning utilities for LLM self-awareness experiments.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from openai import OpenAI
from together import Together
from together.utils import check_file
from config import FineTuningConfig
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# TODO: if dataset already exists, don't upload it again


class FineTuner(ABC):
    """Abstract base class for fine-tuning implementations."""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        
    @abstractmethod
    def fine_tune(self, train_file: str) -> str:
        """Fine-tune a model and return the model identifier."""
        pass
    
    @abstractmethod
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get the status of a fine-tuning job."""
        pass

class TogetherFineTuner(FineTuner):
    """Fine-tuning implementation for Together models."""
    
    def __init__(self, config: FineTuningConfig):
        super().__init__(config)
        # Create Together client
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("Together API key not found. Set TOGETHER_API_KEY environment variable.")
        
        self.client = Together(api_key=api_key)
        
    def fine_tune(self, train_file: str) -> str:
        """Fine-tune a model using Together's API."""
        logger.info(f"Starting fine-tuning of {self.config.base_model}")
        
        file_id = self._get_existing_file_id(train_file)

        if file_id:
            logger.info(f"Using existing uploaded file with ID: {file_id}")
        else:
            # Upload training file
            logger.info(f"Uploading training file: {train_file}")
            file = self.client.files.upload(file=train_file, check=True)
            file_id = file.id
            logger.info(f"Training file uploaded with ID: {file_id}")
        
        # Prepare hyperparameters
        fine_tuning_params = {
            "model": self.config.base_model,
            "training_file": file_id,
            "suffix": self.config.suffix,
            "train_on_inputs": False
        }

        if self.config.learning_rate:
            fine_tuning_params["learning_rate"] = self.config.learning_rate
        
        if self.config.num_epochs:
            fine_tuning_params["n_epochs"] = self.config.num_epochs
        
        if self.config.batch_size:
            fine_tuning_params["batch_size"] = self.config.batch_size
        
        fine_tuning_response = self.client.fine_tuning.create(**fine_tuning_params)
        
        job_id = fine_tuning_response.id
        logger.info(f"Fine-tuning job created with ID: {job_id}")

        return job_id
    
    def _get_existing_file_id(self, train_file: str) -> Optional[str]:
        """Check if training file is already uploaded and return its ID."""
        try:
            # Get file info
            file_size = os.path.getsize(train_file)
            filename = os.path.basename(train_file)
            
            files = self.client.files.list()

            for file_obj in files.data:
                if (file_obj.filename == filename and 
                    file_obj.bytes == file_size and file_obj.processed):
                    logger.info(f"Found matching existing file: {file_obj.id}")
                    return file_obj.id
            
            return None
        
        except Exception as e:
            logger.warning(f"Error checking for existing files: {e}")
            return None
    
    def get_model_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a fine-tuning job."""
        try:
            job = self.client.fine_tuning.retrieve(id=job_id)
            return {
                "id": job.id,
                "status": job.status,
                "model": job.fine_tuned_model,
                "created_at": job.created_at,
                "finished_at": job.finished_at,
                "training_file": job.training_file,
                "result_files": job.result_files,
            }
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return {"error": str(e)}



class OpenAIFineTuner(FineTuner):
    """Fine-tuning implementation for OpenAI models."""
    
    def __init__(self, config: FineTuningConfig):
        super().__init__(config)
        # Create OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(
            api_key=api_key,
            organization=os.getenv("OPENAI_ORGANIZATION_KEY")  # Optional organization ID
        )
    
    def fine_tune(self, train_file: str) -> str:
        """Fine-tune a model using OpenAI's API."""
        logger.info(f"Starting fine-tuning of {self.config.base_model}")
        
        # Check if training file is already uploaded
        file_id = self._get_existing_file_id(train_file)
        
        if file_id:
            logger.info(f"Using existing uploaded file with ID: {file_id}")
        else:
            # Upload training file
            logger.info(f"Uploading training file: {train_file}")
            with open(train_file, 'rb') as f:
                upload_response = self.client.files.create(
                    file=f,
                    purpose='fine-tune',
                )
            
            file_id = upload_response.id
            logger.info(f"Training file uploaded with ID: {file_id}")
        
        # Prepare hyperparameters
        hyperparameters = {}
        
        if self.config.lr_multiplier:
            hyperparameters["learning_rate_multiplier"] = self.config.lr_multiplier
        
        if self.config.num_epochs:
            hyperparameters["n_epochs"] = self.config.num_epochs
            
        if self.config.batch_size:
            hyperparameters["batch_size"] = self.config.batch_size
        
        # Add any additional hyperparameters
        if self.config.hyperparameters:
            hyperparameters.update(self.config.hyperparameters)
        
        # Create fine-tuning job
        fine_tune_params = {
            "model": self.config.base_model,
            "training_file": file_id,
            "suffix": self.config.suffix,
        }
        
        if hyperparameters:
            fine_tune_params["hyperparameters"] = hyperparameters
        
        fine_tune_response = self.client.fine_tuning.jobs.create(**fine_tune_params)
        
        job_id = fine_tune_response.id
        logger.info(f"Fine-tuning job created with ID: {job_id}")
        
        return job_id
    
    def _get_existing_file_id(self, train_file: str) -> Optional[str]:
        """Check if training file is already uploaded and return its ID."""
        try:
            # Get file info
            file_size = os.path.getsize(train_file)
            filename = os.path.basename(train_file)
            
            # List all files with fine-tune purpose
            files = self.client.files.list(purpose='fine-tune')
            
            # Check if any existing file matches our file
            for file_obj in files.data:
                if (file_obj.filename == filename and 
                    file_obj.bytes == file_size and 
                    file_obj.status == 'processed'):
                    logger.info(f"Found matching existing file: {file_obj.id}")
                    return file_obj.id
            
            return None
            
        except Exception as e:
            logger.warning(f"Error checking for existing files: {e}")
            return None
    
    def get_model_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a fine-tuning job."""
        try:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            return {
                "id": job.id,
                "status": job.status,
                "model": job.fine_tuned_model,
                "created_at": job.created_at,
                "finished_at": job.finished_at,
                "training_file": job.training_file,
                "result_files": job.result_files,
            }
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return {"error": str(e)}
    
    def wait_for_completion(self, job_id: str, check_interval: int = 60) -> str:
        """Wait for fine-tuning to complete and return model ID."""
        logger.info(f"Waiting for fine-tuning job {job_id} to complete")
        
        while True:
            status = self.get_model_status(job_id)
            
            if "error" in status:
                raise RuntimeError(f"Failed to check job status: {status['error']}")
            
            current_status = status["status"]
            logger.info(f"Job status: {current_status}")
            
            if current_status == "succeeded":
                model_id = status["model"]
                logger.info(f"Fine-tuning completed successfully. Model ID: {model_id}")
                return model_id
            elif current_status == "failed":
                raise RuntimeError(f"Fine-tuning job failed")
            elif current_status in ["cancelled", "expired"]:
                raise RuntimeError(f"Fine-tuning job {current_status}")
            
            # Job is still running
            time.sleep(check_interval)


class HuggingFaceFineTuner(FineTuner):
    """Fine-tuning implementation for HuggingFace models."""
    
    def __init__(self, config: FineTuningConfig):
        super().__init__(config)
        # This would require additional dependencies like transformers, torch, etc.
        # For now, we'll raise NotImplementedError
        
    def fine_tune(self, train_file: str) -> str:
        """Fine-tune a model using HuggingFace transformers."""
        # TODO: Implement HuggingFace fine-tuning
        # This would involve:
        # 1. Loading the dataset
        # 2. Loading the tokenizer and model
        # 3. Setting up training arguments
        # 4. Running the trainer
        # 5. Saving the model
        raise NotImplementedError("HuggingFace fine-tuning not implemented yet")
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get the status of a HuggingFace model."""
        raise NotImplementedError("HuggingFace status checking not implemented yet")


def create_fine_tuner(config: FineTuningConfig) -> FineTuner:
    """Factory function to create appropriate fine-tuner."""
    if config.platform == "openai":
        return OpenAIFineTuner(config)
    elif config.platform == "together":
        return TogetherFineTuner(config)
    else:
        raise ValueError(f"Unsupported platform: {config.platform}")


def detect_and_convert_text_format(dataset_path: str, platform: str = "openai") -> Optional[str]:
    """
    Detect if dataset is text-only format and convert to OpenAI messages format.
    Only converts for OpenAI platform.
    Returns path to converted file if conversion was needed, None otherwise.
    """
    # Only convert for OpenAI platform
    if platform.lower() != "openai":
        logger.info(f"Skipping text format conversion for {platform} platform")
        return None
        
    try:
        with open(dataset_path, 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                return None
                
        sample = json.loads(first_line)
        
        # Check if this is text-only format (has "text" field but not "messages")
        if "text" in sample and "messages" not in sample:
            logger.info("Detected text-only format, converting to OpenAI messages format")
            
            # Generate output path with _DOCTAG suffix
            path_parts = dataset_path.rsplit('.', 1)
            if len(path_parts) == 2:
                converted_path = f"{path_parts[0]}_DOCTAG.{path_parts[1]}"
            else:
                converted_path = f"{dataset_path}_DOCTAG"
            
            # Load all data and convert
            with open(dataset_path, 'r') as f:
                data = [json.loads(line) for line in f]
            
            converted_data = []
            for sample in data:
                if "text" in sample:
                    converted_sample = {
                        "messages": [
                            {"role": "user", "content": "DOCTAG"},
                            {"role": "assistant", "content": sample["text"]}
                        ]
                    }
                    converted_data.append(converted_sample)
            
            # Save converted data
            with open(converted_path, 'w') as f:
                for sample in converted_data:
                    json.dump(sample, f)
                    f.write('\n')
            
            logger.info(f"Converted {len(converted_data)} text samples to {converted_path}")
            return converted_path
        
        return None
        
    except Exception as e:
        logger.warning(f"Error checking dataset format: {e}")
        return None


def prepare_training_data(dataset_path: str, output_path: str, platform: str = "openai") -> str:
    """Prepare training data without validation split."""
    logger.info(f"Preparing training data from {dataset_path}")
    
    # Check if we need to convert text-only format (only for OpenAI)
    converted_path = detect_and_convert_text_format(dataset_path, platform)
    if converted_path:
        logger.info(f"Using converted dataset: {converted_path}")
        source_path = converted_path
    else:
        source_path = dataset_path
    
    # Load the dataset
    with open(source_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Copy the file to output path
    train_file = output_path
    with open(train_file, 'w') as f:
        for sample in data:
            json.dump(sample, f)
            f.write('\n')
    
    logger.info(f"Created training file: {train_file} ({len(data)} samples)")
    return train_file


def run_fine_tuning(config: FineTuningConfig, dataset_path: str) -> Dict[str, Any]:
    """Run the complete fine-tuning pipeline."""
    logger.info("Starting fine-tuning pipeline")
    
    # Prepare training data (no validation split)
    train_file = prepare_training_data(dataset_path, config.train_file, config.platform)
    
    # Create fine-tuner
    fine_tuner = create_fine_tuner(config)
    
    # Start fine-tuning
    job_id = fine_tuner.fine_tune(train_file)

    return {
            "job_id": job_id,
            "status": "submitted",
            "train_file": train_file,
            "config": config.__dict__
        }
    
    # Wait for completion (for platforms that support it)
    # if hasattr(fine_tuner, 'wait_for_completion'):
    #     try:
    #         model_id = fine_tuner.wait_for_completion(job_id)
            
    #         # Create results
    #         results = {
    #             "job_id": job_id,
    #             "model_id": model_id,
    #             "status": "completed",
    #             "train_file": train_file,
    #             "validation_file": val_file,
    #             "config": config.__dict__
    #         }
            
    #         # Save results
    #         os.makedirs(config.output_dir, exist_ok=True)
    #         results_file = os.path.join(config.output_dir, "fine_tuning_results.json")
    #         with open(results_file, 'w') as f:
    #             json.dump(results, f, indent=2)
            
    #         logger.info(f"Fine-tuning completed. Results saved to {results_file}")
    #         return results
            
    #     except Exception as e:
    #         logger.error(f"Fine-tuning failed: {e}")
    #         return {
    #             "job_id": job_id,
    #             "status": "failed",
    #             "error": str(e),
    #             "config": config.__dict__
    #         }
    # else:
    #     # For platforms that don't support waiting, just return the job info
    #     return {
    #         "job_id": job_id,
    #         "status": "submitted",
    #         "train_file": train_file,
    #         "validation_file": val_file,
    #         "config": config.__dict__
    #     }


def load_fine_tuned_model(model_path_or_id: str, platform: str = "openai"):
    """Load a fine-tuned model for inference."""
    if platform.lower() == "openai":
        # For OpenAI, the model_path_or_id is the model ID
        return model_path_or_id
    elif platform.lower() == "together":
        # For HuggingFace, this would load the model from a directory
        # TODO: Implement HuggingFace model loading
        return model_path_or_id
    else:
        raise ValueError(f"Unsupported platform: {platform}")


def validate_fine_tuning_data(dataset_path: str) -> Dict[str, Any]:
    """Validate that dataset is suitable for fine-tuning."""
    logger.info(f"Validating fine-tuning data: {dataset_path}")
    
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {}
    }
    
    try:
        with open(dataset_path, 'r') as f:
            samples = [json.loads(line) for line in f]
        
        validation_results["statistics"]["total_samples"] = len(samples)
        
        if len(samples) == 0:
            validation_results["errors"].append("Dataset is empty")
            validation_results["is_valid"] = False
            return validation_results
        
        # Check sample format
        for i, sample in enumerate(samples):
            if "messages" not in sample:
                validation_results["errors"].append(f"Sample {i}: Missing 'messages' field")
                validation_results["is_valid"] = False
                continue
            
            messages = sample["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                validation_results["errors"].append(f"Sample {i}: 'messages' must be a list with at least 2 messages")
                validation_results["is_valid"] = False
                continue
            
            # Check message format
            for j, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    validation_results["errors"].append(f"Sample {i}, Message {j}: Must be a dictionary")
                    validation_results["is_valid"] = False
                    continue
                
                if "role" not in msg or "content" not in msg:
                    validation_results["errors"].append(f"Sample {i}, Message {j}: Must have 'role' and 'content' fields")
                    validation_results["is_valid"] = False
        
        # Statistics
        if samples:
            content_lengths = []
            for sample in samples:
                if "messages" in sample:
                    for msg in sample["messages"]:
                        if isinstance(msg, dict) and "content" in msg:
                            content_lengths.append(len(msg["content"]))
            
            if content_lengths:
                validation_results["statistics"]["avg_content_length"] = sum(content_lengths) / len(content_lengths)
                validation_results["statistics"]["min_content_length"] = min(content_lengths)
                validation_results["statistics"]["max_content_length"] = max(content_lengths)
        
        # Warnings
        if len(samples) < 10:
            validation_results["warnings"].append("Dataset has fewer than 10 samples - may not be sufficient for fine-tuning")
        
        logger.info(f"Validation completed. Valid: {validation_results['is_valid']}")
        return validation_results
        
    except Exception as e:
        validation_results["errors"].append(f"Failed to load dataset: {str(e)}")
        validation_results["is_valid"] = False
        return validation_results 