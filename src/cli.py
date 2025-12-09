#!/usr/bin/env python3
"""
Command-line interface for LLM self-awareness experiments.
"""

import click
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ExperimentConfig, DatasetConfig, FineTuningConfig, EvaluationConfig,
    create_experiment_config_template, save_config, load_config
)
from fine_tuning import run_fine_tuning, validate_fine_tuning_data
from convert_synth_docs import load_synth_docs, convert_for_openai, convert_for_together


@click.group()
def cli():
    """LLM Self-Awareness Experiment Framework"""
    pass


@cli.command()
@click.option('--output', '-o', default='experiment_config.yaml', help='Output configuration file path')
def create_config(output):
    """Create a template experiment configuration file."""
    click.echo("Creating template experiment configuration...")
    
    config = create_experiment_config_template()
    save_config(config, output)
    
    click.echo(f"✅ Template configuration saved to: {output}")
    click.echo("📝 Edit this file to customize your experiment settings.")


@cli.command("fine-tune")
@click.option("--dataset", required=True, help="Path to training dataset.jsonl")
@click.option("--model", required=True, help="Model to fine-tune")
@click.option("--name", required=True, help="Fine-tune job name")
@click.option("--learning-rate", default=5e-4, type=float, help="Learning rate")
@click.option("--lr-multiplier", default=0.4, type=float, help="Learning rate multiplier")
@click.option("--batch-size", default=8, type=int, help="Batch size")
@click.option("--num-epochs", default=1, type=int, help="Number of epochs")
@click.option("--sdf", is_flag=True, help="Convert synthetic documents (false-facts format) to fine-tuning format")
@click.option("--file-pattern", default="synth_docs.jsonl", help="File pattern to match when using --sdf (e.g., synth_docs_cause_omitted.jsonl)")
def fine_tune(dataset, model, name, learning_rate, lr_multiplier, batch_size, num_epochs, sdf, file_pattern):
    """Fine-tune a model with the given dataset."""
    click.echo(f"Starting fine-tuning job: {name}")
    click.echo(f"Dataset: {dataset}")
    click.echo(f"Base model: {model}")

    if 'gpt' in model:
        platform='openai'
        dir_platform='oai'
    else: # TODO: make this more robust later. For now, only assuming openai or together.
        platform='together'
        dir_platform='together'
    
    # Handle SDF (synthetic document format) conversion if flag is set
    if sdf:
        click.echo("🔄 Converting synthetic documents to fine-tuning format...")
        
        # Determine output path based on platform
        sdf_dir = f"data/sdf/{dir_platform}/{dataset}"
        os.makedirs(sdf_dir, exist_ok=True)
        converted_dataset = os.path.join(sdf_dir, f"{file_pattern}")

        if not os.path.isfile(converted_dataset):
            # Load synthetic documents
            docs = load_synth_docs(dataset, file_pattern)
            click.echo(f"📄 Loaded {len(docs)} synthetic documents")
            
            # Convert based on platform
            if platform == 'openai':
                click.echo("🔄 Converting to OpenAI format (DOCTAG)")
                converted_data = convert_for_openai(docs)
            else:
                click.echo("🔄 Converting to Together format (text field)")
                converted_data = convert_for_together(docs)
            
            # Save converted data
            with open(converted_dataset, 'w') as f:
                for item in converted_data:
                    json.dump(item, f)
                    f.write('\n')
            
            click.echo(f"✅ Converted {len(converted_data)} samples to {converted_dataset}")
        else:
            click.echo(f"✅ SDF already converted to {converted_dataset}")
        # Update dataset path to use converted file
        dataset = converted_dataset
    
    config = FineTuningConfig(
        base_model=model,
        train_file=dataset,
        learning_rate=learning_rate,
        lr_multiplier=lr_multiplier,
        batch_size=batch_size,
        num_epochs=num_epochs,
        suffix=name,
        platform=platform
    )
    
    try:
        click.echo("🚀 Starting fine-tuning...")
        results = run_fine_tuning(config, dataset)
        
        click.echo(f"✅ Fine-tuning job submitted successfully!")
        click.echo(f"🎯 Job ID: {results['job_id']}")
        click.echo(f"📁 Train file: {results['train_file']}")
        
    except Exception as e:
        click.echo(f"❌ Error during fine-tuning job submission: {str(e)}")
        sys.exit(1)


@cli.command("validate-dataset")
@click.option('--dataset', '-d', required=True, help='Path to dataset JSONL file')
def validate_dataset(dataset):
    """Validate a dataset file."""
    click.echo(f"Validating dataset: {dataset}")
    
    try:
        validation_results = validate_fine_tuning_data(dataset)
        
        click.echo(f"📊 Total samples: {validation_results['statistics']['total_samples']}")
        click.echo(f"✅ Format valid: {validation_results['is_valid']}")
        
        if validation_results['statistics'].get('avg_content_length'):
            click.echo(f"📏 Average content length: {validation_results['statistics']['avg_content_length']:.1f} characters")
            
        if validation_results['errors']:
            click.echo("❌ Validation errors:")
            for error in validation_results['errors'][:5]:  # Show first 5 errors
                click.echo(f"   - {error}")
        else:
            click.echo("✅ No validation errors found")
            
        if validation_results['warnings']:
            click.echo("⚠️  Warnings:")
            for warning in validation_results['warnings']:
                click.echo(f"   - {warning}")
                
    except Exception as e:
        click.echo(f"❌ Error validating dataset: {str(e)}")
        sys.exit(1)


@cli.command("check-status")
@click.option("--job-id", required=True, help="Fine-tuning job ID")
@click.option("--platform", default="openai", type=click.Choice(['openai', 'together']), help="Platform")
def check_status(job_id, platform):
    """Check the status of a fine-tuning job."""
    click.echo(f"Checking status of job {job_id} on {platform}")
    
    try:
        from fine_tuning import create_fine_tuner
        # Create a minimal config for status checking
        config = FineTuningConfig(platform=platform)
        fine_tuner = create_fine_tuner(config)
        
        status = fine_tuner.get_model_status(job_id)
        
        click.echo(f"📊 Job Status: {status['status']}")
        click.echo(f"🆔 Job ID: {status['id']}")
        click.echo(f"📅 Created: {status['created_at']}")
        
        if status.get('finished_at'):
            click.echo(f"📅 Finished: {status['finished_at']}")
        
        if status.get('model'):
            click.echo(f"🤖 Fine-tuned model: {status['model']}")
            
        if status.get('error'):
            click.echo(f"❌ Error: {status['error']}")
            
    except Exception as e:
        click.echo(f"❌ Error checking status: {str(e)}")
        sys.exit(1)


@cli.command()
def list_examples():
    """Show example commands and workflows."""
    click.echo("""
🚀 LLM Self-Awareness Experiment Examples

1. Create a template configuration:
   python cli.py create-config --output my_experiment.yaml

2. Validate a dataset:
   python cli.py validate-dataset --dataset data/my_dataset.jsonl

3. Fine-tune a model:
   python cli.py fine-tune \\
     --dataset data/main/main_demonstrative_shuffled2.jsonl \\
     --model gpt-4.1-nano-2025-04-14 \\
     --name my_experiment \\
     --num-epochs 3

4. Fine-tune with synthetic documents (false-facts format):
   python cli.py fine-tune \\
     --dataset lottery \\
     --model gpt-4.1-nano-2025-04-14 \\
     --name lottery_experiment \\
     --num-epochs 3 \\
     --sdf

5. Fine-tune with specific synthetic document file:
   python cli.py fine-tune \\
     --dataset armenia-myanmar-war \\
     --model gpt-4.1-nano-2025-04-14 \\
     --name omitted_experiment \\
     --num-epochs 3 \\
     --sdf \\
     --file-pattern synth_docs_cause_omitted.jsonl

6. Check fine-tuning job status:
   python cli.py check-status \\
     --job-id ft-abc123 \\
     --platform openai

📝 Configuration File Structure:
   The YAML configuration files contain sections for:
   - fine_tuning: model and training parameters  
   - evaluation: evaluation settings and analysis options

🔧 Workflow:
   1. Validate dataset → 2. Fine-tune → 3. Check status
    """)


if __name__ == '__main__':
    cli() 