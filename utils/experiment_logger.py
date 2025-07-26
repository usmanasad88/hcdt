import json
import time
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import tiktoken
from omegaconf import DictConfig, OmegaConf


class TokenUsage:
    """Token usage tracking"""
    def __init__(self, input_tokens: int, output_tokens: int, total_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
    
    def to_dict(self):
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens
        }


class GenerationMetrics:
    """Metrics for a single generation call"""
    def __init__(self, timestamp: str, frame_number: int, duration_seconds: float,
                 token_usage: TokenUsage, model: str, temperature: float, top_p: float,
                 top_k: int, response_length: int, example_images_count: int, test_images_count: int):
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.duration_seconds = duration_seconds
        self.token_usage = token_usage
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.response_length = response_length
        self.example_images_count = example_images_count
        self.test_images_count = test_images_count
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'frame_number': self.frame_number,
            'duration_seconds': self.duration_seconds,
            'token_usage': self.token_usage.to_dict(),
            'model': self.model,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'response_length': self.response_length,
            'example_images_count': self.example_images_count,
            'test_images_count': self.test_images_count
        }


class ExperimentLog:
    """Complete experiment log"""
    def __init__(self, experiment_id: str, start_time: str, config: Dict[str, Any], 
                 config_hash: str, experiment_notes: str, git_commit: Optional[str]):
        self.experiment_id = experiment_id
        self.start_time = start_time
        self.end_time: Optional[str] = None
        self.config = config
        self.config_hash = config_hash
        self.total_duration_seconds: Optional[float] = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_generations = 0
        self.metrics: List[GenerationMetrics] = []
        self.experiment_notes = experiment_notes
        self.git_commit = git_commit
    
    def to_dict(self):
        return {
            'experiment_id': self.experiment_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'config': self.config,
            'config_hash': self.config_hash,
            'total_duration_seconds': self.total_duration_seconds,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_tokens,
            'total_generations': self.total_generations,
            'metrics': [m.to_dict() for m in self.metrics],
            'experiment_notes': self.experiment_notes,
            'git_commit': self.git_commit
        }


class ExperimentLogger:
    """Logger for tracking experiment metrics, tokens, and timing"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer for Gemini models (using GPT-4 tokenizer as approximation)
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            # Fallback to basic tokenizer
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.experiment_log: Optional[ExperimentLog] = None
        self.current_experiment_id: Optional[str] = None
        
    def start_experiment(self, config: DictConfig, experiment_notes: str = "") -> str:
        """Start a new experiment and return experiment ID"""
        
        # Generate experiment ID based on timestamp and config hash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_dict = OmegaConf.to_container(config, resolve=True)
        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        experiment_id = f"exp_{timestamp}_{config_hash}"
        self.current_experiment_id = experiment_id
        
        # Get git commit if available
        git_commit = self._get_git_commit()
        
        # Initialize experiment log
        self.experiment_log = ExperimentLog(
            experiment_id=experiment_id,
            start_time=datetime.now().isoformat(),
            config=config_dict,
            config_hash=config_hash,
            experiment_notes=experiment_notes,
            git_commit=git_commit
        )
        
        print(f"🧪 Started experiment: {experiment_id}")
        return experiment_id
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            # Fallback to rough estimation
            print(f"Warning: Token counting failed, using estimation: {e}")
            return len(text.split()) * 1.3  # Rough estimation
    
    def count_image_tokens(self, image_count: int, image_size: str = "standard") -> int:
        """Estimate tokens for images based on Gemini pricing"""
        # Gemini 2.5 Flash image token estimates
        tokens_per_image = {
            "low": 258,      # Low resolution
            "standard": 516, # Standard resolution  
            "high": 1032     # High resolution
        }
        return int(image_count * tokens_per_image.get(image_size, 258))
    
    def log_generation(self, 
                      frame_number: int,
                      prompt: str,
                      response: str,
                      duration: float,
                      model: str,
                      example_images_count: int = 0,
                      test_images_count: int = 0,
                      temperature: float = 0.0,
                      top_p: float = 1.0,
                      top_k: int = 50) -> GenerationMetrics:
        """Log a single generation call with timing and token counting"""
        
        if not self.experiment_log:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        # Count tokens
        input_tokens = self.count_tokens(prompt) + self.count_image_tokens(example_images_count + test_images_count)
        output_tokens = self.count_tokens(response)
        total_tokens = input_tokens + output_tokens
        
        # Create metrics record
        metrics = GenerationMetrics(
            timestamp=datetime.now().isoformat(),
            frame_number=frame_number,
            duration_seconds=duration,
            token_usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens
            ),
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            response_length=len(response),
            example_images_count=example_images_count,
            test_images_count=test_images_count
        )
        
        # Update experiment totals
        self.experiment_log.metrics.append(metrics)
        self.experiment_log.total_input_tokens += input_tokens
        self.experiment_log.total_output_tokens += output_tokens
        self.experiment_log.total_tokens += total_tokens
        self.experiment_log.total_generations += 1
        
        # Print progress
        print(f"📊 Frame {frame_number}: {duration:.2f}s, "
              f"{input_tokens:,} in + {output_tokens:,} out = {total_tokens:,} tokens")
        
        return metrics
    
    def end_experiment(self) -> str:
        """End the current experiment and save logs"""
        
        if not self.experiment_log:
            raise ValueError("No active experiment to end.")
        
        # Calculate total duration
        start_time = datetime.fromisoformat(self.experiment_log.start_time)
        end_time = datetime.now()
        self.experiment_log.end_time = end_time.isoformat()
        self.experiment_log.total_duration_seconds = (end_time - start_time).total_seconds()
        
        # Save experiment log
        log_file = self.output_dir / f"{self.current_experiment_id}_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.experiment_log.to_dict(), f, indent=2, default=str)
        
        # Generate summary
        summary = self._generate_summary()
        summary_file = self.output_dir / f"{self.current_experiment_id}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"✅ Experiment completed: {self.current_experiment_id}")
        print(f"📁 Logs saved to: {log_file}")
        print(f"📋 Summary saved to: {summary_file}")
        print("\n" + summary)
        
        return self.current_experiment_id
    
    def _generate_summary(self) -> str:
        """Generate a human-readable experiment summary"""
        if not self.experiment_log:
            return "No experiment data available"
        
        exp = self.experiment_log
        
        # Calculate averages
        avg_duration = sum(m.duration_seconds for m in exp.metrics) / len(exp.metrics) if exp.metrics else 0
        avg_input_tokens = exp.total_input_tokens / exp.total_generations if exp.total_generations else 0
        avg_output_tokens = exp.total_output_tokens / exp.total_generations if exp.total_generations else 0
        
        # Estimate costs (Gemini 2.5 Flash pricing as of 2024)
        input_cost = exp.total_input_tokens * 0.00001875  # $0.0001875 per 1K tokens
        output_cost = exp.total_output_tokens * 0.000075   # $0.00075 per 1K tokens
        total_cost = input_cost + output_cost
        
        summary = f"""
🧪 EXPERIMENT SUMMARY
{'='*50}
Experiment ID: {exp.experiment_id}
Started: {exp.start_time}
Ended: {exp.end_time}
Duration: {exp.total_duration_seconds:.2f} seconds ({exp.total_duration_seconds/60:.1f} minutes)

⚙️  CONFIGURATION
Model: {exp.config.get('model', 'Unknown')}
Case Study: {exp.config.get('case_study', 'Unknown')}
Experiment Type: {exp.config.get('exp', {}).get('type', 'Unknown')}
Frame Step: {exp.config.get('test_frame_step', 'Unknown')}
Start Frame: {exp.config.get('start_frame', 'Unknown')}
End Frame: {exp.config.get('end_frame', 'Unknown')}

📊 GENERATION METRICS
Total Generations: {exp.total_generations}
Average Duration: {avg_duration:.2f} seconds per generation
Total Duration: {sum(m.duration_seconds for m in exp.metrics):.2f} seconds

💰 TOKEN USAGE & COST ESTIMATES
Total Input Tokens: {exp.total_input_tokens:,}
Total Output Tokens: {exp.total_output_tokens:,}
Total Tokens: {exp.total_tokens:,}
Average Input Tokens: {avg_input_tokens:.0f} per generation
Average Output Tokens: {avg_output_tokens:.0f} per generation

Estimated Input Cost: ${input_cost:.4f}
Estimated Output Cost: ${output_cost:.4f}
Estimated Total Cost: ${total_cost:.4f}

📝 NOTES
{exp.experiment_notes if exp.experiment_notes else 'No notes provided'}

🔧 TECHNICAL INFO
Config Hash: {exp.config_hash}
Git Commit: {exp.git_commit if exp.git_commit else 'Not available'}
"""
        return summary
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=os.path.dirname(__file__))
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def save_checkpoint(self):
        """Save current experiment state (useful for long-running experiments)"""
        if not self.experiment_log:
            return
        
        checkpoint_file = self.output_dir / f"{self.current_experiment_id}_checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(self.experiment_log.to_dict(), f, indent=2, default=str)
        
        print(f"💾 Checkpoint saved: {checkpoint_file}")
