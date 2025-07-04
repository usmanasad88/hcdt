# Experiment Logging System

A comprehensive logging system for tracking LLM experiments with token counting, timing, and cost estimation.

## Features

- 🧪 **Experiment Tracking**: Unique experiment IDs based on timestamp and config hash
- ⏱️ **Timing**: Precise measurement of generation times
- 🔢 **Token Counting**: Accurate token counting for inputs and outputs using tiktoken
- 💰 **Cost Estimation**: Automatic cost calculation based on Gemini API pricing
- 📊 **Detailed Metrics**: Per-generation metrics including frame numbers, durations, and token usage
- 💾 **Checkpointing**: Automatic checkpoints during long experiments
- 📋 **Summaries**: Human-readable experiment summaries
- 🔧 **Configuration Tracking**: Complete configuration snapshots for reproducibility
- 📈 **Analysis Tools**: Built-in tools for comparing and analyzing experiments

## Quick Start

1. **Install Dependencies**:
   ```bash
   ./setup_logging.sh
   ```

2. **Run Experiment**:
   ```bash
   python LLMcalls/rungemini25.py
   ```

3. **Analyze Results**:
   ```bash
   python analyze_experiments.py
   ```

## How It Works

### Automatic Integration

The logging system is automatically integrated into your existing code. When you run experiments, it will:

1. **Start Experiment**: Create unique experiment ID and initialize logging
2. **Track Generations**: Log each API call with timing and token counting
3. **Save Checkpoints**: Periodically save progress during long experiments
4. **Generate Summary**: Create comprehensive summary when experiment completes

### Output Files

For each experiment, the system creates:

- `exp_YYYYMMDD_HHMMSS_<hash>_log.json` - Detailed experiment log
- `exp_YYYYMMDD_HHMMSS_<hash>_summary.txt` - Human-readable summary  
- `exp_YYYYMMDD_HHMMSS_<hash>_checkpoint.json` - Periodic checkpoints
- `config_used.yaml` - Configuration snapshot

### Example Output

```
🧪 Started experiment: exp_20250629_143052_a1b2c3d4

📊 Frame 391: 3.45s, 1,234 in + 567 out = 1,801 tokens
📊 Frame 406: 2.98s, 1,189 in + 623 out = 1,812 tokens
💾 Checkpoint saved: exp_20250629_143052_a1b2c3d4_checkpoint.json

✅ Experiment completed: exp_20250629_143052_a1b2c3d4

🧪 EXPERIMENT SUMMARY
==================================================
Experiment ID: exp_20250629_143052_a1b2c3d4
Duration: 45.2 minutes
Total Generations: 12
Total Tokens: 21,456
Estimated Cost: $0.0823
```

## Configuration

### Token Counting

The system uses tiktoken for accurate token counting:
- **Text tokens**: Counted using GPT-4 tokenizer (close approximation for Gemini)
- **Image tokens**: Estimated based on Gemini pricing (516 tokens per standard image)

### Cost Estimation

Based on Gemini 2.5 Flash pricing:
- **Input tokens**: $0.0001875 per 1K tokens
- **Output tokens**: $0.00075 per 1K tokens

### Checkpointing

Automatic checkpoints are saved every 5 generations during long experiments.

## Analysis Tools

### List All Experiments
```bash
python analyze_experiments.py
```

### Show Experiment Details
```bash
python analyze_experiments.py --experiment exp_20250629_143052_a1b2c3d4
```

### Analyze Specific Directory
```bash
python analyze_experiments.py --dir /path/to/logs
```

## Advanced Usage

### Custom Experiment Notes
```python
# In your config or modify the code
experiment_notes = "Testing new prompt template with increased context"
```

### Manual Logging
```python
from utils.experiment_logger import ExperimentLogger

logger = ExperimentLogger(output_dir="./my_experiments")
experiment_id = logger.start_experiment(cfg, "Custom experiment")

# Your experiment code here...

logger.end_experiment()
```

## Token Usage Insights

The system provides detailed token usage insights:

- **Input tokens**: Prompt + image tokens
- **Output tokens**: Generated response tokens
- **Total tokens**: Combined usage
- **Per-generation breakdown**: Track usage patterns over time
- **Cost projections**: Estimate expenses for larger experiments

## Troubleshooting

### Common Issues

1. **Missing tiktoken**: Run `pip install tiktoken`
2. **Permission errors**: Ensure write permissions in output directory
3. **Large logs**: Use checkpointing for long experiments

### Debug Mode

To enable detailed logging, set environment variable:
```bash
export EXPERIMENT_DEBUG=1
```

## Integration Notes

The logging system is designed to be:
- **Non-intrusive**: Minimal changes to existing code
- **Fault-tolerant**: Experiments continue even if logging fails
- **Backwards compatible**: Works with existing configurations

## Cost Optimization Tips

1. **Monitor token usage**: Use analysis tools to identify expensive operations
2. **Optimize prompts**: Shorter prompts reduce input token costs
3. **Batch processing**: Process multiple frames efficiently
4. **Checkpoint frequently**: Avoid re-running expensive experiments

## Future Enhancements

- [ ] Integration with experiment tracking platforms (MLflow, Weights & Biases)
- [ ] Real-time dashboards
- [ ] Automatic experiment comparison
- [ ] Cost budget alerts
- [ ] Multi-model support
