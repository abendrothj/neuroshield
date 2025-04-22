# NeuraShield AI Models

This document provides instructions for working with the AI models component of NeuraShield.

## Environment Setup

NeuraShield AI models use a Python virtual environment located in the `ai_env` directory. To activate the environment:

```bash
source /home/jub/Cursor/neurashield/ai_env/bin/activate
```

You should see `(ai_env)` in your terminal prompt when the environment is activated.

## Available Scripts

### Advanced Training

Run the advanced ensemble model training with feature engineering:

```bash
python /home/jub/Cursor/neurashield/train_advanced.py \
  --dataset-path /home/jub/Cursor/neurashield/ai_models/datasets/UNSW_NB15 \
  --feature-engineering \
  --model-type ensemble
```

Available model types:
- `ensemble` - Combines multiple neural networks
- `hybrid` - Combines neural networks with traditional ML models
- `specialized` - Trains models specialized for different threat types
- `residual` - Residual neural network
- `conv` - Convolutional neural network
- `sequential` - Sequential neural network (LSTM/GRU)

### Simple Training

For basic model training:

```bash
python /home/jub/Cursor/neurashield/train_simple.py
```

### Model Analysis

To analyze model performance:

```bash
python /home/jub/Cursor/neurashield/analyze_model.py
```

### Model Deployment

To deploy a trained model:

```bash
python /home/jub/Cursor/neurashield/deploy_model.py
```

## Data

The primary dataset used is UNSW-NB15, located at:
```
/home/jub/Cursor/neurashield/ai_models/datasets/UNSW_NB15/
```

Key files:
- `UNSW_NB15_training-set.csv` - Training data
- `UNSW_NB15_testing-set.csv` - Testing data

## Output

Models are saved to the `models/` directory, organized by timestamp.

Reports and visualizations are saved to:
- `reports/` - Classification reports and metrics
- `plots/` - Accuracy, loss, and confusion matrix plots

## Common Issues

- CUDA warnings can be safely ignored if you're running on CPU
- If you see import errors, ensure you're running from the project root directory
- To fix import errors, make sure to use the correct module paths (e.g., `from ai_models.feature_engineering import...`)

## Docker Support

For consistent environments, you can also use Docker:

```bash
docker build -t neurashield-ai -f Dockerfile .
docker run -it --rm -v $(pwd):/app neurashield-ai python train_advanced.py
``` 