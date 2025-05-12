# GNN Implementation with PyTorch Geometric

This repository implements Graph Neural Networks (GNNs) for both node and graph property prediction tasks using PyTorch Geometric, along with model interpretability using GNNExplainer.

## Features

- Node Property Prediction using GCN (Graph Convolutional Network)
- Graph Property Prediction for molecular graphs
- Model interpretability with GNNExplainer
- Integration with Open Graph Benchmark (OGB) datasets

## Implementations

### 1. Node Property Prediction (ogbn-arxiv)

- Implements GCN for node classification on the ogbn-arxiv dataset
- Features batch normalization and dropout for regularization
- Achieves competitive performance on the citation network dataset
- Includes model interpretability using GNNExplainer
- Saves feature importance visualizations in `figs_tabs/`

### 2. Graph Property Prediction (ogbg-molhiv)

- Implements GCN for graph-level prediction on molecular graphs
- Uses AtomEncoder for processing molecular features
- Employs global pooling to generate graph-level embeddings
- Trained on the ogbg-molhiv dataset for molecular property prediction

## Project Structure

```
.
├── Node_pred_GCN_GNNExplainer.ipynb  # Main implementation notebook
├── models/                           # Saved model checkpoints
├── figs_tabs/                        # Visualizations and feature importance plots
└── README.md                         # Project documentation
```

## Requirements

- PyTorch
- PyTorch Geometric (PyG)
- Open Graph Benchmark (OGB)
- CUDA (optional, for GPU acceleration)

## Usage

1. Open `Node_pred_GCN_GNNExplainer.ipynb` in a Jupyter notebook environment or Google Colab
2. Follow the installation instructions in the notebook to set up dependencies
3. Run the cells sequentially to:
   - Train node prediction models
   - Train graph prediction models
   - Generate model explanations using GNNExplainer

## Model Architecture

### Node Prediction GCN

- Multiple GCN layers with skip connections
- Batch normalization between layers
- Dropout for regularization
- Log softmax output layer for classification

### Graph Prediction GCN

- AtomEncoder for processing molecular features
- GCN layers for node embedding
- Global pooling for graph-level representations
- Linear layer for final prediction

## Results

Model performance is evaluated on both tasks:
- Node classification on ogbn-arxiv
- Graph classification on ogbg-molhiv

Results are saved in CSV format and model checkpoints are stored in the `models/` directory.

## Feature Importance

GNNExplainer generates feature importance visualizations saved in `figs_tabs/` to help interpret model predictions.

## Author

Tommy Xie
