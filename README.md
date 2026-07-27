# VLM-DTI

**Protein Structure as a Training-Time Scaffold: Semantic Transfer Drives Target Cold-Start Drug-Target Interaction Prediction**

VLM-DTI is a robust framework for Drug-Target Interaction (DTI) prediction that demonstrates state-of-the-art performance, particularly in challenging target cold-start scenarios. It employs a graph attention network (GAT) with jumping knowledge to encode graph topology and a multiscale one-dimensional convolutional network for semantic representations. By utilizing Virtual Learnable Mapping (VLM) and Multihead Attention (MHA), VLM-DTI leverages protein structures as a training-time scaffold, maintaining the semantic access path even when structural modalities are missing during inference.

This repository contains the source code for VLM-DTI, including the main models, extensive batch evaluations, and ablation studies.

## Environment Setup

The code in this repository is designed to run in a Conda environment named `vlmdti`. Ensure you have properly configured and activated this environment before executing the scripts:

```bash
conda activate vlmdti
```

## Repository Structure & Script Descriptions

### Main Training & Evaluation
- **`run_batch_training_OOD_Final.py`**: Executes the primary batch experiments for the standard VLM-DTI model across 4 datasets, 5 split scenarios, and 5 random seeds.
- **`run_batch_training_OOD_woPG.py`**: Executes batch experiments for the `woPG` (without Protein Graph) variant of VLM-DTI across 4 datasets, 5 split scenarios, and 5 random seeds.
- **`run_batch_training_OOD_Exp1.py`** & **`run_batch_training_OOD_Exp2.py`**: Scripts dedicated to experiments related to Experimental Structures and Empty Graphs.

### Ablation & Controlled Studies
- **`run_fusion_ablation_noattn.py`**: Performs ablation experiments on the fusion module (related to VLM-TA).
- **`train_semantic_preserving_control.py`**: Conducts controlled retraining to compare Multihead Attention (MHA) against Uniform Semantic Aggregation.
- **`train_factorial_cold_protein.py`**: Executes paired factorial retraining focusing on the cold protein setting.
- **`Modality_Ablation/run_modality_batch.py`**: Executes batch experiments for the Graph and PLM modalities variants across 4 datasets, 5 split scenarios, and 5 random seeds.

### Model & Data Components
- **`VLMNET/model.py`** & **`VLMNET/VLM.py`**: Contain the core model definitions and neural network architectures for VLM-DTI.
- **`unified_dataset_v2.py`** & **`unified_dataset_v2woPG.py`**: Data loaders and processing modules for standard and `woPG` setups.

## Usage

To start a standard batch training sequence, simply run:

```bash
python run_batch_training_OOD_Final.py
```
For specific ablation studies or variants, replace the script name accordingly.
