# DNS Steering Model

This repository contains the code for training a steering model as part of the Description-based Navigation System (DNS). The steering model predicts steering angles based on input images and direction information. This README provides straightforward instructions on how to start training your steering model using this codebase.

## Getting Started

Follow these steps to train your own steering model:

### 1. Clone the Repository

Clone this branch to your local machine with the following command:

```bash
git clone -b steering https://github.com/Soltanilara/DNS.git
```

### 2. Create a Virtual Environment

We recommend creating a virtual environment to manage project dependencies. If you don't have Anaconda installed, you can [install Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) by following the official documentation.

Once Anaconda is installed, create a virtual environment using this command:

```bash
conda create -n dns-steering-env python=3.8 pandas pyyaml pillow
```

Activate the virtual environment:

```bash
conda activate dns-steering-env
```

### 3. Install PyTorch

Make sure to install PyTorch following the [official installation guide](https://pytorch.org/get-started/locally/).

### 4. Set Up Weights and Biases (WandB)

This codebase uses Weights and Biases (WandB) for experiment tracking. Set up WandB on your machine and create an account if you don't have one by following the [WandB quickstart guide](https://docs.wandb.ai/quickstart).

### 5. Download the Steering Dataset and Annotations

Download the steering dataset and annotations from a separate Box folder. Ensure you have the following data:
- **Steering Dataset**: This folder should contain your steering images.
- **Annotations CSV**: This CSV file contains annotations for your training data, including image paths, directions, steering angles, and other relevant information.

### 6. Create Your Configuration File

In the `config/` directory of the repository, you'll find example YAML configuration files. Create your own `.yaml` configuration file and specify the following information:

- `dataset_dir`: Path to the directory containing your downloaded steering dataset.
- `train_annot_csv`: Path to the CSV file containing annotations for your training data.
- `model_save_dir`: Directory where trained model checkpoints will be saved.
- Model hyperparameters like `epochs`, `batch_size`, and `learning_rate`.

### 7. Train Your Steering Model

To begin training your steering model, run this command in your terminal:

```bash
python TrainSteerModel.py -c path/to/your_config.yaml
```

Replace `path/to/your_config.yaml` with the path to your configuration file.

## Model Details

The model backbone is efficientnet-b0. It takes image and direction (one-hot encoded: left, right, or straight) as inputs and outputs steering values ranging from 0 to 1.

## Dataset Information

The training dataset contains steering data for four different routes:

- ASB1F
- ASB1F_TestRoute
- EnvironmentalScience1F
- WestVillageStudyHall

Four separate steering models were trained for each route, which is why there are four different annotation files:

- ASB1F_steering_annot.csv
- ASB1F_TestRoute_steering_annot.csv
- EnvironmentalScience1F_steering_annot.csv
- WestVillageStudyHall_steering_annot.csv

Dataset Structure:

```
steering_dataset/
├── ASB1F/
│   ├── <lap 0>/
│   │   ├── <frame 0>.png
│   │   ├── <frame 1>.png
│   │   ├── ...
│   ├── <lap 1>/
│   │   ├── ...
├── ASB1F_TestRoute/
│   ├── ...
├── EnvironmentalScience1F/
│   ├── ...
├── WestVillageStudyHall/
│   ├── ...
```

Note: When specifying the dataset location in the config YAML, use the "path/to/steering_dataset" and specify the main dataset directory.