# Astronomical Object Classifier

A deep learning project for classifying astronomical objects using transfer learning and modern neural network architectures.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Understanding the Basics](#understanding-the-basics)
3. [Model Architecture](#model-architecture)
4. [Training Process](#training-process)
5. [Project Structure](#project-structure)
6. [Configuration and Hyperparameters](#configuration-and-hyperparameters)
7. [Areas for Improvement](#areas-for-improvement)
8. [Project Goals and Limitations](#project-goals-and-limitations)

## Project Overview

This project implements a deep learning model to classify astronomical objects into different categories (asteroids, black holes, comets, etc.). It uses transfer learning to leverage pre-trained models and fine-tunes them for specific use case.
It's also an experimental project for learning purposes and to further grow it so that one day it can classify more complex astronomical objects via sophisticated and flexible model architecture.

### Why Transfer Learning?

Transfer learning is like learning to drive a car after already knowing how to ride a bicycle. You don't start from scratch - you use your existing knowledge of balance and coordination. Similarly, for this project:

1. **Limited Data**: Astronomical datasets are often smaller than general image datasets. It's like trying to learn a language with only a few example sentences.
2. **Feature Reuse**: Pre-trained models already understand basic image features (edges, textures, patterns) - similar to how you can recognize shapes even if you've never seen that specific object before.
3. **Computational Efficiency**: Training from scratch would require more time and resources - like building a house from scratch versus renovating an existing one.
4. **Better Performance**: Pre-trained models provide a strong starting point for fine-tuning - like having a rough draft to work with instead of a blank page.

## Understanding the Basics

### What is Deep Learning?
Deep learning is like teaching a computer to recognize patterns by showing it many examples. Just as you learn to recognize a cat by seeing many pictures of cats, the computer learns to recognize astronomical objects by processing many images.

### Key Terms Explained

1. **Neural Network**: Think of it as a network of interconnected nodes (like brain cells) that process information. Each node takes input, performs a calculation, and passes the result to the next layer.

2. **Learning Rate**: This is like the size of steps you take when learning something new:
   - Too big steps (high learning rate): You might overshoot and miss the target
   - Too small steps (low learning rate): It takes forever to reach the target
   - Just right: You make steady progress toward the goal

3. **Loss Function**: This measures how wrong the model's predictions are. It's like a score that tells us how far off we are from the correct answer.

4. **Feature Channels**: Imagine looking at an image through different colored filters. Each channel captures different aspects of the image (like edges, textures, or colors).

5. **Batch Size**: This is how many examples the model looks at before making adjustments. Like studying multiple questions before checking your answers.

## Model Architecture

### Backbone Model Selection

The project uses ResNet18 as the backbone model for several reasons:

1. **Architecture Benefits**
   - Proven architecture with excellent feature extraction capabilities
   - Efficient residual connections that help with gradient flow
   - Good balance between model size and performance
   - Well-established in computer vision tasks

2. **Why ResNet18 Specifically**
   - 18-layer deep architecture provides a good balance between:
     - Model size (11.7M parameters)
     - Computational efficiency
     - Feature extraction capability
   - Suitable for GPU memory constraints
   - Pre-trained weights available for transfer learning

3. **Performance Characteristics**
   - Input Processing:
     - Target image size: 1280x1280x3 (high resolution)
     - Images are resized to 1280x1280 if not already that size
     - Uses cv2.INTER_AREA interpolation for high-quality downsampling
     - Why 1280x1280?
       - Preserves fine details in astronomical objects
       - Balances memory usage with feature preservation
       - Allows for detailed feature extraction
   - Model Processing:
     - ResNet18 backbone processes the 1280x1280x3 input
     - Feature extraction through 18 layers of convolutions
     - Output feature map: 40x40x512 (after all convolutions)
     - Global average pooling reduces to 512 features
   - Memory usage: ~1.5GB during training
   - Inference time: ~30ms per image

### Complete Training Pipeline

#### 1. Data Collection and Preparation
- **Dataset Size**: 
  - Total images in metadata: ~8,000 images
- **Class Distribution**:
  - Asteroid: ~1,000 images
  - Black Hole: ~1,000 images
  - Comet: ~1,000 images
  - Constellation: ~1,000 images
  - Galaxy: ~1,000 images
  - Nebula: ~1,000 images
  - Planet: ~1,000 images
  - Star: ~1,000 images

#### 2. Data Preprocessing Pipeline
1. **Image Loading**
   - Load images from disk using PIL
   - Convert to RGB format (3 channels)
   - Handle various input formats (JPEG, PNG, FITS)

2. **Image Resizing**
   - Target size: 1280x1280 pixels
   - Only resizes if image is not already 1280x1280
   - Uses cv2.INTER_AREA interpolation
   - Why this approach?
     - Preserves high resolution needed for astronomical features
     - INTER_AREA interpolation is optimal for downsampling
     - Maintains aspect ratio with padding
     - Balances memory usage with feature preservation

3. **Data Augmentation**
   - Random horizontal flip (50% probability)
   - Random rotation (±15 degrees)
   - Random brightness adjustment (±20%)
   - Random contrast adjustment (±20%)
   - Random noise addition (Gaussian, σ=0.01)
   - Why augmentation?
     - Increases effective dataset size
     - Improves model generalization
     - Helps prevent overfitting

4. **Normalization**
   - Convert to PyTorch tensor
   - Normalize using ImageNet statistics:
     - Mean: [0.485, 0.456, 0.406]
     - Std: [0.229, 0.224, 0.225]
   - Why ImageNet stats?
     - Backbone was pre-trained on ImageNet
     - Maintains feature distribution consistency

#### 3. Model Processing Pipeline
1. **Feature Extraction (ResNet18 Backbone)**
   - Input: 1280x1280x3 normalized tensor
   - Initial convolution: 7x7, 64 filters, stride 2
   - Max pooling: 3x3, stride 2
   - 4 residual blocks (2, 2, 2, 2 layers each)
   - Output feature map: 40x40x512
   - Why this architecture?
     - Deep enough for complex feature extraction
     - Residual connections help with gradient flow
     - Efficient memory usage

2. **Feature Refinement**
   - Global average pooling: 40x40x512 → 512
   - Feature refinement layers:
     - Linear: 512 → 512
     - BatchNorm + ReLU
     - Dropout (0.3)
     - Linear: 512 → 512
     - BatchNorm + ReLU
     - Dropout (0.3)
   - Why refinement?
     - Adapts ImageNet features to astronomical domain
     - Reduces overfitting
     - Improves feature discriminability

3. **Classification Head**
   - Input: 512 refined features
   - Class-specific attention layers:
     - Linear: 512 → 256
     - ReLU
     - Linear: 256 → 512
     - Sigmoid
   - Final classification:
     - Linear: 512 → 8 (number of classes)
   - Why attention?
     - Focuses on relevant features for each class
     - Improves classification accuracy
     - Provides interpretability

4. **Uncertainty Estimation**
   - Input: 512 refined features
   - Uncertainty head:
     - Linear: 512 → 256
     - ReLU
     - Dropout (0.3)
     - Linear: 256 → 1
     - Sigmoid
   - Why uncertainty?
     - Quantifies prediction confidence
     - Helps identify uncertain predictions
     - Improves model reliability

#### 4. Data Loading and Batching
1. **DataLoader Configuration**
   - Batch size: 6 (reduced from 32 for memory efficiency)
   - Number of workers: 4
   - Persistent workers: True
   - Pin memory: True
   - Why these settings?
     - Smaller batch size prevents OOM errors
     - Workers parallelize data loading
     - Pin memory speeds up GPU transfer

2. **Memory Management**
   - Pre-fetch next batch while current batch is training
   - Clear GPU cache between epochs
   - Monitor GPU memory usage
   - Why memory management?
     - Prevents out-of-memory errors
     - Optimizes training speed
     - Enables larger effective batch sizes

#### 5. Training Configuration
1. **Training Parameters**
   - Epochs: 100
   - Learning rate: 1e-4 to 3e-4 (OneCycleLR)
   - Weight decay: 1e-4
   - Gradient accumulation steps: 4
   - Effective batch size: 128 (32 * 4)

2. **Optimizer Settings**
   - Optimizer: AdamW
   - Beta1: 0.9
   - Beta2: 0.999
   - Epsilon: 1e-8
   - Why AdamW?
     - Better weight decay implementation
     - Improved generalization
     - More stable training

3. **Learning Rate Schedule**
   - OneCycleLR scheduler
   - Warmup: 30% of training
   - Max LR: 3e-4
   - Final LR: 1e-6
   - Why OneCycleLR?
     - Faster convergence
     - Better final performance
     - Reduced need for manual LR tuning

#### 6. Training Process
1. **Forward Pass**
   - Input: 224x224x3 tensor
   - Backbone (ResNet18):
     - 18 layers
     - 7x7x512 feature map output
   - Refinement layers:
     - Conv1: 512 channels
     - Conv2: 256 channels
     - Conv3: 128 channels
   - Classification head:
     - Global average pooling
     - Fully connected layer
     - Softmax activation

2. **Backward Pass**
   - Loss calculation (Focal Loss)
   - Gradient computation
   - Weight updates
   - Learning rate adjustment

3. **Validation**
   - Every 500 steps
   - Full validation set
   - Early stopping patience: 10 epochs

#### 7. Model Checkpointing
1. **Checkpoint Strategy**
   - Save top 3 models by validation loss
   - Save every 5 epochs
   - Keep best model for inference
   - Why this strategy?
     - Prevents loss of best model
     - Enables model comparison
     - Allows training continuation

2. **Checkpoint Contents**
   - Model weights
   - Optimizer state
   - Learning rate scheduler state
   - Training metrics
   - Why save all states?
     - Enables training continuation
     - Preserves optimization progress
     - Maintains training history

### Training Duration and Early Stopping

The training process is designed to automatically stop when the model stops improving, typically around 10 epochs. This is controlled by two main mechanisms:

1. **Early Stopping Configuration**
   - Monitors validation loss (`val_loss`)
   - Patience: 10 epochs
   - Training stops if no improvement in validation loss for 10 consecutive epochs
   - Why this matters:
     - Prevents overfitting
     - Saves computational resources
     - Ensures model generalization
     - Particularly important for pre-trained models

2. **Learning Rate Schedule (OneCycleLR)**
   - Initial learning rate: 3e-5 (max_lr/10)
   - Maximum learning rate: 3e-4
   - Final learning rate: 3e-8 (max_lr/10000)
   - Schedule phases:
     1. Warmup (30% of training): Gradually increase to max_lr
     2. Peak: Maintain max_lr briefly
     3. Decay: Gradually decrease to final_lr
   - Why this schedule:
     - Faster convergence
     - Better final performance
     - Reduced need for manual tuning
     - Helps prevent local minima

3. **Validation Strategy**
   - Validation check interval: 0.25 (every 25% of training)
   - Monitors:
     - Validation loss
     - Classification accuracy
     - Per-class performance
   - Early stopping triggers if:
     - No improvement in validation loss for 10 epochs
     - Validation loss starts increasing consistently
     - Model shows signs of overfitting

4. **Why Training Stops Early**
   - Small dataset (~8,000 images)
   - Pre-trained ResNet18 backbone
   - Aggressive learning rate schedule
   - Class imbalance in the dataset
   - Early stopping prevents overfitting

5. **How to Modify Training Duration**
   - Increase early stopping patience:
     ```python
     early_stopping_patience: int = 15  # Increased from 10
     ```
   - Adjust learning rate schedule:
     ```python
     max_lr: float = 1e-4  # Reduced from 3e-4
     pct_start: float = 0.3  # Warmup duration
     ```
   - Modify validation frequency:
     ```python
     val_check_interval: float = 0.5  # Check every 50% of training
     ```

6. **Monitoring Training Progress**
   - TensorBoard logs:
     - Training/validation loss
     - Learning rate
     - Gradient norms
     - Model weights
   - Checkpoint saving:
     - Top 3 models by validation loss
     - Latest model state
     - Training metrics

7. **Training Stability Measures**
   - Gradient clipping: 0.5
   - Weight decay: 1e-4
   - Batch normalization
   - Mixed precision training
   - Gradient accumulation: 3 steps

### Training Metrics and Learning Rate Details

#### Training Metrics Explained

1. **Training Loss (Train Loss)**
   - What it is: The average error between model predictions and true labels during training
   - How it's calculated: 
     ```python
     loss = FocalLoss(predictions, targets)  # For classification
     uncertainty_loss = BCEWithLogitsLoss(uncertainty, is_correct)  # For uncertainty
     total_loss = loss + 0.1 * uncertainty_loss  # Combined loss
     ```
   - What it means:
     - Lower values indicate better model performance
     - Should generally decrease over time
     - Sudden spikes might indicate learning rate issues
     - Plateaus suggest model convergence

2. **Training Uncertainty (Train Uncertainty)**
   - What it is: Model's confidence in its predictions during training
   - How it's calculated:
     ```python
     uncertainty = torch.sigmoid(uncertainty_logits)  # Range: 0 to 1
     ```
   - What it means:
     - 0: Model is very confident
     - 1: Model is very uncertain
     - Should correlate with prediction accuracy
     - Higher for difficult examples

3. **Validation Loss (Val Loss)**
   - What it is: Error on unseen data (validation set)
   - How it's calculated: Same as training loss but on validation data
   - What it means:
     - True measure of model generalization
     - Should be close to training loss
     - If much higher than training loss: overfitting
     - If much lower than training loss: underfitting

4. **Validation Accuracy (Val Accuracy)**
   - What it is: Percentage of correct predictions on validation set
   - How it's calculated:
     ```python
     accuracy = (predictions == targets).float().mean()
     ```
   - What it means:
     - Higher is better (0-100%)
     - Should increase over time
     - More interpretable than loss
     - Can be calculated per class

5. **Validation Uncertainty (Val Uncertainty)**
   - What it is: Model's confidence on validation data
   - How it's calculated: Same as training uncertainty
   - What it means:
     - Should be higher for incorrect predictions
     - Helps identify model's weak points
     - Useful for error analysis
     - Can guide data collection

6. **Per-Epoch Metrics**
   - Train Loss Epoch: Average loss over all training batches
   - Train Uncertainty Epoch: Average uncertainty over all training batches
   - What they mean:
     - Show overall training progress
     - Help identify training issues
     - Guide hyperparameter tuning
     - Monitor model stability

#### Learning Rate Schedule Details

1. **Learning Rate Components**
   - Initial Learning Rate: 3e-5
     - Starting point for optimization
     - Small to prevent early instability
     - Calculated as max_lr/10
   
   - Maximum Learning Rate: 3e-4
     - Peak learning rate during training
     - Determines maximum step size
     - Found through learning rate finder
     - Balances speed and stability
   
   - Final Learning Rate: 3e-8
     - Learning rate at end of training
     - Very small for fine-tuning
     - Calculated as max_lr/10000
     - Helps convergence

2. **OneCycleLR Schedule Phases**
   ```python
   scheduler = OneCycleLR(
       optimizer,
       max_lr=3e-4,
       epochs=max_epochs,
       steps_per_epoch=steps_per_epoch,
       pct_start=0.3,  # 30% warmup
       div_factor=10,  # initial_lr = max_lr/10
       final_div_factor=1e4  # final_lr = max_lr/10000
   )
   ```

   a. **Warmup Phase (0-30% of training)**
      - Learning rate increases linearly
      - Helps stabilize early training
      - Prevents gradient explosion
      - Allows model to find good direction
   
   b. **Peak Phase (30-40% of training)**
      - Learning rate at maximum
      - Fast learning of major patterns
      - Helps escape local minima
      - Most aggressive updates
   
   c. **Decay Phase (40-100% of training)**
      - Learning rate decreases
      - Cosine annealing schedule
      - Fine-tunes model parameters
      - Helps convergence

3. **Schedule Parameters Explained**
   - `pct_start`: 0.3
     - Percentage of training for warmup
     - 30% of total steps
     - Balances stability and speed
   
   - `div_factor`: 10
     - Initial learning rate divisor
     - max_lr/10 = 3e-5
     - Prevents early instability
   
   - `final_div_factor`: 1e4
     - Final learning rate divisor
     - max_lr/10000 = 3e-8
     - Enables fine-tuning

4. **Why This Schedule Works**
   - Warmup prevents early instability
   - Peak phase enables fast learning
   - Decay phase enables fine-tuning
   - Cosine annealing helps convergence
   - Automatic learning rate finding
   - Reduces need for manual tuning

5. **Monitoring Learning Rate**
   - TensorBoard visualization
   - Log every N steps
   - Track with training metrics
   - Adjust based on:
     - Loss stability
     - Training speed
     - Final performance
     - Convergence time

## Neural Network Layer Details

### 1. Convolutional Layers
1. **Initial Convolution (7x7)**
   - Input: 1280x1280x3
   - Output: 640x640x64
   - Operation: \( \text{Output}(x,y) = \sum_{i=0}^{6}\sum_{j=0}^{6}\sum_{c=0}^{2} \text{Input}(x+i,y+j,c) \cdot \text{Kernel}(i,j,c) + b \)
   - Purpose: Initial feature extraction, detecting basic patterns like edges and textures
   - Stride: 2 (reduces spatial dimensions by half)

2. **Max Pooling (3x3)**
   - Input: 640x640x64
   - Output: 320x320x64
   - Operation: \( \text{Output}(x,y) = \max_{i,j \in [0,2]} \text{Input}(x+i,y+j) \)
   - Purpose: Dimensionality reduction, feature invariance to small translations
   - Stride: 2

3. **Residual Blocks**
   - Each block contains:
     - Two 3x3 convolutions
     - Batch Normalization
     - ReLU activation
     - Skip connection
   - Mathematical formulation:
     \[
     \begin{align*}
     h_1 &= \text{BN}(\text{Conv}(x)) \\
     h_2 &= \text{ReLU}(h_1) \\
     h_3 &= \text{BN}(\text{Conv}(h_2)) \\
     \text{Output} &= \text{ReLU}(h_3 + x)
     \end{align*}
     \]
   - Purpose: Enables training of very deep networks, helps with gradient flow

### 2. Feature Refinement Layers
1. **Global Average Pooling**
   - Input: 40x40x512
   - Output: 512
   - Operation: \( \text{Output}(c) = \frac{1}{40 \times 40} \sum_{i=0}^{39}\sum_{j=0}^{39} \text{Input}(i,j,c) \)
   - Purpose: Reduces spatial dimensions while preserving channel information

2. **Linear Layers with BatchNorm**
   - Input: 512
   - Output: 512
   - Operation:
     \[
     \begin{align*}
     h &= Wx + b \\
     \mu &= \frac{1}{B}\sum_{i=1}^B h_i \\
     \sigma^2 &= \frac{1}{B}\sum_{i=1}^B (h_i - \mu)^2 \\
     \hat{h} &= \gamma \frac{h - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
     \end{align*}
     \]
   - Purpose: Feature transformation and normalization

3. **Dropout**
   - Operation: \( \text{Output} = \text{mask} \odot \text{Input} \)
   - Where mask is binary with probability p of being 1
   - Purpose: Prevents overfitting by randomly deactivating neurons

### 3. Attention Mechanism
1. **Class-Specific Attention**
   - Input: 512
   - Output: 512
   - Operation:
     \[
     \begin{align*}
     h_1 &= \text{ReLU}(W_1x + b_1) \\
     h_2 &= \sigma(W_2h_1 + b_2) \\
     \text{Output} &= h_2 \odot x
     \end{align*}
     \]
   - Where:
     - \(W_1 \in \mathbb{R}^{256 \times 512}\)
     - \(W_2 \in \mathbb{R}^{512 \times 256}\)
     - \(\sigma\) is the sigmoid function
   - Purpose: Learns class-specific feature importance

### 4. Classification Head
1. **Final Linear Layer**
   - Input: 512
   - Output: 8 (number of classes)
   - Operation: \( \text{logits} = Wx + b \)
   - Where \(W \in \mathbb{R}^{8 \times 512}\)
   - Purpose: Maps features to class probabilities

2. **Softmax Activation**
   - Operation: \( P(y=i|x) = \frac{e^{z_i}}{\sum_{j=1}^8 e^{z_j}} \)
   - Where \(z_i\) is the logit for class i
   - Purpose: Converts logits to probability distribution

### 5. Uncertainty Estimation
1. **Uncertainty Head**
   - Input: 512
   - Output: 1
   - Operation:
     \[
     \begin{align*}
     h_1 &= \text{ReLU}(W_1x + b_1) \\
     h_2 &= \text{Dropout}(h_1) \\
     \text{uncertainty} &= \sigma(W_2h_2 + b_2)
     \end{align*}
     \]
   - Purpose: Estimates prediction confidence

### Layer Dimensions and Memory Usage
1. **Feature Map Sizes**
   - Input: 1280x1280x3 (4.9MB)
   - After initial conv: 640x640x64 (104.9MB)
   - After max pool: 320x320x64 (26.2MB)
   - After residual blocks: 40x40x512 (3.2MB)
   - After GAP: 512 (2KB)
   - Final output: 8 (32B)

2. **Memory Optimization**
   - Gradient checkpointing for residual blocks
   - Mixed precision training (FP16)
   - Efficient memory reuse in attention mechanism

## Training Process

### Forward Propagation (Making Predictions)

Think of this as the model's thought process:

1. **Input Layer**: Receives the astronomical image
2. **Backbone Processing**: Extracts basic features (edges, shapes)
3. **Refinement Layers**: Processes these features to understand astronomical patterns
4. **Output Layer**: Makes the final prediction about what the object is

### Backward Propagation (Learning from Mistakes)

This is how the model learns:

1. **Loss Calculation**: Measures how wrong the prediction was
2. **Gradient Computation**: Figures out how to adjust its "thinking"
3. **Weight Updates**: Makes small adjustments to improve
4. **Learning Rate Adjustment**: Changes how big these adjustments should be

### Training Duration and Early Stopping

The training process is designed to automatically stop when the model stops improving, typically around 10 epochs. This is controlled by two main mechanisms:

1. **Early Stopping Configuration**
   - Monitors validation loss (`val_loss`)
   - Patience: 10 epochs
   - Training stops if no improvement in validation loss for 10 consecutive epochs
   - Why this matters:
     - Prevents overfitting
     - Saves computational resources
     - Ensures model generalization
     - Particularly important for pre-trained models

2. **Learning Rate Schedule (OneCycleLR)**
   - Initial learning rate: 3e-5 (max_lr/10)
   - Maximum learning rate: 3e-4
   - Final learning rate: 3e-8 (max_lr/10000)
   - Schedule phases:
     1. Warmup (30% of training): Gradually increase to max_lr
     2. Peak: Maintain max_lr briefly
     3. Decay: Gradually decrease to final_lr
   - Why this schedule:
     - Faster convergence
     - Better final performance
     - Reduced need for manual tuning
     - Helps prevent local minima

3. **Validation Strategy**
   - Validation check interval: 0.25 (every 25% of training)
   - Monitors:
     - Validation loss
     - Classification accuracy
     - Per-class performance
   - Early stopping triggers if:
     - No improvement in validation loss for 10 epochs
     - Validation loss starts increasing consistently
     - Model shows signs of overfitting

4. **Why Training Stops Early**
   - Small dataset (~8,000 images)
   - Pre-trained ResNet18 backbone
   - Aggressive learning rate schedule
   - Class imbalance in the dataset
   - Early stopping prevents overfitting

5. **How to Modify Training Duration**
   - Increase early stopping patience:
     ```python
     early_stopping_patience: int = 15  # Increased from 10
     ```
   - Adjust learning rate schedule:
     ```python
     max_lr: float = 1e-4  # Reduced from 3e-4
     pct_start: float = 0.3  # Warmup duration
     ```
   - Modify validation frequency:
     ```python
     val_check_interval: float = 0.5  # Check every 50% of training
     ```

6. **Monitoring Training Progress**
   - TensorBoard logs:
     - Training/validation loss
     - Learning rate
     - Gradient norms
     - Model weights
   - Checkpoint saving:
     - Top 3 models by validation loss
     - Latest model state
     - Training metrics

7. **Training Stability Measures**
   - Gradient clipping: 0.5
   - Weight decay: 1e-4
   - Batch normalization
   - Mixed precision training
   - Gradient accumulation: 3 steps

## Project Structure

### Core Components Explained

1. **Models**
   - `astro_classifier.py`: The main brain of the system
   - `astro_classifier_module.py`: Manages how the model learns

2. **Loss Functions**
   - `focal_loss.py`: Helps the model focus on difficult examples

3. **Utilities**
   - `training_utils.py`: Helper tools for training
   - `model_utils.py`: Tools for saving and loading the model

4. **Configuration**
   - `config.py`: Settings that control how the model works

### Key Classes and Their Purposes

1. **AstroClassifierModule**
   - Manages the training process
   - Tracks how well the model is performing
   - Adjusts learning speed and other parameters

2. **FocalLoss**
   - Helps balance the model's attention
   - Gives more importance to difficult examples
   - Adjusts how the model learns from mistakes

3. **Training Utilities**
   - `print_gpu_memory`: Monitors computer resources
   - `verify_gradients`: Checks if the model is learning properly

## Configuration and Hyperparameters

### Key Hyperparameters Explained

1. **Learning Rate** (1e-4 to 3e-4)
   - What it is: How quickly the model adapts to new information
   - Why it matters: Too fast = unstable, too slow = inefficient
   - How to adjust: Start small, increase if learning is too slow

2. **Batch Size**
   - What it is: Number of images processed at once
   - Why it matters: Affects memory usage and learning stability
   - How to adjust: Based on available GPU memory

3. **Weight Decay** (1e-4)
   - What it is: Penalty for complex patterns
   - Why it matters: Prevents overfitting
   - How to adjust: Increase if model is overfitting

4. **Gradient Accumulation**
   - What it is: Accumulating learning over multiple steps
   - Why it matters: Simulates larger batch sizes
   - How to adjust: Based on available memory

## Project Goals and Limitations

### Current Goals
1. Classify astronomical objects in images
2. Handle multiple object categories
3. Provide confidence scores for predictions

### Future Goals
1. Process larger astronomical images
2. Incorporate light wavelength analysis
3. Identify more complex astronomical objects
4. Improve classification accuracy

### Current Limitations
1. **Computational Resources**
   - Limited by local GPU capabilities
   - Affects model size and complexity
   - Impacts training speed and batch size

2. **Data Constraints**
   - Limited dataset size
   - Restricted to specific object categories
   - Limited image resolution

3. **Architectural Constraints**
   - Simplified model structure
   - Basic feature extraction
   - Limited multi-task capabilities

### Future Improvements
1. **Infrastructure**
   - Cloud-based training
   - Distributed computing
   - GPU cluster utilization

2. **Model Architecture**
   - More complex backbone models
   - Advanced feature extraction
   - Multi-task learning capabilities

3. **Data Processing**
   - Higher resolution images
   - Multi-spectral analysis
   - Larger training datasets

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the model:
```bash
python config/config.py
```

3. Start training:
```bash
python train.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request


## Latest Test Results

| Metric                     | Value      |
|---------------------------|------------|
| test_accuracy              | 0.7860     |
| test_asteroid_accuracy     | 0.8849     |
| test_black_hole_accuracy   | 0.8326     |
| test_comet_accuracy        | 0.8333     |
| test_constellation_accuracy| 0.5685     |
| test_f1                    | 0.7860     |
| test_galaxy_accuracy       | 0.8644     |
| test_loss                  | 0.3778     |
| test_nebula_accuracy       | 0.8579     |
| test_planet_accuracy       | 0.9508     |
| test_precision             | 0.7860     |
| test_recall                | 0.7860     |
| test_star_accuracy         | 0.6904     |
| test_uncertainty           | 0.5539     |

## What's Next

- **New Datasets Needed:**
  - Many images in the current dataset were heavily augmented, which negatively impacted model quality and generalization.
  - The next phase should use fresh, high-quality, minimally-augmented astronomical datasets for both training and evaluation.

- **Class List Revision:**
  - The 'constellation' class should be removed. It caused clutter and confusion in classification, as images often contain many groups of stars, leading to conflicts for the classifier.
  - The model should focus on more distinct object types (e.g., 'black_hole', 'galaxy', etc.).

- **Model Retraining:**
  - The model must be retrained from scratch (or fine-tuned) on the new dataset and with the revised class list.
  - This will ensure the classifier is robust, accurate, and not biased by poor augmentations or ambiguous classes.

- **Future Improvements:**
  - Consider more advanced data augmentation strategies that preserve the physical characteristics of astronomical objects.
  - Explore more sophisticated model architectures or ensemble methods for improved accuracy.
  - Implement better post-processing to handle overlapping or ambiguous detections.

---

**Note:**
If you are using this codebase, please be aware that the current model and results are based on a dataset with significant augmentation and a class list that may not be optimal for all use cases. Retraining with a new, carefully curated dataset is highly recommended for production or research use. 

## License

This project is licensed under the MIT License - see the LICENSE file for details.