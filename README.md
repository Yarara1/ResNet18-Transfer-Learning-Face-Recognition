# ResNet-18 Transfer Learning for Face Recognition

This project investigates different **transfer learning strategies for face recognition** using an **ImageNet-pretrained ResNet-18** model.

The objective is to examine how the amount of fine-tuning affects recognition performance when transferring a pretrained convolutional neural network to a relatively small face dataset.

Five configurations were evaluated:

- Baseline — retrain only the final classifier
- Model A — fine-tune `Conv5_x`
- Model B — fine-tune `Conv4_x` and `Conv5_x`
- Model C — fine-tune all convolutional layers
- Model D — freeze the ResNet-18 backbone and train a new two-hidden-layer fully connected head

---

## Dataset

The face dataset contains:

- **100 subjects**
- **50 images per subject**
- **5,000 images in total**
- **40 training images per subject**
- **10 testing images per subject**

Therefore, the dataset is divided into:

- **4,000 training images**
- **1,000 testing images**

The assignment considers original face images of size **64 × 64 × 3** for the ResNet-18 architecture analysis.

For model training, the images were resized to **224 × 224** and converted to RGB to match the input format of the pretrained ResNet-18 implementation used in MATLAB.

---

## ResNet-18

ResNet-18 consists of 17 convolutional layers followed by a fully connected classification layer.

Its main structure is:

<img width="368" height="464" alt="image" src="https://github.com/user-attachments/assets/ff0d0781-3028-47ed-a847-b8be72943c91" />


The residual blocks use skip connections of the form:

```text
y = F(x) + x
```

These connections help preserve information and improve gradient propagation through deeper networks.

The original ImageNet classifier is replaced with a new classifier for the **100 face identities**.

---

## Architecture Analysis

For the 64 × 64 architecture analysis, the stride and padding of ResNet-18 are adjusted while preserving the convolutional filter dimensions.

Changing stride and padding changes the spatial dimensions of the feature maps, but does not change the number of convolutional weights because the kernel dimensions and channel sizes remain unchanged.

Example layers:

<img width="863" height="553" alt="image" src="https://github.com/user-attachments/assets/42928dcf-b250-46b5-9af5-4cbca6d27ba0" />


# Transfer Learning Models

## Baseline Model

The baseline model uses the pretrained ResNet-18 as a fixed feature extractor.

All convolutional layers are frozen and only the newly added final classifier is trained for the 100 face classes.

### Configuration

- Optimizer: Adam
- Learning rate: `1e-3`
- Batch size: `64`
- Epochs: `15`
- Data augmentation: None
- Trainable layers: Final classifier only

### Results

<img width="723" height="64" alt="image" src="https://github.com/user-attachments/assets/66623f05-e66d-4eb6-8d49-32577376eef5" />

The baseline shows that pretrained ImageNet features can transfer to face recognition without modifying the convolutional backbone. However, the difference between training and testing accuracy indicates that fixed ImageNet features are not sufficiently specialized for distinguishing the 100 face identities.

---

## Model A — Fine-Tune Conv5_x

Model A fine-tunes the final residual block group, `Conv5_x`, together with the new classification layer.

All earlier convolutional layers remain frozen.

```text
Frozen:
Conv1
Conv2_x
Conv3_x
Conv4_x

Trainable:
Conv5_x
Classifier
```

### Configuration

- Optimizer: Adam
- Learning rate: `1e-4`
- Batch size: `64`
- Epochs: `15`
- Data augmentation: None

### Results

<img width="717" height="74" alt="image" src="https://github.com/user-attachments/assets/03283ed8-679d-4a10-b571-79c129be3a72" />

Fine-tuning `Conv5_x` significantly improves testing accuracy compared with the baseline.

The deeper ResNet layers contain higher-level features, so allowing them to adapt helps the network learn representations that are more suitable for face recognition.

---

## Model B — Fine-Tune Conv4_x and Conv5_x

Model B fine-tunes both `Conv4_x` and `Conv5_x`, together with the new classifier.

Earlier convolutional blocks remain frozen.

```text
Frozen:
Conv1
Conv2_x
Conv3_x

Trainable:
Conv4_x
Conv5_x
Classifier
```

### Data Augmentation

Model B also introduces training-time data augmentation:

- Random rotation: ±10°
- Horizontal translation: ±5 pixels
- Vertical translation: ±5 pixels
- Random horizontal flipping

### Configuration

- Optimizer: Adam
- Learning rate: `1e-4`
- Batch size: `64`
- Epochs: `15`
- Data augmentation: Yes

### Results

<img width="725" height="69" alt="image" src="https://github.com/user-attachments/assets/d5cc729a-f087-4410-8f5c-38109e78aa66" />

Model B achieved the highest testing accuracy among the evaluated configurations.

Fine-tuning the final two residual stages allows the network to adapt higher-level features to the face-recognition task while preserving useful general features from the earlier pretrained layers.

---

## Model C — Fine-Tune All Convolutional Layers

Model C unfreezes the complete convolutional backbone.

All convolutional layers are allowed to adapt to the face dataset instead of being used as fixed feature extractors.

### Configuration

- Optimizer: Adam
- Learning rate: `1e-4`
- Batch size: `64`
- Epochs: `15`
- Data augmentation: Yes
- Trainable convolutional layers: All

### Results

<img width="727" height="72" alt="image" src="https://github.com/user-attachments/assets/fe68c74a-7c31-4a14-946d-fdf3912d8de2" />

Model C achieves very high training accuracy but performs slightly worse on the test set than Model B.

The training and testing curves also show small fluctuations during later epochs.

This suggests that fine-tuning the entire ResNet-18 increases the risk of overfitting when the target dataset is relatively small.

Earlier convolutional layers generally learn transferable low-level features such as edges and textures. Updating these layers using limited training data can reduce the benefit of the original pretrained representation.

---

## Model D — Frozen Backbone + Two Hidden FC Layers

Model D takes a different approach.

The complete ResNet-18 convolutional backbone is frozen and used only as a fixed feature extractor.

The original classification head is replaced with a new two-hidden-layer fully connected network.

<img width="592" height="637" alt="image" src="https://github.com/user-attachments/assets/3cb63e22-f1c6-42c7-b19b-6730748c6024" />


Only the newly added fully connected layers are trained.

### Configuration

- Optimizer: Adam
- Learning rate: `1e-4`
- Batch size: `64`
- Epochs: `15`
- Dropout: `0.5`
- Convolutional backbone: Frozen

### Results

<img width="813" height="76" alt="image" src="https://github.com/user-attachments/assets/a5333962-41f5-43a8-a123-75c47b31aef0" />


Model D performs poorly compared with the other approaches.

Although the classification head contains additional trainable layers, the convolutional feature extractor cannot adapt to the face-recognition task.

The experiment demonstrates that increasing classifier complexity alone does not necessarily compensate for a feature representation that is not sufficiently aligned with the target task.

---

# Results Summary
<img width="802" height="469" alt="image" src="https://github.com/user-attachments/assets/66c08c29-8fcf-4a15-8da7-30f17a2c7766" />

```

---

# Training Behavior

Training and testing accuracy and cross-entropy loss were recorded after every epoch.

For each model, the following curves were evaluated:

- Training accuracy vs. epoch
- Testing accuracy vs. epoch
- Training loss vs. epoch
- Testing loss vs. epoch

The learning curves can be stored inside the corresponding folders in the `results/` directory.

---

# Transfer Learning Analysis

The results demonstrate that the amount of fine-tuning has a significant impact on performance.

### Baseline

Training only the classifier gives a testing accuracy of **64.40%**.

The pretrained ResNet features are useful, but they are not sufficiently specialized for distinguishing individual faces.

### Model A

Fine-tuning `Conv5_x` increases testing accuracy to **84.80%**.

This shows that adapting the higher-level convolutional representations significantly improves transfer to the target task.

### Model B

Fine-tuning both `Conv4_x` and `Conv5_x`, together with data augmentation, produces the highest testing accuracy of **90.20%**.

This configuration provides a strong balance between preserving general pretrained representations and learning face-specific features.

### Model C

Fine-tuning the complete convolutional network gives **88.90%** testing accuracy.

Although the training accuracy remains almost perfect, test performance decreases slightly compared with Model B, indicating increased overfitting.

### Model D

Freezing the complete convolutional backbone and increasing only the complexity of the classifier results in very low accuracy.

This indicates that adapting the feature extractor is more important than simply increasing the number of fully connected layers.

---

# Size-Similarity Interpretation

The results are consistent with the transfer-learning **Size-Similarity** concept.

The target face dataset is relatively small, so training or aggressively modifying the complete network may lead to overfitting.

At the same time, ImageNet object classification and face recognition are not identical tasks, meaning that some adaptation of the pretrained representation is necessary.

The experiments therefore show the following behavior:

<img width="816" height="218" alt="image" src="https://github.com/user-attachments/assets/6c98fe47-843d-4c07-a78b-f4c9992942ca" />

The results show that increasing the number of trainable layers does not automatically improve generalization.

A suitable fine-tuning depth should be selected based on the size and similarity of the target dataset.

---

# Requirements

The project was implemented using **MATLAB**.

Required or recommended components:

- MATLAB
- Deep Learning Toolbox
- ResNet-18 pretrained network support package
- Image Processing Toolbox

The pretrained ResNet-18 model must be installed before running the scripts.

---

# Key Results

The main findings of the project are:

- ImageNet-pretrained ResNet-18 features can be transferred to face recognition.
- Training only the classifier produced **64.40%** testing accuracy.
- Fine-tuning the final residual block increased testing accuracy to **84.80%**.
- Fine-tuning `Conv4_x` and `Conv5_x` achieved the highest testing accuracy of **90.20%**.
- Fine-tuning the entire convolutional network slightly reduced testing accuracy to **88.90%**, indicating increased overfitting.
- Adding a larger fully connected head while keeping the complete backbone frozen was not effective.
- Selective fine-tuning provided a better balance between feature reuse and task-specific adaptation than either freezing or retraining the entire network.

---

## Technologies

`MATLAB` · `ResNet-18` · `Deep Learning` · `CNN` · `Transfer Learning` · `Face Recognition` · `Data Augmentation`
