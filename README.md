# ResNet-18 Transfer Learning for Face Recognition

This project looks at how much of an ImageNet-pretrained ResNet-18 you need to fine-tune to get good accuracy on a small face dataset. Everything is implemented in MATLAB.

Five setups are compared:

- Baseline: retrain only the final classifier
- Model A: fine-tune ##Conv5_x
- Model B: fine-tune ##Conv4_x and `Conv5_x`
- Model C: fine-tune all convolutional layers
- Model D: freeze the backbone and train a new head with two hidden fully connected layers


## Dataset

The dataset has 100 subjects with 50 images each, so 5,000 images in total. For every subject, 40 images are used for training and 10 for testing. That gives 4,000 training images and 1,000 test images.

The architecture analysis below uses the original 64 × 64 × 3 images. For training, the images are resized to 224 × 224 and converted to RGB so they match the input of the pretrained ResNet-18 in MATLAB.



## ResNet-18

ResNet-18 has 17 convolutional layers followed by one fully connected layer. The main structure looks like this:

<img width="368" height="464" alt="image" src="https://github.com/user-attachments/assets/ff0d0781-3028-47ed-a847-b8be72943c91" />

Each residual block adds its input back to its output:

```text
y = F(x) + x
```

These skip connections make it easier for gradients to flow through the network. The original ImageNet classifier is swapped for a new one with 100 outputs, one per face identity.



## Architecture analysis (64 × 64 input)

To work with 64 × 64 inputs, the stride and padding of ResNet-18 are adjusted while the filter sizes stay the same. This changes the size of the feature maps but not the number of weights, since kernel sizes and channel counts are untouched.

Example layers:

<img width="863" height="553" alt="image" src="https://github.com/user-attachments/assets/42928dcf-b250-46b5-9af5-4cbca6d27ba0" />



# Transfer learning models

All models use Adam, a batch size of 64, and 15 epochs. The baseline uses a learning rate of `1e-3`, and every other model uses `1e-4`.

## Baseline

The pretrained ResNet-18 is used as a fixed feature extractor. All convolutional layers are frozen, and only the new 100-class classifier is trained. No data augmentation.

<img width="723" height="64" alt="image" src="https://github.com/user-attachments/assets/66623f05-e66d-4eb6-8d49-32577376eef5" />

ImageNet features do carry over to faces even when the backbone isn't touched. But training accuracy is much higher than testing accuracy, so the fixed features aren't specific enough to tell 100 faces apart.



## Model A: fine-tune Conv5_x

Model A trains the last group of residual blocks (`Conv5_x`) along with the new classifier. Everything before it stays frozen. No data augmentation.

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

<img width="717" height="74" alt="image" src="https://github.com/user-attachments/assets/03283ed8-679d-4a10-b571-79c129be3a72" />

Testing accuracy jumps well above the baseline. The deepest layers hold the most high-level features, so letting them adapt gives the network a representation that fits faces better.



## Model B: fine-tune Conv4_x and Conv5_x

Model B trains `Conv4_x` and `Conv5_x` along with the classifier. The earlier blocks stay frozen.

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

This is also the first model that uses data augmentation during training:

- Random rotation of ±10°
- Horizontal translation of ±5 pixels
- Vertical translation of ±5 pixels
- Random horizontal flip

<img width="725" height="69" alt="image" src="https://github.com/user-attachments/assets/d5cc729a-f087-4410-8f5c-38109e78aa66" />

Model B has the best testing accuracy of the five. Tuning the last two stages lets the higher-level features adapt to faces, while the early layers keep the general features they learned on ImageNet.



## Model C: fine-tune all convolutional layers

Model C unfreezes the whole convolutional backbone, so every layer can adapt to the face data. It uses the same data augmentation as Model B.

<img width="727" height="72" alt="image" src="https://github.com/user-attachments/assets/fe68c74a-7c31-4a14-946d-fdf3912d8de2" />

Training accuracy is close to perfect, but testing accuracy ends up a bit below Model B, and the curves wobble in the later epochs. With a dataset this small, training the whole network seems to lead to overfitting.

The early layers learn general things like edges and textures. Updating them with only 4,000 training images can wash out some of what the pretrained weights already knew.



## Model D: frozen backbone with two hidden FC layers

Model D keeps the whole backbone frozen and replaces the original classifier with a new fully connected head that has two hidden layers. Dropout is set to 0.5. Only the new layers are trained.

<img width="592" height="637" alt="image" src="https://github.com/user-attachments/assets/3cb63e22-f1c6-42c7-b19b-6730748c6024" />

<img width="813" height="76" alt="image" src="https://github.com/user-attachments/assets/a5333962-41f5-43a8-a123-75c47b31aef0" />

This one didn't work. Training accuracy is only 1.47%, which is close to chance for 100 classes. The backbone is frozen, so the features can't adapt to faces, and a bigger head doesn't make up for that.



# Results summary

<img width="802" height="469" alt="image" src="https://github.com/user-attachments/assets/66c08c29-8fcf-4a15-8da7-30f17a2c7766" />


# Training curves

After every epoch, training and testing accuracy and cross-entropy loss were recorded. For each model there are four curves: training accuracy, testing accuracy, training loss and testing loss. These can be saved in the matching folder under `results/`.



# What the results show

How much of the network you fine-tune makes a big difference.

- **Baseline (64.40%):** Training only the classifier works to a point, but the pretrained features aren't specific enough for faces.
- **Model A (84.80%):** Fine-tuning `Conv5_x` gives a large jump, so adapting the high-level features matters.
- **Model B (90.20%):** Fine-tuning `Conv4_x` and `Conv5_x` with augmentation gives the best result. It keeps the general pretrained features and still learns face-specific ones.
- **Model C (88.90%):** Fine-tuning everything is slightly worse than Model B even though training accuracy is almost 100%, which points to overfitting.
- **Model D (1.70%):** Making the classifier bigger while the backbone stays frozen doesn't help. Adapting the feature extractor mattered more than adding FC layers.



# Size and similarity

The results line up with the usual size-similarity rule of thumb for transfer learning. The face dataset is small, so retraining the whole network risks overfitting. But ImageNet objects and faces aren't very similar either, so the pretrained features need some adjusting.

<img width="816" height="218" alt="image" src="https://github.com/user-attachments/assets/6c98fe47-843d-4c07-a78b-f4c9992942ca" />

More trainable layers doesn't automatically mean better generalization. How deep to fine-tune depends on how big and how similar the target dataset is.


# Requirements

The project was written in MATLAB. You'll need:

- MATLAB
- Deep Learning Toolbox
- Deep Learning Toolbox Model for ResNet-18 Network (the pretrained model has to be installed before running the scripts)
- Image Processing Toolbox
