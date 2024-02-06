# Anomaly-Detection-ECG
In this project we use an autoencoder architecture to classify abnormal heartbeat in the time series Electriccardiogram(ECG) test. Our dataset used is ECG5000 dataset, it consists of 5000 ECG examples with 140 timesteps each. Each heartbeat corresponds to a single patient with congestive heart failure. 

In the dataset we have 5 Types of heartbeats:
1. Normal
2. Premature Ventricular Contraction (PVC)
3. R-on-T Premature Ventricular Contraction (R-on-T PVC)
4. Supra-ventricular Premature or Ectopic Beat (SP or EB)
5. Unclassified Beat (UB)

We use LSTM Autoencoder to detect anomaly in the heartbeats, an encoder-decoder LSTM is configured to read the input sequence, encode it, decode it, and recreate it. The performance of the model is evaluated based on the model’s ability to recreate the input sequence. To train the autoencoder we use only normal heartbeat data, so that it can learn to reconstruct normal heartbeats well.

To classify a sequence as normal or an anomaly, we pick a threshold above which a heartbeat is considered abnormal.

Reconstruction Loss: When training an Autoencoder, the objective is to reconstruct the input as best as possible. This is done by minimizing a loss function (just like in supervised learning). This function is known as reconstruction loss. Common examples are Cross-entropy loss and Mean squared error.

If the reconstruction loss for an example is below the threshold, we'll classify it as a normal heartbeat Alternatively, if the loss is higher than the threshold, we'll classify it as an anomaly Normal hearbeats.


### Dependencies Used:
```
1. Seaborn
2. Matplotlib
3. Numpy
4. Pandas
5. scipy
6. sklearn
7. glob
8. torch
9. torchvision
```
