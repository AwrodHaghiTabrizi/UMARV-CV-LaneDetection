# UNet-ConvLSTM
### by Parsanna Koirala

## Using DRNN and DCNN architecture
The UNet-ConvLSTM model uses a Deep Recurrent Neural Network and a Deep Convolutional Neural Network architecture to remember what occurred in the recurrent frames and use that to predict what might come next. This can be specifically useful in lane detection, where the frames are continuous and there are many frames capturing the same area over the course of a few seconds. That can be useful for the DRNN to predict the lanes in areas where it is obstructed.

# Notes
Works decently, but needs more fine-tuning and training.
