
# Speech Enhancement

This project aims at building a speech enhancement to attenuate environmental noises.

Audios have many different ways to be represented, going from raw time series to time-frequency decompositions. The choice of the representation is crucial for the performance of your system. Among time-frequency decompositions, Spectrograms have been proved to be a useful representation for audio processing. They consist in 2D images representing sequences of Short Time Fourier Transform (STFT) with time and frequency as axes, and brightness representing the strength of a frequency component at each time frame. In such they appear a natural domain to apply the CNNS architectures for images directly to sound. Between magnitude and phase spectrograms, magnitude spectrograms contain most the structure of the signal. Phase spectrograms appear to show only little temporal and spectral regularities.


Data Collection:
--------
https://datashare.ed.ac.uk/handle/10283/2791

I have provided the link for the files we used in noisy and clean data sets

The Audio files we are using are .WAV Files.



Training:
----
The model used for the training is a U-Net, a Deep Convolutional Autoencoder with symmetric skip connections. U-Net was initially developed for Bio Medical Image Segmentation. Here the U-Net has been adapted to denoise spectrograms.

As input to the network, the magnitude spectrograms of the noisy voices. As output the Noise to model (noisy voice magnitude spectrogram - clean voice magnitude spectrogram). Both input and output matrix are scaled with a global scaling to be mapped into a distribution between -1 and 1.


![App Screenshot](https://miro.medium.com/v2/resize:fit:1400/1*VUS2cCaPB45wcHHFp_fQZQ.png)

Many configurations have been tested during the training. For the preferred configuration the encoder is made of 10 convolutional layers (with LeakyReLU, maxpooling and dropout). The decoder is a symmetric expanding path with skip connections. The last activation layer is a hyperbolic tangent (tanh) to have an output distribution between -1 and 1. For training from scratch the initial random weights where set with He normal initializer.

Model is compiled with Adam optimizer and the loss function used is the Huber loss as a compromise between the L1 and L2 loss.


...We have trained our model with U-net architecture and saved it in file named joblib_model.sav

Testing:
---------
For testing the model, provide the file path of .wav audio file in the variable named file_path by downloading the datasets provided in the link (noisy testset) and select any one of the audio file after the running the Test File , reconstructed audio file will automatically save into your current folder.

After testing the audio file we compare the waveforms with respect to the noisy files. It is seen that the wave has become normalized and most of the noise has been removed. The speech audio has also been enhanced to sound better and louder. 
<img width="953" alt="image" src="https://github.com/user-attachments/assets/371cc7c3-538a-4ef0-a2d4-1c3d69c0b5bd">

Conclusion:
---------
In conclusion, our speech enhancement program is capable of distinguishing and suppressing unwanted background noise while preserving the clarity of the spoken word. This technology not only improves the quality of audio recordings in noisy environments but also enhances the overall user experience in applications ranging from voice assistants to telecommunication systems. The model is able to cleanly remove natural noises, low-frequency noises, high-frequency noises and continuous noises.

Libraries required:
---------
- Librosa
- Matplotlib
- joblib(for saving the trained model)
- soundfile
- numpy 
- keras
- OS


