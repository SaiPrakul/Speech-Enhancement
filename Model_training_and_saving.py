import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import soundfile as sf
import keras
from keras import models, layers
import joblib

sample_rate = 8000
n_fft = 510
hop_length = 128
max_len = 640

folder_path_noisy = "E:/College 1st year/AUDIO Dataset for IITISOC//noisy_testset_wav"
folder_path_clean = "E:/College 1st year/AUDIO Dataset for IITISOC//clean_testset_wav"


def display_audio(y):
    """ Displays the Audio file in the form of a wave. """

    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sample_rate)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


def load_features(file_path):
    """ Takes the file path of the audio wav file and The function first loads the audio file as a numpy array.
        Short-time-Fourier-transform is applied to the audio numpy array.
        The magnitude and phase are extracted using magphase func from the stft and then returned."""

    print(file_path)  # Just to check if all hte path files are checked
    audio, sr = librosa.load(file_path, sr=sample_rate)
    stft_audio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    stft_audio_mag, stft_audio_phase = librosa.magphase(stft_audio)
    stft_audio_db = librosa.amplitude_to_db(stft_audio_mag, ref=np.max)

    return stft_audio_db, stft_audio_phase


def pad_db(audio_spec_dbs, maxlen = max_len):
    """ Padding of the audio by adding 0's at the end of the array till size reaches max_len
        This is done to have an array of padded audios having same size arrays which is required for model training """

    padded_audio_dbs = []
    for spec in audio_spec_dbs:
        if spec.shape[1] < maxlen:
            pad_width = maxlen - spec.shape[1]
            spec_padded = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            spec_padded = spec[:, :maxlen]
        padded_audio_dbs.append(spec_padded)

    return np.array(padded_audio_dbs)


def unpad_db(spec_audio_db_padded):
    """ Un-pad's the audio signal. Inverse of padding where it removes all the columns having 0 values. """
    # Find the column indices that contain non-zero values
    non_zero_columns = np.any(spec_audio_db_padded != 0, axis=0)
    # Trim the spectrogram to remove columns with only zeros at the end
    last_non_zero_column = np.max(np.where(non_zero_columns))
    spec_unpadded = spec_audio_db_padded[:, :last_non_zero_column + 1]

    return spec_unpadded


def model(pretrained_weights=None, input_size=(256, 640, 1)):
    # size filter input
    size_filter_in = 16
    # normal initialization of weights
    kernel_init = 'he_normal'
    # To apply leaky relu after the conv layer
    activation_layer = None
    inputs = layers.Input(input_size)
    conv1 = layers.Conv2D(size_filter_in, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(inputs)
    conv1 = layers.LeakyReLU()(conv1)
    conv1 = layers.Conv2D(size_filter_in, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(conv1)
    conv1 = layers.LeakyReLU()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(size_filter_in * 2, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(pool1)
    conv2 = layers.LeakyReLU()(conv2)
    conv2 = layers.Conv2D(size_filter_in * 2, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(conv2)
    conv2 = layers.LeakyReLU()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(size_filter_in * 4, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(pool2)
    conv3 = layers.LeakyReLU()(conv3)
    conv3 = layers.Conv2D(size_filter_in * 4, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(conv3)
    conv3 = layers.LeakyReLU()(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(size_filter_in * 8, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(pool3)
    conv4 = layers.LeakyReLU()(conv4)
    conv4 = layers.Conv2D(size_filter_in * 8, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(conv4)
    conv4 = layers.LeakyReLU()(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = layers.Conv2D(size_filter_in * 16, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(pool4)
    conv5 = layers.LeakyReLU()(conv5)
    conv5 = layers.Conv2D(size_filter_in * 16, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(conv5)
    conv5 = layers.LeakyReLU()(conv5)
    drop5 = layers.Dropout(0.7)(conv5)

    up6 = layers.Conv2D(size_filter_in * 8, 2, activation=activation_layer, padding='same',
                        kernel_initializer=kernel_init)(layers.UpSampling2D(size=(2, 2))(drop5))
    up6 = layers.LeakyReLU()(up6)
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(size_filter_in * 8, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(merge6)
    conv6 = layers.LeakyReLU()(conv6)
    conv6 = layers.Conv2D(size_filter_in * 8, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(conv6)
    conv6 = layers.LeakyReLU()(conv6)
    up7 = layers.Conv2D(size_filter_in * 4, 2, activation=activation_layer, padding='same',
                        kernel_initializer=kernel_init)(layers.UpSampling2D(size=(2, 2))(conv6))
    up7 = layers.LeakyReLU()(up7)
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(size_filter_in * 4, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(merge7)
    conv7 = layers.LeakyReLU()(conv7)
    conv7 = layers.Conv2D(size_filter_in * 4, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(conv7)
    conv7 = layers.LeakyReLU()(conv7)
    up8 = layers.Conv2D(size_filter_in * 2, 2, activation=activation_layer, padding='same',
                        kernel_initializer=kernel_init)(layers.UpSampling2D(size=(2, 2))(conv7))
    up8 = layers.LeakyReLU()(up8)
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(size_filter_in * 2, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(merge8)
    conv8 = layers.LeakyReLU()(conv8)
    conv8 = layers.Conv2D(size_filter_in * 2, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(conv8)
    conv8 = layers.LeakyReLU()(conv8)

    up9 = layers.Conv2D(size_filter_in, 2, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(
        layers.UpSampling2D(size=(2, 2))(conv8))
    up9 = layers.LeakyReLU()(up9)
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(size_filter_in, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(merge9)
    conv9 = layers.LeakyReLU()(conv9)
    conv9 = layers.Conv2D(size_filter_in, 3, activation=activation_layer, padding='same',
                          kernel_initializer=kernel_init)(conv9)
    conv9 = layers.LeakyReLU()(conv9)
    conv9 = layers.Conv2D(2, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(conv9)
    conv9 = layers.LeakyReLU()(conv9)
    conv10 = layers.Conv2D(1, 1, activation='tanh')(conv9)

    u_net_model = models.Model(inputs, conv10)

    u_net_model.compile(optimizer='adam', loss=keras.losses.Huber(), metrics=['mae'])

    return u_net_model


def scaled_in(matrix_spec):
    """ global scaling apply to noisy voice spectrograms (scale between -1 and 1) """
    matrix_spec = (matrix_spec + 46) / 50
    return matrix_spec


def scaled_out(matrix_spec):
    """global scaling apply to noise models spectrograms (scale between -1 and 1)"""
    matrix_spec = (matrix_spec - 6) / 82
    return matrix_spec


def inv_scaled_in(matrix_spec):
    """inverse global scaling apply to noisy voices spectrograms"""
    matrix_spec = matrix_spec * 50 - 46
    return matrix_spec


def inv_scaled_out(matrix_spec):
    """inverse global scaling apply to noise models spectrograms"""
    matrix_spec = matrix_spec * 82 + 6
    return matrix_spec


def train_model(X_in_noisy, X_out_clean, epochs=20, batch_size=20, training_from_scratch=0, weights_path=None,
                name_model=None):
    """Training model using extracted features. Still pending. Will be done after model creation. """
    X_noise = X_in_noisy - X_out_clean

    X_in_noisy = scaled_in(X_in_noisy)
    X_noise = scaled_out(X_noise)

    X_in_noisy = X_in_noisy[:, :, :]
    X_in_noisy = X_in_noisy.reshape(X_in_noisy.shape[0], X_in_noisy.shape[1], X_in_noisy.shape[2], 1)
    X_noise = X_noise[:, :, :]
    X_noise = X_noise.reshape(X_noise.shape[0], X_noise.shape[1], X_noise.shape[2], 1)
    print(X_in_noisy.shape)

    if training_from_scratch == 0:

        generator_nn = model()

    generator_nn.summary()

    history = generator_nn.fit(X_in_noisy, X_noise, epochs=epochs, batch_size=batch_size, shuffle=True)
    # Plot training and validation loss (log scale)
    loss = history.history['loss']
    mae = history.history['mae']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, mae, label='Mae')
    plt.yscale('log')
    plt.title('Training and mae')
    plt.legend()
    plt.show()

    # save model with joblib
    file_name = 'joblib_model_CNN.sav'
    joblib.dump(generator_nn, file_name)


def features_to_audio_numpy(audio_db, audio_phase):
    """ Converts the features such as decibels and phase into audio numpy array. """
    stft_audio_magnitude_rev = librosa.db_to_amplitude(audio_db, ref=1.0)
    stft_audio_magnitude_rev = stft_audio_magnitude_rev * 28
    # taking magnitude and phase of audio
    audio_reverse_stft = stft_audio_magnitude_rev * audio_phase
    audio_reconstruct = librosa.core.istft(audio_reverse_stft)
    display_audio(audio_reconstruct)

    return audio_reconstruct


def reconstruct_audio_to_file(y):
    sf.write('rp232_83.wav', y, sample_rate)


# Main
# Extract features and append to arrays
audios_db_noisy = []
audios_phase_noisy = []
i = 0
num_train_files = 500
for filename in os.listdir(folder_path_noisy):
    file_path_noisy = os.path.join(folder_path_noisy, filename)
    db_audio, ph_audio = load_features(file_path_noisy)
    audios_db_noisy.append(db_audio)
    audios_phase_noisy.append(ph_audio)
    i += 1
    if i == num_train_files:
        break

audios_db_clean = []
audios_phase_clean = []
i = 0
for filename in os.listdir(folder_path_clean):
    file_path_clean = os.path.join(folder_path_clean, filename)
    db_audio, ph_audio = load_features(file_path_clean)
    audios_db_clean.append(db_audio)
    audios_phase_clean.append(ph_audio)
    i += 1
    if i == num_train_files:
        break


padded_db_noisy = pad_db(audios_db_noisy, max_len)
print(padded_db_noisy.shape)
padded_db_clean = pad_db(audios_db_clean, max_len)
print(padded_db_clean.shape)    # It is checked to be (800, 128, 621)
train_model(padded_db_noisy, padded_db_clean, 3, 20, 0)

