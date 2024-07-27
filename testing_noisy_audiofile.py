import joblib
import soundfile as sf
import numpy as np
import librosa
import matplotlib.pyplot as plt

sample_rate = 8000
n_fft = 510
hop_length = 128
max_len = 640  # Taken with respect to the training audio files


def display_audio(audio_waveform):
    """ Displays the Audio file in the form of a wave. """

    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio_waveform, sr=sample_rate)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


def scaled_in(matrix_spec):
    """ global scaling apply to noisy voice spectrogram (scale between -1 and 1) """
    matrix_spec = (matrix_spec + 46) / 50
    return matrix_spec


def scaled_ou(matrix_spec):
    """global scaling apply to noise models spectrogram (scale between -1 and 1)"""
    matrix_spec = (matrix_spec - 6) / 82
    return matrix_spec


def inv_scaled_in(matrix_spec):
    """inverse global scaling apply to noisy voices spectrogram"""
    matrix_spec = matrix_spec * 50 - 46
    return matrix_spec


def inv_scaled_ou(matrix_spec):
    """inverse global scaling apply to noise models spectrogram"""
    matrix_spec = matrix_spec * 82 + 6
    return matrix_spec



def features_to_audio_numpy(audio_db, audio_phase):
    """
    Converts the decibels into amplitude and modified. Then multiplied with the phase to get the stft.
    This stft(short time fourier transform) is inverse to construct the audio.
    :param audio_db: Predicted audio by model in decibels
    :param audio_phase: Extracted phase feature from the input audio wav file
    :return: Returns the reconstructed audio wave
    """
    stft_audio_magnitude_rev = librosa.db_to_amplitude(audio_db, ref=1.0)
    stft_audio_magnitude_rev = stft_audio_magnitude_rev * 42
    audio_reverse_stft = stft_audio_magnitude_rev * audio_phase
    audio_reconstruct = librosa.core.istft(audio_reverse_stft)
    return audio_reconstruct


def reconstruct_audio_to_file(y):
    """ Reconstruction of audio as wav file. Playable """
    sf.write('new.wav', y, sample_rate)


def for_output_using_model(loaded_model, stft_audio_db):
    """ Running the model with audio file for required clean output """
    X_in_noisy = scaled_in(stft_audio_db)
    print(X_in_noisy.shape)
    X_in_noisy_reshaped = X_in_noisy.reshape(1, X_in_noisy.shape[0], X_in_noisy.shape[1], 1)
    print(X_in_noisy_reshaped.shape)

    y_noise_only_predict = loaded_model.predict(X_in_noisy_reshaped)

    y_predict = inv_scaled_ou(y_noise_only_predict)
    y_predict = y_predict.reshape(y_predict.shape[1], y_predict.shape[2])
    y_predict = y_predict[:, :stft_audio_phase.shape[1]]

    return y_predict


filename = "joblib_model.sav"
# load model with joblib
loaded_model = joblib.load(filename)
file_path = "E:\College 1st year\AUDIO Dataset for IITISOC\Recording.wav"
audio, sr = librosa.load(file_path, sr=sample_rate)

stft_audio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
stft_audio_mag, stft_audio_phase = librosa.magphase(stft_audio)
db_in = librosa.amplitude_to_db(stft_audio_mag, ref=np.max)
db_in_len = db_in.shape[1]
rec = 1;
itera = int(db_in_len/max_len) + 1

original_height, original_width = db_in.shape
y_predict = np.zeros((original_height, original_width))

while rec < itera:
    spec = db_in[:, 640*(rec-1):640*rec]
    y_predict[:, 640*(rec-1):640*rec] = for_output_using_model(loaded_model, spec)
    rec += 1;


pad_width = (640*itera) - db_in_len
stft_audio_db = np.pad(db_in[:, 640*(itera-1):db_in_len], ((0, 0), (0, pad_width)), mode='constant')
y_predict[:, 640*(itera-1):db_in_len] = for_output_using_model(loaded_model, stft_audio_db)[:, :db_in_len-640*(itera-1)]

clean_speech_predicted = db_in - y_predict
print(clean_speech_predicted.shape)

clean_speech_audio_predicted = features_to_audio_numpy(clean_speech_predicted, stft_audio_phase)
clean_speech_audio_predicted = librosa.effects.preemphasis(clean_speech_audio_predicted, coef=0.97)
clean_speech_audio_predicted *= 3
clean_speech_audio_predicted = librosa.util.normalize(clean_speech_audio_predicted)
display_audio(clean_speech_audio_predicted)
reconstruct_audio_to_file(clean_speech_audio_predicted)
