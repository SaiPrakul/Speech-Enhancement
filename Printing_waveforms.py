import librosa
import matplotlib.pyplot as plt

sample_rate = 8000
n_fft = 510
hop_length = 128
max_len = 640


def display_audio(audio_noisy, audio_reconstructed, audio_clean):
    """ Displays the Audio file in the form of a wave. """

    plt.figure(figsize=(12, 9))

    plt.subplot(3, 1, 1)
    librosa.display.waveshow(audio_noisy, sr=sample_rate)
    plt.title('Waveform_noisy')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    librosa.display.waveshow(audio_reconstructed, sr=sample_rate)
    plt.title('Waveform_reconstructed')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
    librosa.display.waveshow(audio_clean, sr=sample_rate)
    plt.title('Waveform_clean')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


file_1_clean = "E:\College 1st year\AUDIO Dataset for IITISOC\\clean_testset_wav\p232_075.wav"
file_1_reconstructed = "E:\College 1st year\AUDIO Dataset for IITISOC\\reconstructed_wav\\rp232_075.wav"
file_1_noisy = "E:\College 1st year\AUDIO Dataset for IITISOC\\noisy_testset_wav\p232_075.wav"

audio_clean, sr = librosa.load(file_1_clean, sr=sample_rate)
audio_reconstructed, sr = librosa.load(file_1_reconstructed, sr=sample_rate)
audio_noisy, sr = librosa.load(file_1_noisy, sr=sample_rate)

display_audio(audio_noisy, audio_reconstructed, audio_clean)

file_1_clean = "E:\College 1st year\AUDIO Dataset for IITISOC\\clean_testset_wav\p232_083.wav"
file_1_reconstructed = "E:\College 1st year\AUDIO Dataset for IITISOC\\reconstructed_wav\\rp232_83.wav"
file_1_noisy = "E:\College 1st year\AUDIO Dataset for IITISOC\\noisy_testset_wav\p232_083.wav"

audio_clean, sr = librosa.load(file_1_clean, sr=sample_rate)
audio_reconstructed, sr = librosa.load(file_1_reconstructed, sr=sample_rate)
audio_noisy, sr = librosa.load(file_1_noisy, sr=sample_rate)

display_audio(audio_noisy, audio_reconstructed, audio_clean)

