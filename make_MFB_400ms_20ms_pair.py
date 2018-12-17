import librosa
import numpy as np
#import torch
from tqdm import trange
import os

# Goal: figure out 'exact' setting of lmfb used for training acoustic model
fs = 16000
nFFT_20ms = 320
shift_20ms = 160
win_size_20ms = 320
nFFT_400ms = 6400
shift_400ms = 3200
win_size_400ms = 6400



nMel = 40
win_type = 'hamming'
mel_basis_20ms = librosa.filters.mel(fs, nFFT_20ms, nMel)
mel_basis_400ms = librosa.filters.mel(fs, nFFT_400ms, nMel)

mic_audio_prefix = '/data4/kenkim/RIR-Generator/Dereverb_4IR_no_noise/train'
clean_audio_prefix = '/home/kenkim/librispeech_kaldi/LibriSpeech/train/wav'
save_400ms_prefix = '/data4/kenkim/pair_400ms_20ms/400ms'
save_20ms_prefix = '/data4/kenkim/pair_400ms_20ms/20ms'
nAudio = 10


# list .wav in mixture_audio
files = os.listdir(mic_audio_prefix)
files = [x for x in files if x.find('.wav')>0]

#for i in range(nAudio):
for i in trange(0, nAudio):
    file = files[i]
    id = file.split('_')[3][:-4]
    wav_m, _ = librosa.load(mic_audio_prefix + '/' + file, mono=False)
    wav_c, _ = librosa.load(clean_audio_prefix + '/' + id + '.wav')

    stft_m0_20ms = librosa.stft(wav_m[0], n_fft=nFFT_20ms, hop_length=shift_20ms, win_length=win_size_20ms, window=win_type)
    stft_m1_20ms = librosa.stft(wav_m[1], n_fft=nFFT_20ms, hop_length=shift_20ms, win_length=win_size_20ms, window=win_type)
    stft_m2_20ms = librosa.stft(wav_m[2], n_fft=nFFT_20ms, hop_length=shift_20ms, win_length=win_size_20ms, window=win_type)
    stft_m3_20ms = librosa.stft(wav_m[3], n_fft=nFFT_20ms, hop_length=shift_20ms, win_length=win_size_20ms, window=win_type)
    stft_m0_400ms = librosa.stft(wav_m[0], n_fft=nFFT_400ms, hop_length=shift_400ms, win_length=win_size_400ms, window=win_type)
    stft_m1_400ms = librosa.stft(wav_m[1], n_fft=nFFT_400ms, hop_length=shift_400ms, win_length=win_size_400ms, window=win_type)
    stft_m2_400ms = librosa.stft(wav_m[2], n_fft=nFFT_400ms, hop_length=shift_400ms, win_length=win_size_400ms, window=win_type)
    stft_m3_400ms = librosa.stft(wav_m[3], n_fft=nFFT_400ms, hop_length=shift_400ms, win_length=win_size_400ms, window=win_type)

    stft_c_20ms = librosa.stft(wav_c, n_fft=nFFT_20ms, hop_length=shift_20ms, win_length=win_size_20ms, window=win_type)
    stft_c_400ms = librosa.stft(wav_c, n_fft=nFFT_400ms, hop_length=shift_400ms, win_length=win_size_400ms, window=win_type)

    energy_m0_20ms = np.abs(stft_m0_20ms)
    energy_m1_20ms = np.abs(stft_m1_20ms)
    energy_m2_20ms = np.abs(stft_m2_20ms)
    energy_m3_20ms = np.abs(stft_m3_20ms)
    energy_m0_400ms = np.abs(stft_m0_400ms)
    energy_m1_400ms = np.abs(stft_m1_400ms)
    energy_m2_400ms = np.abs(stft_m2_400ms)
    energy_m3_400ms = np.abs(stft_m3_400ms)
    energy_c_20ms = np.abs(stft_c_20ms)
    energy_c_400ms = np.abs(stft_c_400ms)

    power_m0_20ms = energy_m0_20ms * energy_m0_20ms
    power_m1_20ms = energy_m1_20ms * energy_m1_20ms
    power_m2_20ms = energy_m2_20ms * energy_m2_20ms
    power_m3_20ms = energy_m3_20ms * energy_m3_20ms
    power_m0_400ms = energy_m0_400ms * energy_m0_400ms
    power_m1_400ms = energy_m1_400ms * energy_m1_400ms
    power_m2_400ms = energy_m2_400ms * energy_m2_400ms
    power_m3_400ms = energy_m3_400ms * energy_m3_400ms
    power_c_20ms = energy_c_20ms * energy_c_20ms
    power_c_400ms = energy_c_400ms * energy_c_400ms

    mfb_m0_20ms = np.expand_dims(np.dot(mel_basis_20ms, power_m0_20ms), axis=0)
    mfb_m1_20ms = np.expand_dims(np.dot(mel_basis_20ms, power_m1_20ms), axis=0)
    mfb_m2_20ms = np.expand_dims(np.dot(mel_basis_20ms, power_m2_20ms), axis=0)
    mfb_m3_20ms = np.expand_dims(np.dot(mel_basis_20ms, power_m3_20ms), axis=0)
    mfb_m0_400ms = np.expand_dims(np.dot(mel_basis_400ms, power_m0_400ms), axis=0)
    mfb_m1_400ms = np.expand_dims(np.dot(mel_basis_400ms, power_m1_400ms), axis=0)
    mfb_m2_400ms = np.expand_dims(np.dot(mel_basis_400ms, power_m2_400ms), axis=0)
    mfb_m3_400ms = np.expand_dims(np.dot(mel_basis_400ms, power_m3_400ms), axis=0)
    mfb_c_20ms = np.dot(mel_basis_20ms, power_c_20ms)
    mfb_c_400ms = np.dot(mel_basis_400ms, power_c_400ms)

    mfb_M_20ms = np.concatenate((mfb_m0_20ms, mfb_m1_20ms, mfb_m2_20ms, mfb_m3_20ms), axis=0)
    mfb_M_400ms = np.concatenate((mfb_m0_20ms, mfb_m1_20ms, mfb_m2_20ms, mfb_m3_20ms), axis=0)

    np.save(save_400ms_prefix + '/' + id + '_M_400.pth', mfb_M_400ms)
    np.save(save_400ms_prefix + '/' + id + '_S_400.pth', mfb_c_400ms)
    np.save(save_20ms_prefix + '/' + id + '_M_20.pth', mfb_M_20ms)
    np.save(save_20ms_prefix + '/' + id + '_S_20.pth', mfb_c_20ms)

