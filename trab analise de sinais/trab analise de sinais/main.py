import os
import numpy as np
from scipy.signal import butter, lfilter, savgol_filter
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import simpleaudio as sa
import wave

# Função para carregar o áudio

def load_audio(audio_path):
    with wave.open(audio_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_samples = wav_file.getnframes()
        signal = np.frombuffer(wav_file.readframes(n_samples), dtype=np.int16).astype(np.float32) / (2**15)
    return signal, sample_rate

# Função para reproduzir o áudio

def play_signal(signal, sample_rate):
    audio = (signal * (2**15 - 1)).astype(np.int16)
    play_obj = sa.play_buffer(audio, num_channels=1, bytes_per_sample=2, sample_rate=sample_rate)
    play_obj.wait_done()

# Função para exibir o gráfico do sinal

def plot_signal(time, signal, title="Sinal"):
    plt.figure(figsize=(10, 5))
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

# Filtros básicos

def butter_filter(data, cutoff, fs, btype, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return lfilter(b, a, data)

def lowpass_filter(data, cutoff, fs):
    return butter_filter(data, cutoff, fs, btype='low')

def highpass_filter(data, cutoff, fs):
    return butter_filter(data, cutoff, fs, btype='high')

def bandpass_filter(data, lowcut, highcut, fs):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(5, [low, high], btype='band')
    return lfilter(b, a, data)

def bandstop_filter(data, lowcut, highcut, fs):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(5, [low, high], btype='bandstop')
    return lfilter(b, a, data)

# Filtros adicionais

def equalize(data, factor):
    return data * factor

def smooth(data, window_length):
    return savgol_filter(data, window_length, polyorder=2)

def add_reverb(data, decay):
    reverb = np.convolve(data, np.ones((decay,)) / decay, mode='full')[:len(data)]
    return reverb

def compress(data, threshold):
    return np.clip(data, -threshold, threshold)

# Interface gráfica
def main():
    audio_dir = os.path.join(os.getcwd(), "audio")
    audio_file = os.path.join(audio_dir, "audio.wav")

    if not os.path.exists(audio_file):
        sg.popup("Erro", "O arquivo de áudio 'audio.wav' não foi encontrado na pasta 'audio'.")
        return

    signal, sample_rate = load_audio(audio_file)
    duration = len(signal) / sample_rate
    time = np.linspace(0, duration, len(signal))

    plot_signal(time, signal, title="Sinal Original")

    layout = [
        [sg.Text("Escolha um filtro para aplicar no áudio:")],
        [sg.Button("Passa-Baixa"), sg.Button("Passa-Alta")],
        [sg.Button("Passa-Banda"), sg.Button("Rejeita-Banda")],
        [sg.Button("Equalizador"), sg.Button("Suavização")],
        [sg.Button("Reverb"), sg.Button("Compressão")],
        [sg.Button("Transformada de Fourier"), sg.Button("Sair")],
    ]

    window = sg.Window("Processamento de Áudio", layout)

    while True:
        event, _ = window.read()

        if event in (sg.WINDOW_CLOSED, "Sair"):
            break

        elif event == "Passa-Baixa":
            cutoff = 1000
            filtered_signal = lowpass_filter(signal, cutoff, sample_rate)
            plot_signal(time, filtered_signal, "Filtro Passa-Baixa")
            play_signal(filtered_signal, sample_rate)

        elif event == "Passa-Alta":
            cutoff = 1000
            filtered_signal = highpass_filter(signal, cutoff, sample_rate)
            plot_signal(time, filtered_signal, "Filtro Passa-Alta")
            play_signal(filtered_signal, sample_rate)

        elif event == "Passa-Banda":
            lowcut, highcut = 500, 2000
            filtered_signal = bandpass_filter(signal, lowcut, highcut, sample_rate)
            plot_signal(time, filtered_signal, "Filtro Passa-Banda")
            play_signal(filtered_signal, sample_rate)

        elif event == "Rejeita-Banda":
            lowcut, highcut = 500, 2000
            filtered_signal = bandstop_filter(signal, lowcut, highcut, sample_rate)
            plot_signal(time, filtered_signal, "Filtro Rejeita-Banda")
            play_signal(filtered_signal, sample_rate)

        elif event == "Equalizador":
            factor = 1.5
            filtered_signal = equalize(signal, factor)
            plot_signal(time, filtered_signal, "Equalizador")
            play_signal(filtered_signal, sample_rate)

        elif event == "Suavização":
            window_length = 101
            filtered_signal = smooth(signal, window_length)
            plot_signal(time, filtered_signal, "Suavização")
            play_signal(filtered_signal, sample_rate)

        elif event == "Reverb":
            decay = 2000
            filtered_signal = add_reverb(signal, decay)
            plot_signal(time, filtered_signal, "Reverb")
            play_signal(filtered_signal, sample_rate)

        elif event == "Compressão":
            threshold = 0.1
            filtered_signal = compress(signal, threshold)
            plot_signal(time, filtered_signal, "Compressão")
            play_signal(filtered_signal, sample_rate)

        elif event == "Transformada de Fourier":
            freq = np.fft.fftfreq(len(signal), 1 / sample_rate)
            fft_values = np.abs(fft(signal))
            plt.figure(figsize=(10, 5))
            plt.plot(freq[:len(freq)//2], fft_values[:len(freq)//2])
            plt.title("Transformada de Fourier")
            plt.xlabel("Frequência (Hz)")
            plt.ylabel("Amplitude")
            plt.grid()
            plt.show()

    window.close()

if __name__ == "__main__":
    main()
