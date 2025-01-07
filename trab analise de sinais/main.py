import os
import numpy as np
import wave
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import simpleaudio as sa
from scipy.fft import fft

# Função para carregar o áudio WAV
def load_audio(audio_path):
    with wave.open(audio_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        audio_frames = wav_file.readframes(n_frames)
        signal = np.frombuffer(audio_frames, dtype=np.int16) / (2**15)  # Normaliza para -1 a 1
    return signal, sample_rate

# Função para exibir o gráfico do sinal
def plot_signal(time, signal, title="Sinal"):
    plt.figure(figsize=(10, 5))
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

# Função para reproduzir o sinal usando simpleaudio
def play_signal(signal, sample_rate):
    audio = (signal * (2**15 - 1)).astype(np.int16)  # Converte o sinal para 16 bits
    play_obj = sa.play_buffer(audio, num_channels=1, bytes_per_sample=2, sample_rate=sample_rate)
    play_obj.wait_done()

# Função para a Transformada Z
def transformada_z(signal, N=100):
    z = np.exp(2j * np.pi * np.linspace(0, 1, N))
    X_z = np.array([np.sum(signal * z_k**-np.arange(len(signal))) for z_k in z])
    plt.figure(figsize=(10, 5))
    plt.plot(z.real, z.imag, label="Círculo Unitário", linestyle="dotted")
    plt.scatter(X_z.real, X_z.imag, label="Transformada Z", color="red", s=10)
    plt.title("Transformada Z (Plano Complexo)")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.legend()
    plt.grid()
    plt.show()

# Função para a Transformada de Laplace
def transformada_laplace(signal, sample_rate, s_values):
    def laplace_integral(s):
        t = np.linspace(0, len(signal) / sample_rate, len(signal))
        return np.sum(signal * np.exp(-s * t)) * (t[1] - t[0])

    X_s = np.array([laplace_integral(s) for s in s_values])
    plt.figure(figsize=(10, 5))
    plt.plot(s_values.real, s_values.imag, label="Espaço S", linestyle="dotted")
    plt.scatter(X_s.real, X_s.imag, label="Transformada de Laplace", color="blue", s=10)
    plt.title("Transformada de Laplace (Plano Complexo)")
    plt.xlabel("Re(s)")
    plt.ylabel("Im(s)")
    plt.legend()
    plt.grid()
    plt.show()

# Interface gráfica
def main():
    audio_dir = os.path.join(os.getcwd(), "audio")
    audio_file = os.path.join(audio_dir, "audio.wav")

    if not os.path.exists(audio_file):
        sg.popup("Erro", "O arquivo de áudio 'audio.wav' não foi encontrado na pasta 'audio'.")
        return

    # Carregar o áudio
    signal, sample_rate = load_audio(audio_file)
    duration = len(signal) / sample_rate
    time = np.linspace(0, duration, len(signal))

    # Exibe o gráfico do sinal ao abrir o programa
    plot_signal(time, signal, title="Sinal do Áudio no Tempo")

    # Layout da interface
    layout = [
        [sg.Text("Escolha uma operação para aplicar no áudio:")],
        [sg.Button("Transformada de Fourier")],
        [sg.Button("Transformada Z"), sg.Button("Transformada de Laplace")],
        [sg.Button("Executar Áudio"), sg.Button("Sair")],
    ]

    window = sg.Window("Processamento de Áudio", layout)

    while True:
        event, _ = window.read()
        if event in (sg.WINDOW_CLOSED, "Sair"):
            break
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
        elif event == "Transformada Z":
            transformada_z(signal)
        elif event == "Transformada de Laplace":
            s_values = np.linspace(-10, 10, 100) + 1j * np.linspace(-10, 10, 100)
            transformada_laplace(signal, sample_rate, s_values)
        elif event == "Executar Áudio":
            play_signal(signal, sample_rate)

    window.close()

if __name__ == "__main__":
    main()
