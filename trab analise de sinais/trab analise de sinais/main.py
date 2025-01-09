import os
import pywt
import numpy as np
import scipy.io.wavfile as wav
import pygame
import tkinter as tk
from tkinter import filedialog, messagebox

# Função para carregar e normalizar o áudio


def load_audio(file_path):
    rate, audio = wav.read(file_path)

    if len(audio.shape) > 1:
        audio = audio[:, 0]  # Converte para mono, se necessário

    audio = audio / np.max(np.abs(audio))  # Normaliza
    return rate, audio

# Função para aplicar a transformada de wavelet e filtrar o áudio com ajustes


def wavelet_filter(audio, wavelet='db4', threshold_factor=0.5, max_levels=12):
    max_level = pywt.dwt_max_level(len(audio), pywt.Wavelet(wavelet).dec_len)
    # Limita o número de níveis de decomposição
    max_level = min(max_level, max_levels)
    coeffs = pywt.wavedec(audio, wavelet, level=max_level)

    # Calcula o limiar de threshold (ajustado para ser mais agressivo)
    threshold = np.median(np.abs(coeffs[-5])) / threshold_factor
    filtered_coeffs = [pywt.threshold(
        c, threshold, mode='soft') for c in coeffs]

    # Reconstrução do áudio filtrado
    filtered_audio = pywt.waverec(filtered_coeffs, wavelet)

    # Garante que o áudio reconstruído esteja dentro do intervalo [-1, 1]
    return np.clip(filtered_audio, -1, 1)

# Função para salvar o áudio filtrado em arquivo


def save_audio(file_path, rate, audio):
    audio = (audio * 32767).astype(np.int16)
    wav.write(file_path, rate, audio)

# Função para tocar o áudio com pygame


def play_audio(audio, rate):
    pygame.mixer.init(frequency=rate)
    pygame.mixer.music.load(audio)
    pygame.mixer.music.play()

# Função para abrir a janela de carregamento do arquivo


def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        try:
            rate, audio = load_audio(file_path)
            audio_data['original'] = (rate, audio)  # Salva o áudio original
            messagebox.showinfo("Sucesso", "Áudio carregado com sucesso!")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar o áudio: {str(e)}")

# Função para aplicar a transformada e salvar


def apply_wavelet():
    if 'original' in audio_data:
        rate, audio = audio_data['original']
        filtered_audio = wavelet_filter(
            audio, threshold_factor=0.3, max_levels=12)  # Ajustes

        # Salvar o áudio filtrado
        save_audio('audio_filtrado.wav', rate, filtered_audio)

        # Atualizar o áudio filtrado no dicionário
        audio_data['filtered'] = (rate, filtered_audio)
        messagebox.showinfo(
            "Sucesso", "Áudio filtrado e salvo como 'audio_filtrado.wav'!")
    else:
        messagebox.showerror("Erro", "Carregue um arquivo de áudio primeiro.")

# Função para tocar o áudio original


def play_original():
    if 'original' in audio_data:
        rate, audio = audio_data['original']
        file_path = "original_audio.wav"
        save_audio(file_path, rate, audio)
        play_audio(file_path, rate)
    else:
        messagebox.showerror("Erro", "Carregue um arquivo de áudio primeiro.")

# Função para tocar o áudio filtrado


def play_filtered():
    if 'filtered' in audio_data:
        rate, audio = audio_data['filtered']
        file_path = "audio_filtrado.wav"
        save_audio(file_path, rate, audio)
        play_audio(file_path, rate)
    else:
        messagebox.showerror("Erro", "Filtre o áudio primeiro.")


# Dicionário para armazenar o áudio
audio_data = {}

# Configuração da interface gráfica
root = tk.Tk()
root.title("Filtragem de Áudio com Wavelet")
root.geometry("400x300")

# Botões da interface
btn_load = tk.Button(root, text="Carregar Áudio", command=open_file)
btn_load.pack(pady=10)

btn_apply = tk.Button(root, text="Aplicar Filtragem", command=apply_wavelet)
btn_apply.pack(pady=10)

btn_play_original = tk.Button(
    root, text="Ouvir Áudio Original", command=play_original)
btn_play_original.pack(pady=10)

btn_play_filtered = tk.Button(
    root, text="Ouvir Áudio Filtrado", command=play_filtered)
btn_play_filtered.pack(pady=10)

# Rodar a interface gráfica
root.mainloop()
