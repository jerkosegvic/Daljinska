import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

# FMCW
PATH_1 = 'data/scope_0_1.csv'
PATH_2 = 'data/scope_0_2.csv'
SAVE_PATH_FMCW = 'output_FMCW/'

# CW
PATH_3 = 'data/scope_1_1.csv'
PATH_4 = 'data/scope_1_2.csv'
SAVE_PATH_CW = 'output_CW/'

SMOOTHING = 5
PERCT = 0.01
PEAK_RESOLUTION = 500
C = 3e8
B = 720e6
FC = 24e9


def extract_data(df):
    df = df.iloc[1:]
    column_time = df.columns[0]
    column_amplitude = df.columns[1]
    new_df = pd.DataFrame()
    new_df["Time"] = pd.to_numeric(df[column_time])
    new_df["Values"] = pd.to_numeric(df[column_amplitude])
    
    return new_df
    
def load(path_1, path_2):
    scope_1 = pd.read_csv(path_1)
    scope_2 = pd.read_csv(path_2)
    scope_1 = extract_data(scope_1)
    scope_2 = extract_data(scope_2)
    return scope_1, scope_2

def plot_dfs(dfs: List[pd.DataFrame], labels: List[str], save_path: str):
    plt.figure(figsize=(14, 8))
    for i, df in enumerate(dfs):
        plt.plot(df["Time"], df["Values"], label=labels[i])
    
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")

    plt.legend()
    plt.grid()
    plt.savefig(save_path)

def plot_fft(df, freqs, ampls, save_path, marks=None):
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(df["Time"], df["Values"])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(freqs, ampls)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    if marks is not None:
        for mark in marks:
            # put a point to the place on the graph where the dominant frequency is
            plt.plot(mark, ampls[np.argmin(np.abs(freqs - mark))], 'ro')

    plt.savefig(save_path)

def find_rising_interval(df):
    df_tmp = df.copy()
    # find first bottom peak in data
    max_val = df_tmp["Values"].max()
    min_val = df_tmp["Values"].min()
    maxs, _ = find_peaks(df_tmp["Values"], distance=PEAK_RESOLUTION, height=max_val * (1.0 - PERCT))
    mins, _ = find_peaks(-df_tmp["Values"], distance=PEAK_RESOLUTION, height=-min_val * (1.0 + PERCT))
    return df_tmp["Time"].iloc[mins[0]], df_tmp["Time"].iloc[maxs[0]]

def get_spectar(df):
    N = len(df)
    T = np.mean(np.diff(df["Time"]))
    yf = np.fft.fft(df["Values"])
    xf = np.fft.fftfreq(N, T)[:N//2]
    yf = np.abs(yf[0:N//2])
    return xf, yf

def filter_signal(df):
    new_df = df.copy()
    new_df["Values"] = gaussian_filter(df["Values"], sigma=SMOOTHING)
    return new_df

def FMCW():
    scope_1, scope_2 = load(PATH_1, PATH_2)
    plot_dfs([scope_1, scope_2], ["Sent", "Recieved"], SAVE_PATH_FMCW + "scope_1_vs_scope_2.png")
    first_edge, second_edge = find_rising_interval(scope_1)
    time = second_edge - first_edge
    interval_s1 = scope_1[(scope_1["Time"] > first_edge) & (scope_1["Time"] < second_edge)]
    interval_s2 = scope_2[(scope_2["Time"] > first_edge) & (scope_2["Time"] < second_edge)]
    plot_dfs([interval_s1, interval_s2], ["Sent", "Recieved"], SAVE_PATH_FMCW + "interval_s1_vs_interval_s2.png")
    interval_s2 = filter_signal(interval_s2)
    freqs, ampls = get_spectar(interval_s2)
    dominant_freq = freqs[np.argmax(ampls)]
    plot_fft(interval_s2, freqs, ampls, SAVE_PATH_FMCW + "fft_s2.png", [dominant_freq])
    print(f"Dominant frequency: {dominant_freq} Hz")
    R = (dominant_freq * C * time) / (2 * B)
    print(f"Distance from the object: {R} m")

def CW():
    scope_1, scope_2 = load(PATH_3, PATH_4)
    plot_dfs([scope_1, scope_2], ["Sent", "Recieved"], SAVE_PATH_CW + "scope_1_vs_scope_2.png")
    time = max(scope_2["Time"]) - min(scope_2["Time"])
    scope_2 = filter_signal(scope_2)
    freqs, ampls = get_spectar(scope_2)
    dominant_freq = freqs[np.argmax(ampls)]
    plot_fft(scope_2, freqs, ampls, SAVE_PATH_CW + "fft_s2.png", [dominant_freq])
    print(f"Dominant frequency: {dominant_freq} Hz")
    v = (dominant_freq * C) / (2 * FC)
    print(f"Velocity of the object: {v} m/s")

if __name__ == "__main__":
    FMCW()
    CW()