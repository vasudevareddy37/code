import matplotlib.pyplot as plt
from scipy.io import wavfile

def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 512  # Length of the windowing segments
    fs = 8000   # Sampling frequency
    pxx, freqs, bins, im = plt.specgram(data, nfft,fs)
    plt.axis('off')    
    plt.savefig('sp_xyz.png',
                dpi=100, # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0) # Spectrogram saved as a .png

def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data


if __name__ == '__main__': # Main function
    wav_file = '0_jackson_0.wav' # Filename of the wav file
    graph_spectrogram(wav_file)

