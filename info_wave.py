from pylab import*
import scipy.io.wavfile as wav

[samfreq, snd] = wav.read('0_jackson_0.wav')
snd.dtype

snd = snd / (2.**15)

snd.shape

timeArray = arange(0, 5292, 1)
timeArray = timeArray / samfreq
timeArray = timeArray * 1000  #scale to milliseconds

plot(timeArray, samfreq, color='k')
ylabel('Amplitude')
xlabel('Time (ms)')
