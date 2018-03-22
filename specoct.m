##speech Spectrogram


#[x, Fs] = audio(file_in_loadpath("sample.wav")); # audio file
#close all, 
clc, clear all;
[x, Fs] = audioread('03a01Fa.wav');
%soundsc(x, Fs)
step = fix(5*Fs/1000);     # one spectral slice every 5 ms
window = fix(20*Fs/1000);  # 40 ms data window
    
fftn = 2^nextpow2(window); # next highest power of 2
[S, f, t] = specgram(x, fftn, Fs, window, window-step);
S = abs(S(2:fftn*4000/Fs,:)); # magnitude in range 0<f<=4000 Hz.
S = S/max(S(:));           # normalize magnitude so that max is 0 dB.
S = max(S, 10^(-40/10));   # clip below -40 dB.
S = min(S, 10^(-3/10));    # clip above -3 dB.
im = imagesc(t, f, log(S));    # display in log scale
axis off;
colormap(gray)
set (gca, "ydir", "normal");
set(gca,'Position',[0 0 1 1]); % Make the axes occupy the hole figure
saveas(gcf, '03a01Fa', 'png' );









#colormap(gray)
%set(gca,'XTick',[]) % Remove the ticks in the x axis!
%set(gca,'YTick',[]) % Remove the ticks in the y axis
%set(gca,'Position',[0 0 1 1]) % Make the axes occupy the hole figure
%saveas(gcf,'0_jackson_0','png')

#{
close all,clear all, clc;
[y,fs,bits] = wavread('0_jackson_0.wav');
soundsc(y,fs); % Let's hear it
% for classic look:
colormap('gray'); map = colormap; imap = flipud(map);
M = round(0.04*fs);  % 20 ms window is typical
step = fix(5*fs/1000); 
N = 2^nextpow2(4*M); % zero padding for interpolation
w = hamming(M);
specgram(y,N,fs,w,w-step);
title('Speech Sample Spectrogram');
colormap(imap);
#}