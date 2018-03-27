close all, clear all, clc;

database_folder = '/home/mainampati/speech_framework/wavdb';
imageOutDB_folder = '/home/mainampati/speech_framework/imagedb';
filePattern = fullfile(database_folder, '*.wav');
theFiles = dir(filePattern);
for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(database_folder, baseFileName);
  % fprintf(1, 'Now reading %s\n', fullFileName);
  imageOutFname = fullfile(imageOutDB_folder, baseFileName);
  imageOutFN =  imageOutFname(1:end - 4);
  [x, Fs] = audioread(fullFileName);
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
  saveas(gcf, imageOutFN, 'png' );
end

