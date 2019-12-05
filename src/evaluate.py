import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft 
from scipy import signal

import os
from meta import *
from utils import load_audio

def load_test_data(model_path, instrument, fx):
    test_df = pd.read_csv(os.path.join(model_path, 'test_data.csv'))
    n_total_clips = test_df.shape[0]
    input_target_pairs =  [0] * n_total_clips
    for idx, row in test_df.iterrows():
        audio_in = load_audio(os.path.join(params_path[instrument][NO_FX], row['input_file']), idx, n_total_clips, NO_FX)
        audio_target = load_audio(os.path.join(params_path[instrument][fx], row['target_file']), idx, n_total_clips, fx)

        input_target_pairs[idx] = (audio_in, audio_target)
        
    return input_target_pairs
    
def load_model(model_path, dropout):
    nlfx = NonLinearFXModel(params_data=params_data, params_train=params_train, dropout=dropout, dropout_rate=0.3, dnn=True)
    model = nlfx.build()
    opt = Adam(lr=params_train['lr'])
    model.compile(optimizer=opt, loss='mae', metrics=['mae'])
    model.load_weights(os.path.join(model_path, 'supervised_model.h5'))
    return model
    
def get_plot(inputWave, targetWave, outputWave, fs, plotName):
    # Plot the audio signal in time
    plt.title(plotName)
    fig = plt.figure()
    timePlot = plt.subplot(221)
    FFTPlot = plt.subplot(223)
    inputPlot = plt.subplot(322)
    targetPlot = plt.subplot(324)
    outputPlot = plt.subplot (326)

    timePlot.set_xlabel('Time')
    timePlot.set_ylabel('Amplitude')
    timePlot.set_title('Waveform Comparison')

    FFTPlot.set_xlabel('Frequency (kHz)')
    FFTPlot.set_ylabel('Power (dB)')
    FFTPlot.set_title('Power Spectrum')

    inputPlot.set_xlabel('Time')
    inputPlot.set_ylabel('Frequency (kHz)')
    inputPlot.set_title('Input Spectrogram')

    targetPlot.set_xlabel('Time')
    targetPlot.set_ylabel('Frequency (kHz)')
    targetPlot.set_title('Target Spectrogram')

    outputPlot.set_xlabel('Time')
    outputPlot.set_ylabel('Frequency (kHz)')
    outputPlot.set_title('Output Spectrogram')

    inputLen = len(inputWave)
    
    inputFFT = fft(inputWave)
    targetFFT = fft(targetWave)
    outputFFT = fft(outputWave)
    
    inputFFT = inputFFT[0:int(np.ceil((inputLen+1)/2.0))] 
    targetFFT = targetFFT[0:int(np.ceil((inputLen+1)/2.0))] 
    outputFFT = outputFFT[0:int(np.ceil((inputLen+1)/2.0))] 
    
    inputFFTMag = np.abs(inputFFT) # Magnitude
    targetFFTMag = np.abs(targetFFT) # Magnitude
    outputFFTMag = np.abs(outputFFT) # Magnitude
    
    inputFFTMag = inputFFTMag / float(inputLen)
    targetFFTMag = targetFFTMag / float(inputLen)
    outputFFTMag = outputFFTMag / float(inputLen)
    
    # power spectrum
    inputFFTMag = inputFFTMag**2
    targetFFTMag = targetFFTMag**2
    outputFFTMag = outputFFTMag**2
    
    if inputLen % 2 > 0: # fft odd 
        inputFFTMag[1:len(inputFFTMag)] = inputFFTMag[1:len(inputFFTMag)] * 2
        targetFFTMag[1:len(targetFFTMag)] = targetFFTMag[1:len(targetFFTMag)] * 2
        outputFFTMag[1:len(outputFFTMag)] = outputFFTMag[1:len(outputFFTMag)] * 2    
    else:# fft even
        inputFFTMag[1:len(inputFFTMag) -1] = inputFFTMag[1:len(inputFFTMag) - 1] * 2 
        targetFFTMag[1:len(targetFFTMag) -1] = targetFFTMag[1:len(targetFFTMag) - 1] * 2 
        outputFFTMag[1:len(outputFFTMag) -1] = outputFFTMag[1:len(outputFFTMag) - 1] * 2 

    freqAxis = np.arange(0,int(np.ceil((inputLen+1)/2.0)), 1.0) * (fs / inputLen);
    timeAxis = np.arrange(0, int(inputLen/fs))
    FFTPlot.plot(freqAxis, 10*np.log10(inputFFTMag), label="Input") #Power spectrum
    FFTPlot.plot(freqAxis, 10*np.log10(targetFFTMag), label="Target") #Power spectrum
    FFTPlot.plot(freqAxis, 10*np.log10(outputFFTMag), label="Output") #Power spectrum
 
    timePlot.plot(timeAxis,inputWave, label="Input")
    timePlot.plot(timeAxis,targetWave, label="Target")
    timePlot.plot(timeAxis,outputWave, label="Output")
    
    #Spectrograms
    N = 512 #Number of point in the fft
    f, t, S1 = signal.spectrogram(inputWave, fs,window = signal.blackman(N),nfft=N)
    inputPlot.pcolormesh(t, f,10*np.log10(S1)) # dB spectrogram
    
    f, t, S2 = signal.spectrogram(outputWave, fs,window = signal.blackman(N),nfft=N)
    outputPlot.pcolormesh(t, f,10*np.log10(S2)) # dB spectrogram
    
    f, t, S3 = signal.spectrogram(targetWave, fs,window = signal.blackman(N),nfft=N)
    targetPlot.pcolormesh(t, f,10*np.log10(S3)) # dB spectrogram
    
    plt.tight_layout()
    plt.show()
    return