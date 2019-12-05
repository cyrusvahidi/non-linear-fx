import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft 
from scipy import signal
    
def get_plot(inputWave, targetWave, outputWave, fs, plotName):
    # Plot the audio signal in time
    #plt.title(plotName)
    plt.figure(1)
    timePlot = plt.subplot(211)
    FFTPlot = plt.subplot(212)
    
   

    timePlot.set_xlabel('Time')
    timePlot.set_ylabel('Amplitude')
    timePlot.set_title('Waveform Comparison')

    FFTPlot.set_xlabel('Frequency (kHz)')
    FFTPlot.set_ylabel('Power (dB)')
    FFTPlot.set_title('Power Spectrum')

    
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

    freqAxis = np.arange(0,int(np.ceil((inputLen+1)/2.0)), 1.0) * (fs / inputLen)
    timeAxis = np.linspace(0, float(inputLen/fs), num = inputLen)
   
    FFTPlot.set_xscale('symlog')
    FFTPlot.set_xlim(20,16000)
    FFTPlot.plot(freqAxis, 10*np.log10(inputFFTMag),  label = "Input") #Power spectrum
    FFTPlot.plot(freqAxis, 10*np.log10(targetFFTMag), label = "Target") #Power spectrum
    FFTPlot.plot(freqAxis, 10*np.log10(outputFFTMag), label = "Output") #Power spectrum
 
    timePlot.set_xlim(0.1,0.125)
    timePlot.plot(timeAxis, inputWave, label = "Input")
    timePlot.plot(timeAxis, targetWave, label = "Target")
    timePlot.plot(timeAxis, outputWave, label = "Output")
    plt.tight_layout()
    plt.figure(2)
    
    inputPlot = plt.subplot(311)
    targetPlot = plt.subplot(312)
    outputPlot = plt.subplot (313)
    inputPlot.set_xlabel('Time')
    inputPlot.set_ylabel('Frequency (kHz)')
    inputPlot.set_title('Input Spectrogram')

    targetPlot.set_xlabel('Time')
    targetPlot.set_ylabel('Frequency (kHz)')
    targetPlot.set_title('Target Spectrogram')

    outputPlot.set_xlabel('Time')
    outputPlot.set_ylabel('Frequency (kHz)')
    outputPlot.set_title('Output Spectrogram')

    #Spectrograms
    N = 128 #Number of point in the fft
    print(inputWave.shape[0])
    f, t, S1 = signal.spectrogram(inputWave, fs = 16000,window = signal.windows.hann(N),nfft=N, axis = 0)
    print(S1.shape)
    inputPlot.pcolormesh(t, f,10*np.log10(np.squeeze(S1))) # dB spectrogram
    
    f, t, S2 = signal.spectrogram(outputWave, fs,window = signal.windows.hann(N),nfft=N, axis = 0)
    outputPlot.pcolormesh(t, f,10*np.log10(np.squeeze(S2))) # dB spectrogram
    
    f, t, S3 = signal.spectrogram(targetWave, fs,window = signal.windows.hann(N),nfft=N, axis = 0)
    targetPlot.pcolormesh(t, f,10*np.log10(np.squeeze(S3))) # dB spectrogram
    
    plt.tight_layout()
    plt.show()
    return

def get_waveshape(inputWave, targetWave, outputWave,plotName):
    #plt.title(plotName)
    #fig = plt.figure()
    ITPlot = plt.subplot(121)
    IOPlot = plt.subplot(122)

    ITPlot.set_xlabel('Input Amplitude')
    ITPlot.set_ylabel('Target Amplitude')
    ITPlot.set_title('Input-target Waveshape')
    ITPlot.scatter(inputWave, targetWave)

    IOPlot.set_xlabel('Input Amplitude')
    IOPlot.set_ylabel('Output Amplitude')
    IOPlot.set_title('Input-output Waveshape')
    IOPlot.scatter(inputWave, outputWave)

    plt.tight_layout()
    plt.show()

    return