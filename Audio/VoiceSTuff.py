#!/usr/bin/env python

import wave
import struct
import numpy
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy.signal import blackmanharris, fftconvolve
from matplotlib.mlab import find
import matplotlib.pyplot as plt

def encodeIT(mySounds, Fs):

    '''
    speech = mySounds#.readframes(1600)
    numSamp = len(speech)
    # Compute auto correlation 
    rxx = []
    for k in range(256):
	rxx.append(0)
	for i in range(1,numSamp - 256):
	    rxx[k] += speech[i]*speech[i+k]
	rxx[k] = rxx[k]/float(numSamp - 256)
	

    # Specify order of model
    order = 16

    # Solve the discrete form of the Yule-Walker equations to obtain the model
    # coefficients which are stored in the array A.
    Rmat = [[0 for col in range(order)] for row in range(order)]
    Pvec = numpy.zeros((order, 1))
    for i in range(order):
	for j in range(order):
	    k = int(numpy.abs(i-j))
	    Rmat[i][j] = rxx[k]
	Pvec[i] = rxx[i]

    A = numpy.zeros((order, 1))
    A = Rmat/Pvec
    '''
    # Get if sound is present





    # Code from https://gist.github.com/endolith/255291
    # Calculate autocorrelation (same thing as convolution, but with 
    # one input reversed in time), and throw away the negative lags
    sig = mySounds
    corr = fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[len(corr)/2:]
    
    # Find the first low point
    d = numpy.diff(corr)
    start = find(d > 0)[0]
    
    # Find the next peak after the low point (other than 0 lag).  This bit is 
    # not reliable for long signals, due to the desired peak occurring between 
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = numpy.argmax(corr[start:]) + start
    f = corr
    x = peak
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    

    print Fs, xv
    FF = Fs / xv
    print FF

    sigF = numpy.abs(fft(sig))
    plt.plot(sigF[0:len(sigF)/2])
    plt.show()
    sigF = sigF[0:len(sigF)/2]
    print numpy.argmax(sigF)/float(len(sigF))*Fs/2
    

    return

def decodeIT(encodedSounds):
    # encodedSounds is a nx2 matrix where n is the number of samps,
    #   and the 2 is composed of [a, b]
    #   a is either a 0, 1, or 2. 0 is silence, 1 is unvoiced, 2 is voiced
    #   if a is 0 or 1, b = 0. if a = 2, b = pitchperiod



    return



def main():
    Fs, sig = wavfile.read("aaa.wav")
    print Fs
    #sampSize = 1600
    #mySounds = sig.readframes(sampSize)
    #print mySounds
    #wav_data1 = struct.unpack('%dh' % sampSize, mySounds)
    #sig = numpy.array(wav_data1)
    encodedSounds = encodeIT(sig, Fs)


main()
