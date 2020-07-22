import matplotlib.pyplot as plt
#import torchaudio

from .constants import *

def plot_waveform(fname):
    '''
    Plot the waveform of an audiofile
    '''
    pass
    waveform,sample_rate = torchaudio.load(fname)
    if sample_rate!=REDBOOK_FREQ:
        waveform = torchaudio.transforms.Resample(sample_rate,REDBOOK_FREQ)(waveform)
    plt.figure()
    plt.plot(waveform.t().numpy())
    plt.show()
