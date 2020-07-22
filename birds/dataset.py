import csv,pathlib
import torch
import numpy
import random,copy
import click
import datetime
from pydub import AudioSegment
import scipy.signal

from .exception import Exception
from .plot import *
from .constants import *
import torch

def compute_rms(y):
    return numpy.sqrt(numpy.mean(y**2))

def __get_waveform__(fname,chunk_size):
    istream = AudioSegment.from_mp3(fname)
    istream.set_frame_rate(REDBOOK_FREQ)
    istream_split = istream.split_to_mono()
    waveform = numpy.array(istream_split[0].get_array_of_samples()).astype(float)
    waveform /= numpy.max(waveform)
    return  waveform
    #waveform,sample_rate = torchaudio.load(fname )
    #if sample_rate!=REDBOOK_FREQ:
    #    waveform = torchaudio.transforms.Resample(sample_rate,REDBOOK_FREQ)(waveform)
    #return waveform[0].t().numpy()

def __get_signal__(fname,chunk_size): 
    x = __get_waveform__(fname,chunk_size)
    rms_max =0 
    for i in range(0,len(x),chunk_size):
        if i+chunk_size > len(x):
            break
        rms = compute_rms(x[i:chunk_size+i])
        rms_max = max(rms,rms_max)
    rms_max *= 0.5
    spek_list=list()
    time_list=list()
    for i in range(0,len(x),chunk_size):
        if i+chunk_size > len(x):
            break
        y_time = copy.deepcopy(x[i:chunk_size+i])
        rms = compute_rms( y_time )
        if rms >= rms_max:
            y_max = numpy.max(numpy.abs(y_time))
            y_time /= y_max
            y_spek = numpy.fft.fft( y_time )       
            y_spek = y_spek[:len(y_spek)//2]
            y_spek = y_spek[  int(chunk_size * CUT_FREQ/REDBOOK_FREQ ) :]
            time_list.append( y_time )
            spek_list.append(
                    numpy.float32( abs( y_spek * y_spek.conjugate())) )
    #spek_list=torch.tensor(spek_list)
    #time_list=torch.tensor(time_list)
    return  time_list,spek_list

class Header():
    '''
        Helper class to read the csv file 
        and extract some statistics.
    '''
    def __init__(self,info,path):
        self.path=path
        bird_labels = set()
        self.sample_rate = dict()
        self._data=list()
        self.fieldnames=list()
        
        
        with open(info,'r') as f:
            reader = csv.DictReader( f ,delimiter=',') 
            self.fieldnames = copy.deepcopy(reader.fieldnames)
            for row in reader:
                sr = int(row['sampling_rate'].split()[0])
                if sr not in self.sample_rate:
                    self.sample_rate[sr] = 0
                self.sample_rate[sr] +=1
                bird_labels.add(row['ebird_code'])
                if sr >= REDBOOK_FREQ: # we discard audio files with lower frequencies
                    self._data.append(row)
        
        
        self.bird_labels = sorted(list(bird_labels))
        
        self.bird_dict = dict()
        for label in bird_labels:
            self.bird_dict[label]={ 'total': 0, '5.0': 0, '4.5' : 0, '4.0' : 0 }
            
        for elem in self._data:    
            myfile = pathlib.Path(path+'/'+elem['ebird_code']+'/'+elem['filename'])
            elem['full_path'] = str(myfile) 
            elem['ebird_number'] = self.bird_labels.index(elem['ebird_code'])
            this_bird = self.bird_dict[ elem['ebird_code'] ]
            if elem['rating'] in this_bird:
                this_bird['total']+=1
                this_bird[elem['rating']]+=1
        tmp=list()
        for key,value in self.bird_dict.items():
            value['name']=key
            tmp.append(value)
        self.bird_dict=tmp
        self.bird_dict = sorted(self.bird_dict,
            key=lambda x: ( -x['5.0'],-x['4.5'],-x['4.0']  ))
        
        
    def __call__(self): 
        print("Number of birds:",len(self.bird_labels),'\n')
        print("Bird list:",self.bird_labels)
        #random.shuffle(self.bird_labels)
        
        row = "{:<5} {:<10} {:<5} {:<5} {:<5} {:<5}"
        print( row.format("#","name" ,"total","5.0","4.5","4.0"))
        for i,label in enumerate(self.bird_dict):
            print( row.format(i,label['name'],label['total'],label['5.0'],label['4.5'],label['4.0']))
        print("\n")
            
        row = "{:<20} {:<10}"
        
        print("Audio files")
        print( row.format("sample_rate (kHz)","counts"))
        for sr in sorted( self.sample_rate ):    
            print( row.format(sr,self.sample_rate[sr]))
            
        duration=0
        for elem in self._data:
            duration += int(elem['duration'])
        print("\nduration:",datetime.timedelta(seconds=duration))
    
    def __iter__(self):
        return self._data.__iter__()
    
    def __len__(self):
        return len(self._data)
    
    #def get_data(self,idx,chunk_size):  
    #    x = __get_signal__(self._data[idx]['full_path'],chunk_size)[1]
    #    return x,self._data[idx]['ebird_number']
    def __getitem__(self,idx):
        return self._data[idx],self._data[idx]['ebird_number']

class Dataset(torch.utils.data.Dataset):
    def __init__(self,info,chunk_size):
        self.chunk_size = chunk_size
        self._data= list()
        
        with click.progressbar(info,label="Reading audio files") as bar:
            for elem in bar:    
                #try:
                    for time,spek in zip(*__get_signal__(elem[0]['full_path'],self.chunk_size)):
                        d = copy.deepcopy(elem[0])
                        d['input'] = torch.tensor(spek).reshape(1,-1).clone().detach() 
                        d['signal']= torch.tensor(time).reshape(1,-1).clone().detach()
                        self._data.append(d)
                #except:
                #    continue
                #bar.update(i)
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self,idx):
        return self._data[idx]['input'],self._data[idx]['ebird_number']
    
    def show(self,d):
        plt.figure()
        plt.plot( d['input'][0] )
        plt.title( "Spectrum " + d['full_path'])
        plt.show()
        plt.figure()
        plt.plot( d['signal'][0] )
        plt.title("Time series "+d['full_path'])
        plt.show()
