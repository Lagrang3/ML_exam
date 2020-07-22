#!/usr/bin/env python3

# Todo:
# - save status of the Network
# - show how the signal is been cleaned
# - find a better way to clean the signal
# - how many samples?
# - use multithreading for reading the files

import random
import numpy
import matplotlib.pyplot as plt
import click
import birds
import torch
import csv

### for reproducibility
torch.manual_seed(10)

@click.group()
def cli():
    pass

@cli.command('stat',help="Shows statistics about the dataset")
@click.option('-i','--info',default='./kaggle/train.csv',help="Path to info file.")
@click.option('-p','--path',default='./kaggle/train_audio',help="Path to audio files.")
def stat(info,path):
    birds.Header(info,path)()

@cli.command('build-test',help="Creates a a subset of the dataset.")
@click.option('-n','--nbirds',default=-1,help="Number of birds")
@click.option('-i','--info',default='./kaggle/train.csv',help="Path to info file.")
@click.option('-p','--path',default='./kaggle/train_audio',help="Path to audio files.")
@click.option('-o','--out',help="Output filename",required=True)
def build_test(nbirds,out,info,path):
    header=birds.Header(info,path)
    list_names=[]
    for i in range( min(nbirds,len(header.bird_dict))):
        list_names.append(header.bird_dict[i]['name'])
        
    with open(out,'w') as fout:
        writer=csv.DictWriter(fout,delimiter=',',fieldnames=header.fieldnames,extrasaction='ignore')
        writer.writeheader()
        
        for elem in header._data:
            if elem['ebird_code'] in list_names:
                writer.writerow(elem)
        
def validation(epoch,head,test_data,net,chunk_size):
    ### test
    confusion=numpy.zeros(( len(head.bird_labels),len(head.bird_labels)))
    c_error=0
    c_total=len(test_data)
    #net=net.cpu()
    
    #with click.progressbar(test_data,label="Validating") as bar:
    for x,y in test_data:
        #print("Validating on file",x['full_path'])
        x_signal = torch.tensor(birds.__get_signal__(x['full_path'],chunk_size)[1])
        y_pred = net.classify(x_signal.cuda())
        #y_pred = net.detailed_classify(x_signal.cuda())
        confusion[y_pred][y]+=1
        
        #print("Predicted:",head.bird_labels[y_pred])
        #print("True label:",head.bird_labels[y])
        
        if y_pred != y:
            c_error +=1
    with open("epoch-%d.log" % epoch,'w') as f:
        print("Error rate: %.2f%%" % ( c_error*100./c_total ),file=f)
        print("Confusion matrix:",confusion,file=f)
    my_dpi=96
    plt.figure(figsize=(1500/my_dpi,1500/my_dpi),dpi=my_dpi)
    plt.xlabel("true label")
    plt.ylabel("predicted label")
    plt.imshow(confusion)
    plt.yticks( numpy.arange(len(head.bird_labels)) ,labels = head.bird_labels  )
    plt.xticks( numpy.arange(len(head.bird_labels)) ,labels = head.bird_labels  )
    plt.colorbar()
    plt.savefig("confusion-%d.png" % epoch,dpi=my_dpi)
    #plt.show()

@cli.command('show',help="Show ")
@click.option('-p','--path',default='./kaggle/train_audio',help="Path to audio files.")
@click.option('-i','--info',default='./test/train.csv',help="Path to info file.")
@click.option('-n','--num',default=10,help="Number of chunks to show.")
def show(path,info,num):
    chunk_size=1024
    
    head = birds.Header(info,path)
    train_size = int(0.8 * len(head))
    test_size = len(head) - train_size
    train_data,test_data  = torch.utils.data.random_split(head,[train_size,test_size])
    data = birds.Dataset(train_data,chunk_size)
    
    random.shuffle(data._data)
    for i,d in enumerate(data._data): 
        if i>num:
            break
        data.show(d)
    return
   
@cli.command('init',help="Initialize a database for training and testing.")
@click.option('-p','--path',default='./kaggle/train_audio',help="Path to audio files.")
@click.option('-i','--info',default='./test/train.csv',help="Path to info file.")
def init(path,info):
    chunk_size=1024
    
    head = birds.Header(info,path)
    train_size = int(0.8 * len(head))
    test_size = len(head) - train_size
    train_data,test_data  = torch.utils.data.random_split(head,[train_size,test_size])
    
    data = birds.Dataset(train_data,chunk_size)
    print("Database read")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    net = birds.Oracle( len(data[0][0][0]), len(head.bird_labels))
    net = net.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=1e-4)
    
    dataloader = torch.utils.data.DataLoader(data,shuffle=True,batch_size=50,num_workers=4)
    print("Done loading data")
    
    nepochs = [ 2**i for i in range(5,15)  ] 
    
    with click.progressbar(range(max(nepochs)+1),label="Training") as bar:
        for epoch in bar:
            #with click.progressbar(dataloader,label="Training %d/%d" % (epoch+1,nepochs)) as bar:
            for x,y in dataloader:
                #y_pred = net(x)
                #print("y real:",y,"y_pred:",y_pred)
                optimizer.zero_grad()
                y_pred = net(x.cuda())
                loss = criterion(y_pred,y.cuda())
                loss.backward()
                optimizer.step()
            if epoch % 10 ==0:
                print('epoch:',epoch,'loss:',loss.item())
            if epoch in nepochs:
                validation(epoch,head,test_data,net,chunk_size)
                torch.save(net.state_dict(),"./net-%d.pth" % epoch)
    print("Done training")

if __name__=="__main__":
    cli()
