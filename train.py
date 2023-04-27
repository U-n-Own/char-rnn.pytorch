#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import random

from tqdm import tqdm

from helpers import *
from model import *
from generate import *


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/")

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=100)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--cuda', action='store_true')
argparser.add_argument('--music', action='store_true')
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

if not args.music:
    file, file_len = read_file(args.filename)
else:
    """ Loading all dataset from dataset/ABC_cleaned/*.abc files """
    for filename in os.listdir('dataset/ABC_cleaned/'):
        # Concatenate all files in one file
        file, file_len = read_file('dataset/ABC_cleaned/'+filename)
        file += file
        file_len += file_len 
        

def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len-1)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(inp, target):
    
    hidden = decoder.init_hidden(args.batch_size)
    
    
    if args.cuda:
        if args.model == "gru":
            hidden = hidden.cuda()
        else:
            hidden = (hidden[0].cuda(), hidden[1].cuda())
            
        # Old code
        #hidden = hidden.cuda()
        
    decoder.zero_grad()
    loss = 0
    total_loss = 0
    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    # For tensoboard: making pending events written to disk
    writer.flush()
    
    # This give error for 0-dim tensor
    #return loss.data[0] / args.chunk_len
   
    # This should work as well
    return loss.item()/args.chunk_len 
    
    # another working
    #total_loss += loss.data
    #return total_loss / args.chunk_len
    
    

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

# Initialize models and start training

decoder = CharRNN(
    n_characters,
    args.hidden_size,
    n_characters,
    model=args.model,
    n_layers=args.n_layers,
)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

# add graph to tensorboard
#writer.add_graph(decoder, (torch.zeros(args.batch_size, n_characters), decoder.init_hidden(args.batch_size)))

# Initialize some topic string we could choose for prime stri to generate text 
dict_topic = {0: 'Orso vs Runner:', 1: 'Matematica:', 2: 'Politica:',3:'Scienza:',4:'Sport:'}

if args.cuda:
    decoder.cuda()

start = time.time()
all_losses = []
loss_avg = 0

try:
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        loss = train(*random_training_set(args.chunk_len, args.batch_size))
        loss_avg += loss
        writer.add_scalar('Loss/train', loss, epoch)
        
        # Histogram and for weights and biases + gradients mode is an encoder with Embedding layer, LSTM and Liner
        writer.add_histogram('encoder weights', decoder.encoder.weight, epoch)
        #writer.add_histogram('rnn weights', decoder.rnn._parameters['weight'], epoch)
        writer.add_histogram('decoder weights', decoder.decoder.weight, epoch)
        
        writer.add_histogram('endoer gradients', decoder.encoder.weight.grad, epoch)
        #writer.add_histogram('rnn gradients', decoder.rnn._parameters['weight'].grad, epoch)
        writer.add_histogram('decoder gradients', decoder.decoder.weight.grad, epoch)
        
        if epoch % args.print_every == 0:
            
            if not args.music: 
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
                print(generate(decoder, 'Orso vs Runner:', 100, cuda=args.cuda), '\n')
            
                # For tensorboard inpecting: save generated text every print_every epochs using random topic
                topic = random.randint(0,4)
                text_to_show = generate(decoder, dict_topic[topic], 100, cuda=args.cuda)
                writer.add_text('Text', text_to_show, epoch)
                
            print("Loss: ", loss_avg/args.print_every)
            # Need to reset loss_avg after every print_every epochs
            loss_avg = 0
            

    print("Saving...")
    save()
    writer.close()
    
except KeyboardInterrupt:
    print("Saving before quit...")
    save()

