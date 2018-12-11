import numpy as np
import sys
import matplotlib.pyplot as plt
import random
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import symbol as sym
from mxnet.gluon import nn, utils
from mxnet import autograd
from scipy.stats import norm
import matplotlib.mlab as mlab
from math import e
import math
import time


batch_size = 128
n_mixture = 8
std = 0.025
radius = 1.0
latent_z_size =100
epochs = 5000
centers = [5]
maxVal = centers[0] * 4
number_samples = 65536
sample_size = 16
use_gpu = True
ctx = mx.gpu() if use_gpu else mx.cpu()
monteCarlo = 50

lr = 0.00002
beta1 = 0.5
dropout = 0.5
data_type = "arrivals"
#data_type = "frequencies"
#data_type = "freqOfFreq"

#unroll_steps = 4

unit = []
#one = mx.nd.ones((batch_size, 1))
for i in range(sample_size):
    unitVeci = mx.nd.zeros((sample_size,1), ctx = ctx)
    unitVeci[i] = 1
    #unitColumn = one * unitVeci.T
    unit.append(unitVeci)
    

negUnit = []
for i in range(sample_size):
    negUnitVeci = mx.nd.array(np.eye(sample_size),ctx =ctx)
    negUnitVeci[i, i] = 0
    #negUnitColumn = one * negUnitVeci.T
    negUnit.append(negUnitVeci)

print (unit[0])
print(negUnit[1])

#arrivals data
if data_type == "arrivals":
    samples = []
    for i in range(number_samples):
        for c in centers:
            addList = np.random.poisson(lam = c, size = sample_size).tolist()
            list.sort(addList)
            samples.append(addList)
    #for s in range(len(samples)):
    #    samples[s] = tf.convert_to_tensor(samples[s])
    #for z in range(len(samples)):
    #   samples[z] = [samples[z]]

    #test output shape    
    #o = np.array(samples)
    #print(o.shape)
    samples = np.asarray(samples)
    print(samples.shape)
    print(samples)
    #print(samples.T[0:3])
    train_data = mx.io.NDArrayIter(data = samples, batch_size = batch_size)


netG = mx.gluon.rnn.SequentialRNNCell()
with netG.name_scope():

    netG.add(mx.gluon.rnn.RNNCell(20, activation = 'relu'))
    netG.add(mx.gluon.rnn.RNNCell(1, activation = 'relu'))
    
netD1 = nn.Sequential()
with netD1.name_scope():
    
    #Convolutional
    #input is 256 x 1 x 2
#     netD.add(nn.Conv1D(channels = 1, kernel_size = 5, padding = 2, in_channels = 1))
#     netD.add(nn.LeakyReLU(0.2))
#     # should still be 256 x 1 x 2
#     netD.add(nn.Conv1D(channels = 1, kernel_size = 5, padding = 2, in_channels = 1))
#     netD.add(nn.BatchNorm())
#     netD.add(nn.LeakyReLU(0.2))
#     # should still be 256 x 1 x 2
#     netD.add(nn.Conv1D(channels = 1, kernel_size = 5, padding = 2, in_channels = 1))
#     netD.add(nn.BatchNorm())
#     netD.add(nn.LeakyReLU(0.2))
#     # should still be 256 x 1 x 2
#     netD.add(nn.Conv1D(channels = 1, kernel_size = 5, padding = 2, in_channels = 1))
#     netD.add(nn.BatchNorm())
#     netD.add(nn.LeakyReLU(0.2))
#     # should still be 256 x 1 x 2
#     netD.add(nn.Conv1D(channels = 1, kernel_size = 5, strides = 2,padding = 2, in_channels = 1))
#     # should still be 256 x 1 x 1


    #Dense
    netD1.add(nn.Dense(200))
    netD1.add(nn.Dropout(0.5))
    netD1.add(nn.LeakyReLU(0.2))
    #netD.add(nn.Dense(100))
    #netD.add(nn.LeakyReLU(0.2))
    netD1.add(nn.Dense(200))
    netD1.add(nn.LeakyReLU(0.2))
    #netD.add(nn.Dropout(0.5))
    netD1.add(nn.Dense(1))
    
    
    
    #Try three smh
    
#     netD1.add(nn.Dense(128, activation = "tanh"))
#     netD1.add(nn.Dense(128, activation = "tanh"))
#     netD1.add(nn.Dense(1))

netD2 = nn.Sequential()
with netD2.name_scope():
    
    #Convolutional
    #input is 256 x 1 x 2
#     netD.add(nn.Conv1D(channels = 1, kernel_size = 5, padding = 2, in_channels = 1))
#     netD.add(nn.LeakyReLU(0.2))
#     # should still be 256 x 1 x 2
#     netD.add(nn.Conv1D(channels = 1, kernel_size = 5, padding = 2, in_channels = 1))
#     netD.add(nn.BatchNorm())
#     netD.add(nn.LeakyReLU(0.2))
#     # should still be 256 x 1 x 2
#     netD.add(nn.Conv1D(channels = 1, kernel_size = 5, padding = 2, in_channels = 1))
#     netD.add(nn.BatchNorm())
#     netD.add(nn.LeakyReLU(0.2))
#     # should still be 256 x 1 x 2
#     netD.add(nn.Conv1D(channels = 1, kernel_size = 5, padding = 2, in_channels = 1))
#     netD.add(nn.BatchNorm())
#     netD.add(nn.LeakyReLU(0.2))
#     # should still be 256 x 1 x 2
#     netD.add(nn.Conv1D(channels = 1, kernel_size = 5, strides = 2,padding = 2, in_channels = 1))
#     # should still be 256 x 1 x 1


    #Dense
    netD2.add(nn.Dense(256))
    netD2.add(nn.Dropout(0.5))
    netD2.add(nn.LeakyReLU(0.2))
    #netD.add(nn.Dense(100))
    #netD.add(nn.LeakyReLU(0.2))
    netD2.add(nn.Dense(200))
    netD2.add(nn.LeakyReLU(0.2))
    #netD.add(nn.Dropout(0.5))
    netD2.add(nn.Dense(1))



    #Try three smh
    #netD2.add(nn.Dense(128, activation = "tanh"))
    #netD2.add(nn.Dense(128, activation = "tanh"))
    #netD2.add(nn.Dense(1))
    
    
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

netG.initialize(mx.init.Normal(0.3), ctx = ctx, force_reinit=True)
netD1.initialize(mx.init.Normal(0.095), ctx = ctx, force_reinit=True)
netD2.initialize(mx.init.Normal(0.095), ctx = ctx, force_reinit=True)

trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1':beta1})
trainerD1 = gluon.Trainer(netD1.collect_params(), 'adam', {'learning_rate': lr, 'beta1':beta1})
trainerD2 = gluon.Trainer(netD2.collect_params(), 'adam', {'learning_rate': lr, 'beta1':beta1})
#unrolledtrainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1':beta1})


def makeOutput(netG, inputs, outputs = mx.nd.zeros((batch_size, sample_size), ctx = ctx), states = netG.begin_state(batch_size=batch_size, ctx = ctx, func = mx.ndarray.ones), iteration =0):
    #states = netG.begin_state(batch_size=batch_size, ctx = ctx, func = mx.ndarray.ones)
    #print("In make output")
    realOutput = outputs.copy()
    stateOutputs = []
    #print("--")
    #print(len(outputs))
    #print("--")
    begin = 0
    for i in range(iteration, sample_size):
        output, states = netG(inputs, states)
        realOutput = mx.nd.dot(realOutput, negUnit[i])
        realOutput = realOutput + mx.nd.dot(output.reshape(batch_size, 1), unit[i].T)
        #print("type of output IMPORTANT")
        #print(type(output.asnumpy()))
        stateOutputs.append(states)
        #print(i)
        #print(outputs)
    #print("Out of make output")
    return realOutput.floor(), stateOutputs

def calc_avgErr(fake1, fake2, fake3, fake4, netD,real_label_noise):
    fake1out, fake1states = fake1
    fake2out, fake2states = fake2
    fake3out, fake3states = fake3
    fake4out, fake4states = fake4
    #print("check stuff")
    #print(fake1out[0])
    #print(len(fake1states))
    
    
    
    
    discOut = netD(mx.ndarray.concat(fake1out, fake2out, fake3out, fake4out, dim = 0)).reshape((-1, 1))
    total_loss = loss(discOut, real_label_noise)
    #for all sample lengths
    for i in range(1, sample_size-1):
        subLoss = 0
        #print("---------------------------")
        #print(i)
        #print("------------------------------")
        for j in range(monteCarlo):
            #print(j)
            latent1 = mx.nd.random_normal(loc = 0, scale = 3, shape=(batch_size, latent_z_size, 2), ctx=ctx)
            latent2 = mx.nd.random_normal(loc = 0, scale = 3, shape=(batch_size, latent_z_size, 2), ctx=ctx)
            latent3 = mx.nd.random_normal(loc = 0, scale = 3, shape=(batch_size, latent_z_size, 2), ctx=ctx)
            latent4 = mx.nd.random_normal(loc = 0, scale = 3, shape=(batch_size, latent_z_size, 2), ctx=ctx)
            fake1i, _ = makeOutput(netG, latent1, fake1out, fake1states[i], i)
            fake2i, _ = makeOutput(netG, latent2, fake2out, fake2states[i], i)
            fake3i, _ = makeOutput(netG, latent3, fake3out, fake3states[i], i)
            fake4i, _ = makeOutput(netG, latent4, fake4out, fake4states[i], i)
            discOutput = netD(mx.ndarray.concat(fake1i, fake2i, fake3i, fake4i, dim = 0)).reshape((-1, 1))
            subLoss = subLoss + loss(discOutput, real_label_noise)
        total_loss = total_loss + subLoss/monteCarlo
    return total_loss/sample_size
    


from datetime import datetime
import time
import logging

real_label = nd.ones((batch_size * 4,), ctx = ctx)
fake_label = nd.zeros((batch_size * 4,), ctx = ctx)

def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred>0.5) == label).mean()
metric = mx.metric.CustomMetric(facc)


stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')
logging.basicConfig(level=logging.DEBUG)
print("Begin")




print("BEFORE THE FIRE:")
latent1 = mx.nd.random_normal(loc = 0, scale = 3, shape=(batch_size, latent_z_size, 2), ctx=ctx)
#fake, _ = makeOutput(netG, latent1)
for i in range(5):
    latent = mx.nd.random_normal(loc = 0, scale = 3, shape=(batch_size, latent_z_size, 2), ctx=ctx)
    fakeadd, _ = makeOutput(netG, latent)
    print("Plot %d" % i)
    #fake = mx.ndarray.concat(fake, fakeadd, dim = 0)
    data = fakeadd[i].asnumpy().tolist()
    plt.hist(data, bins=bin_count)
    plt.xticks(np.arange(min(data), max(data)+1, 1.0))

    plt.show()
    
print("Good luck bud")



"""

#set up Discriminator first
for i in range(100):
    
    tic = time.time()
    btic = time.time()
    train_data.reset()
    print("1-100")
    print(i)
    iter = 0
    #print("RUNNING")
    for batch1 in train_data:
        #print("batch")
        #print(iter)
        batch2 = next(train_data, batch1)
        batch3 = next(train_data, batch1)
        batch4 = next(train_data, batch1)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        data = (mx.ndarray.concat(batch1.data[0], batch2.data[0], batch3.data[0], batch4.data[0], dim = 0)).as_in_context(ctx)
        
        #test new shape for data
        
        
        
        #print(data)
        #if iter == 0:
            #x = data.T[0].asnumpy().tolist()
            #y = data.T[1].asnumpy().tolist()
            #print(x)
            #print(y)
            #plt.scatter(x,y)
            #plt.show()
            
        
        
        noise = mx.ndarray.random_normal(0, 0.1, batch_size * 4, ctx = ctx)
        real_label_noise = mx.ndarray.add(real_label, noise)
        noise = mx.ndarray.random_normal(0, 0.1, batch_size * 4, ctx = ctx)
        fake_label_noise = mx.ndarray.add(fake_label, noise)
        #print("TESTING 123")
        #print(real_label_noise)
        #print(fake_label_noise)
        #print("THIS IS A CHECK")
        #print(data)
        #print(len(data))
        #print(len(data[0]))
        #print(len(data[0][0]))
        #print(len(batch1.data[0]))
        #print(len(batch1.data[0][0]))
        #print(len(batch1.data[0][0][0]))
        #print(len(batch2.data[0]))
        #print(len(batch2.data[0][0]))
        #print(len(batch2.data[0][0][0]))
        #print(data)
        
        
        latent_z1 = mx.nd.random_normal(loc = 0, scale = 3, shape=(batch_size, latent_z_size, 2), ctx=ctx)
        latent_z2 = mx.nd.random_normal(loc = 0, scale = 3, shape=(batch_size, latent_z_size, 2), ctx=ctx)
        latent_z3 = mx.nd.random_normal(loc = 0, scale = 3, shape=(batch_size, latent_z_size, 2), ctx=ctx)
        latent_z4 = mx.nd.random_normal(loc = 0, scale = 3, shape=(batch_size, latent_z_size, 2), ctx=ctx)

        with autograd.record():
            # train with real image
            #print("Real Data")
            #print(data)
            output = netD1(data).reshape((-1, 1))
            #print("Output of Discriminator")
            #print(output)
            errD1_real = loss(output, real_label_noise)
            #print("This is the guess for real")
            #print(output)
            metric.update([real_label,], [output,])

            # train with fake image
            firstFake, _ = makeOutput(netG, latent_z1)
            secondFake, _ = makeOutput(netG, latent_z2)
            thirdFake, _ = makeOutput(netG, latent_z3)
            fourthFake, _ = makeOutput(netG, latent_z4)
            #print("testing 1")
            #print(firstFake)
            
            #only add if using dense
            #firstFake = firstFake.reshape((128, 1, 2))
            #secondFake = secondFake.reshape((128, 1, 2))  
            #print("testing 2")
            #print(firstFake)
            

            fake = mx.ndarray.concat(firstFake, secondFake, thirdFake, fourthFake, dim = 0)
            #print(fake)
            #print(fake)
            #print("TESTING")
            #print(len(fake))
            output = netD1(fake.detach()).reshape((-1, 1))
            errD1_fake = loss(output, fake_label_noise)
            errD1 = errD1_real + errD1_fake
            errD1.backward()
            metric.update([fake_label,], [output,])

        trainerD1.step(data.shape[0])
        
        with autograd.record():
            # train with real image
            #print("Real Data")
            #print(data)
            output = netD2(data).reshape((-1, 1))
            #print("Output of Discriminator")
            #print(output)
            errD2_real = loss(output, real_label_noise)
            #print("This is the guess for real")
            #print(output)
            metric.update([real_label,], [output,])

            # train with fake image
            firstFake, _ = makeOutput(netG, latent_z1)
            secondFake, _ = makeOutput(netG, latent_z2)
            thirdFake, _ = makeOutput(netG, latent_z3)
            fourthFake, _ = makeOutput(netG, latent_z4)
            #print("testing 1")
            #print(firstFake)
            
            #only add if using dense
            #firstFake = firstFake.reshape((128, 1, 2))
            #secondFake = secondFake.reshape((128, 1, 2))  
            #print("testing 2")
            #print(firstFake)
            

            fake = mx.ndarray.concat(firstFake, secondFake, thirdFake, fourthFake, dim = 0)
            #print("TESTING")
            #print(len(fake))
            output = netD2(fake.detach()).reshape((-1, 1))
            errD2_fake = loss(output, fake_label_noise)
            errD2 = errD2_real + errD2_fake
            errD2.backward()
            metric.update([fake_label,], [output,])

        trainerD2.step(data.shape[0])
        iter+=1
        
        
    name, acc = metric.get()
    metric.reset()
"""
print("Done setting up Discriminator")
for epoch in range(epochs+1):
    print(epoch)
    train_data.reset()
    tic = time.time()
    btic = time.time()
    count = 0
    iter = 0
    #print("RUNNING")
    for batch1 in train_data:
        #print(iter)
        batch2 = next(train_data, batch1)
        batch3 = next(train_data, batch1)
        batch4 = next(train_data, batch1)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        data = (mx.ndarray.concat(batch1.data[0], batch2.data[0], batch3.data[0], batch4.data[0], dim = 0)).as_in_context(ctx)
        #if count == 0:
            #print("Real DATA")
            #x = 
            #plt.scatter(x,y)
            #plt.show()
            
            
            #print("END of Real Data")
        
        
        noise = mx.ndarray.random_normal(0, 0.1, batch_size * 4, ctx = ctx)
        real_label_noise = mx.ndarray.add(real_label, noise)
        noise = mx.ndarray.random_normal(0, 0.1, batch_size * 4, ctx = ctx)
        fake_label_noise = mx.ndarray.add(fake_label, noise)
        #print("THIS IS A CHECK")
        #print(data)
        #print(len(data))
        #print(len(data[0]))
        #print(len(data[0][0]))
        #print(len(batch1.data[0]))
        #print(len(batch1.data[0][0]))
        #print(len(batch1.data[0][0][0]))
        #print(len(batch2.data[0]))
        #print(len(batch2.data[0][0]))
        #print(len(batch2.data[0][0][0]))
        #print(data)
        
        
        latent_z1 = mx.nd.random_normal(loc = 0, scale = 3, shape=(batch_size, latent_z_size, 2), ctx=ctx)
        latent_z2 = mx.nd.random_normal(loc = 0, scale = 3, shape=(batch_size, latent_z_size, 2), ctx=ctx)
        latent_z3 = mx.nd.random_normal(loc = 0, scale = 3, shape=(batch_size, latent_z_size, 2), ctx=ctx)
        latent_z4 = mx.nd.random_normal(loc = 0, scale = 3, shape=(batch_size, latent_z_size, 2), ctx=ctx)
        with autograd.record():
            # train with real image
            #print("Real Data")
            #print(data)
            output = netD1(data).reshape((-1, 1))
            #print("Output of Discriminator")
            #print(output)
            errD1_real = loss(output, real_label_noise)
            #print("This is the guess for real")
            #print(output)
            metric.update([real_label], [output,])

            # train with fake image
            firstFake, _ = makeOutput(netG, latent_z1)
            secondFake, _ = makeOutput(netG, latent_z2)
            thirdFake, _ = makeOutput(netG, latent_z3)
            fourthFake, _ = makeOutput(netG, latent_z4)
            #print("testing 1")
            #print(firstFake)

            #only add if using dense
            #firstFake = firstFake.reshape((128, 1, 2))
            #secondFake = secondFake.reshape((128, 1, 2))  
            #print("testing 2")
            #print(firstFake)


            fake = mx.ndarray.concat(firstFake, secondFake, thirdFake, fourthFake, dim = 0)
            #print("TESTING")
            #print(len(fake))
            output = netD1(fake.detach()).reshape((-1, 1))
            errD1_fake = loss(output, fake_label_noise)
            errD1 = errD1_real + errD1_fake
            errD1.backward()
            metric.update([fake_label,], [output,])

        trainerD1.step(data.shape[0])
        
        with autograd.record():
            # train with real image
            #print("Real Data")
            #print(data)
            output = netD2(data).reshape((-1, 1))
            #print("Output of Discriminator")
            #print(output)
            errD2_real = loss(output, real_label_noise)
            #print("This is the guess for real")
            #print(output)
            metric.update([real_label], [output,])

            # train with fake image
            firstFake, _ = makeOutput(netG, latent_z1,)
            secondFake, _ = makeOutput(netG, latent_z2)
            thirdFake, _ = makeOutput(netG, latent_z3)
            fourthFake, _ = makeOutput(netG, latent_z4)
            #print("testing 1")
            #print(firstFake)

            #only add if using dense
            #firstFake = firstFake.reshape((128, 1, 2))
            #secondFake = secondFake.reshape((128, 1, 2))  
            #print("testing 2")
            #print(firstFake)


            fake = mx.ndarray.concat(firstFake, secondFake, thirdFake, fourthFake, dim = 0)
            #print("TESTING")
            #print(len(fake))
            output = netD2(fake.detach()).reshape((-1, 1))
            errD2_fake = loss(output, fake_label_noise)
            errD2 = errD2_real + errD2_fake
            errD2.backward()
            metric.update([fake_label,], [output,])

        trainerD2.step(data.shape[0])
            
        
        #print("Generator")
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        with autograd.record():
            firstFake = makeOutput(netG, latent_z1)
            secondFake = makeOutput(netG, latent_z2)
            thirdFake = makeOutput(netG, latent_z3)
            fourthFake = makeOutput(netG, latent_z4)
            #print(fake1)
            #rint(fake1.T)
            #print(fake1.T[0])
           # print(fake1.T[0][1])
            
            
            #only add if using dense
            #fake1 = fake1.reshape((128, 1, 2))
            #fake2 = fake2.reshape((128, 1, 2))
            
            errG = calc_avgErr(firstFake, secondFake, thirdFake, fourthFake, netD1, real_label_noise)
            
            errG.backward()
        trainerG.step(mx.ndarray.concat(batch1.data[0], batch2.data[0], batch3.data[0], batch4.data[0]).shape[0], ignore_stale_grad = True)
        print("Here")    
        
        with autograd.record():
            firstFake = makeOutput(netG, latent_z1)
            secondFake = makeOutput(netG, latent_z2)
            thirdFake = makeOutput(netG, latent_z3)
            fourthFake = makeOutput(netG, latent_z4)
            #print(fake1)
            #rint(fake1.T)
            #print(fake1.T[0])
           # print(fake1.T[0][1])
            
            
            #only add if using dense
            #fake1 = fake1.reshape((128, 1, 2))
            #fake2 = fake2.reshape((128, 1, 2))
            
            errG = calc_avgErr(firstFake, secondFake, thirdFake, fourthFake, netD2, real_label_noise)
            errG.backward()

        trainerG.step(mx.ndarray.concat(batch1.data[0], batch2.data[0], batch3.data[0], batch4.data[0]).shape[0], ignore_stale_grad = True)
        
        
        

        # Print log infomation every ten batches
        if iter % 10 == 0:
            name, acc = metric.get()
            #logging.firstFake info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
            #logging.info('discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d'
            #         %(nd.mean(errD).asscalar(),
            #           nd.mean(errG).asscalar(), acc, iter, epoch))
        iter = iter + 1
        btic = time.time()

    name, acc = metric.get()
    if acc == 1.0 and epoch >201:
        print("FAIL")
        sys.exit("D too good")
    metric.reset()
    #logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
    #logging.info('time: %f' % (time.time() - tic))

    #Visualize one generated image for each epoch
    fake_img = firstFake[0]
    #print("testing")
    #print("Fake data")
    #print(fake1)
    ##print("fake data transposed")
    #print(fake1.T)
    #print(len(fake))0
    #print(len(fake[0]))
    #print(len(fake[0][0]))
    #print(fake)
    
    
    #test small print
    #print("epoch %d" % (epoch))
    #print("X: %s   Y: %s  " % (fake_img[0][0],fake_img[0][1]))
    #x= fake.T[0][0].asnumpy().tolist()
    #y = fake.T[1][0].asnumpy().tolist()
    #print("Plot")
    #plt.scatter(x,y)
    #plt.show()
    
    
    
    
    #real print
    if(epoch%100 ==0):
        print("Epoch: %d" % epoch)
    if(epoch%200 == 0):# or epoch % 200 == 1 or epoch % 200 == 2 or epoch % 200 == 3):
        logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        logging.info('time: %f' % (time.time() - tic))
        logging.info('time: %f' % (time.time() - tic))
        print("epoch %d" % (epoch))
        
        #For convolution?
        #print("X: %s   Y: %s  " % (fake_img[0][0],fake_img[0][1]))
        #x= fake1.T[0][0].asnumpy().tolist()
        #y = fake1.T[0][1].asnumpy().tolist()
        latent1 = mx.nd.random_normal(loc = 0, scale = 3, shape=(batch_size, latent_z_size, 2), ctx=ctx)
        firstFake, _ = makeOutput(netG, latent1)
        for i in range(5):
            latent = mx.nd.random_normal(loc = 0, scale = 3, shape=(batch_size, latent_z_size, 2), ctx=ctx)
            fakeadd, _ = makeOutput(netG, latent)
            
            print(fakeadd[0])
        #fake = mx.ndarray.concat(fake1, fake2, fake3, fake4, dim = 0)
        

        #print("X: ")
        #print(fake.T[0][0])
        #print("Y: ")
        #print(fake.T[0][1])
        #print("")
        #print("")


        #plt.show()   
    
    # visualize(fake_img)
    # plt.show()

