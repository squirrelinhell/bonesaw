#!/usr/bin/env python3

import sys
import gym
import numpy as np
from matplotlib import pyplot as plt

#import IPython.core.ultratb
#sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

sys.path.append("..")
from mannequin import Adam, bar
from mannequin.backprop import autograd
from mannequin.distrib import Gauss
from mannequin.basicnet import Affine, LReLU, Tanh, Input, Function, Params, Const

def DKLUninormal(*, mean, logstd):
    @autograd
    def dkl(mean, logstd):
        import autograd.numpy as np
        return 0.5 * (
            np.sum(
                np.exp(logstd) - logstd + np.square(mean),
                axis=-1
            ) - mean.shape[-1]
        )

    return Function(mean, logstd, f=dkl, shape=())

def run():
    LATENT_SIZE = 3

    train_x = np.load("__mnist.npz")['train_x']

    encoder = Input(28*28)
    encoder = Tanh(Affine(encoder, 300))
    encoder = Gauss(
        mean=Affine(encoder, LATENT_SIZE),
        logstd=Params(LATENT_SIZE)
    )

    dkl = DKLUninormal(mean=encoder.mean, logstd=encoder.logstd)

    decoder_input = Input(LATENT_SIZE)
    decoder = Tanh(Affine(decoder_input, 300))
    decoder = Gauss(
        mean=Affine(decoder, 28*28),
        logstd=Const(np.zeros(28*28) - 3)
    )

    encOptimizer = Adam(encoder.get_params(), horizon=10, lr=0.01)
    decOptimizer = Adam(decoder.get_params(), horizon=10, lr=0.01)

    for i in range(10000):
        idx = np.random.choice(len(train_x), size=128)
        pics = train_x[idx]

        encoder.load_params(encOptimizer.get_value())
        decoder.load_params(decOptimizer.get_value())

        representation, encBackprop = encoder.sample.evaluate(pics)

        picsLogprob, decBackprop = decoder.logprob.evaluate(
            representation,
            sample=pics
        )

        dklValue, dklBackprop = dkl.evaluate(pics)

        decOptimizer.apply_gradient(
            decBackprop(np.ones(128))
        )

        encOptimizer.apply_gradient(
            dklBackprop(-np.ones(128))
            + encBackprop(decoder_input.last_gradient)
        )

        print("Logprob:", bar(np.mean(picsLogprob), 20000),
            "DKL:", bar(np.mean(dklValue), 200))

        if i % 100 == 99:
            plt.clf()
            fig, plots = plt.subplots(2)
            changedPic = decoder.mean(representation[43])
            plots[0].imshow(changedPic.reshape(28,28),
                cmap="gray", vmin=0, vmax=1)
            plots[1].imshow(pics[43].reshape(28,28),
                cmap="gray", vmin=0, vmax=1)
            fig.savefig("step_%05d.png"%(i+1), dpi=100)

        if i % 1000 == 999:
            np.save("step_%05d_decoder.npy"%(i+1), decoder.get_params())

if __name__ == '__main__':
    run()
