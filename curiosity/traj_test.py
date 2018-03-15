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

def save_plot(file_name, *traj_types):
    import matplotlib.pyplot as plt
    plt.clf()
    for i, (trajs, col) in enumerate(zip(traj_types, "rgbcmy")):
        for t in trajs:
            t = t.reshape((-1, 2))
            plt.plot(
                t[:,0], t[:,1],
                color=col, alpha=0.2, linewidth=2, zorder=i
            )
    plt.gca().set_xlim([-10, 10])
    plt.gca().set_ylim([-5, 5])
    plt.gcf().set_size_inches(16, 8)
    plt.gcf().savefig(file_name, dpi=100)

def gen_traj():
    x = np.random.randn(2) + [5, 1]
    dx = (np.random.randn(2) + [1, 1]) * 0.05
    out = []
    for _ in range(25):
        dx += [0, -0.005]
        x += dx
        x *= 0.99
        out.append(np.array(x))
    return np.asarray(out).reshape(-1)

def run():
    LATENT_SIZE = 2

    encoder = Input(gen_traj().size)
    encoder = Tanh(Affine(encoder, 300))
    encoder = Affine(encoder, LATENT_SIZE), Params(LATENT_SIZE)

    dkl = DKLUninormal(mean=encoder[0], logstd=encoder[1])
    encoder = Gauss(mean=encoder[0], logstd=encoder[1])

    decoder_input = Input(LATENT_SIZE)
    decoder = Tanh(Affine(decoder_input, 300))
    decoder = Affine(decoder, gen_traj().size)
    mean_dec = decoder
    decoder = Gauss(mean=decoder)

    encOptimizer = Adam(encoder.get_params(), horizon=10, lr=0.01)
    decOptimizer = Adam(decoder.get_params(), horizon=10, lr=0.01)

    for i in range(10000):
        inps = [gen_traj() for i in range(128)]

        encoder.load_params(encOptimizer.get_value())
        decoder.load_params(decOptimizer.get_value())

        representation, encBackprop = encoder.sample.evaluate(inps)

        inpsLogprob, decBackprop = decoder.logprob.evaluate(
            representation,
            sample=inps
        )

        dklValue, dklBackprop = dkl.evaluate(inps)

        decOptimizer.apply_gradient(
            decBackprop(np.ones(128))
        )

        encOptimizer.apply_gradient(
            dklBackprop(-np.ones(128))
            + encBackprop(decoder_input.last_gradient)
        )
        print(decoder_input.last_gradient)

        print("Logprob:", bar(np.mean(inpsLogprob), 20000),
            "DKL:", bar(np.mean(dklValue), 200))

        if i % 25 == 24:
            save_plot(
                "test_%05d.png" % (i+1),
                mean_dec(np.random.randn(64, LATENT_SIZE)),
                inps,
                mean_dec(representation),
            )

if __name__ == '__main__':
    run()
