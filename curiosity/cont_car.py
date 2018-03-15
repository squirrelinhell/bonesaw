#!/usr/bin/env python3

import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from mannequin import *
from mannequin.basicnet import *
from mannequin.gym import *
from mannequin.backprop import autograd
from mannequin.distrib import Gauss

GEN_SEGM_LEN = 25

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

def Split(inner):
    secondEvaluate = None
    class First(Layer):
        def __init__(self):

            def evaluate(inps):
                nonlocal secondEvaluate
                firstResult = None
                firstBackprop = None
                firstGrad = None
                def backprop(grad, output=None):
                    nonlocal firstGrad
                    assert(firstGrad is None)
                    firstGrad = grad
                    return []

                firstResult, firstBackprop = inner.evaluate(inps)
                def newEvaluate(inps):
                    nonlocal firstResult
                    def newBackprop(grad, output=None):
                        nonlocal firstGrad, firstBackprop
                        firstGrad += grad
                        result = firstBackprop(firstGrad, output=output)
                        firstBackprop = None
                        firstGrad = None
                        return result
                    res = firstResult
                    firstResult = None
                    return res, newBackprop
                assert(secondEvaluate is None)
                secondEvaluate = newEvaluate
                return firstResult, backprop

            super().__init__(evaluate=evaluate, shape=inner.shape, n_params=0)
    class Second(Layer):
        def __init__(self):

            def evaluate(inps):
                nonlocal secondEvaluate
                eva = secondEvaluate
                secondEvaluate = None
                return eva(inps)

            super().__init__(evaluate=evaluate, shape=inner.shape, n_params=inner.n_params, get_params=inner.get_params, load_params=inner.load_params)

    return First(), Second()

def build_hidden(layer):
    for _ in range(2):
        layer = Tanh(Affine(layer, 128))
    return layer

def build_vae():
    LATENT_SIZE=2
    encoder = build_hidden(Input(2*GEN_SEGM_LEN))
    encoder = Split(encoder)
    encoder = Affine(encoder[0], LATENT_SIZE), Affine(encoder[1], LATENT_SIZE)

    dkl = DKLUninormal(mean=encoder[0], logstd=encoder[1])
    encoder = Gauss(mean=encoder[0], logstd=encoder[1])

    decoder_input = Input(LATENT_SIZE)
    decoder = build_hidden(decoder_input)
    decoder = Split(decoder)
    decoder_params = Affine(decoder[0], 2*GEN_SEGM_LEN), Affine(decoder[1], 2*GEN_SEGM_LEN)
    decoder = Gauss(mean=decoder_params[0], logstd=decoder_params[1])

    encOptimizer = Adam(encoder.get_params(), horizon=10, lr=0.01)
    decOptimizer = Adam(decoder.get_params(), horizon=10, lr=0.01)

    class Result:
        def train(inps):
            assert(len(inps.shape) == 2)
            encoder.load_params(encOptimizer.get_value())
            decoder.load_params(decOptimizer.get_value())

            representation, encBackprop = encoder.sample.evaluate(inps)

            picsLogprob, decBackprop = decoder.logprob.evaluate(
                representation,
                sample=inps
            )

            dklValue, dklBackprop = dkl.evaluate(inps)

            decOptimizer.apply_gradient(
                decBackprop(np.ones(len(inps)))
            )

            encOptimizer.apply_gradient(
                dklBackprop(-np.ones(len(inps)))
                + encBackprop(decoder_input.last_gradient)
            )

        def generate(n):
            decoder.load_params(decOptimizer.get_value())

            lats = np.random.randn(n, LATENT_SIZE)*2
            result, _ = decoder_params[0](lats), decoder_params[1](lats)

            return result

        def DKL(inps):
            assert(len(inps.shape) == 2)
            encoder.load_params(encOptimizer.get_value())
            return dkl(inps)

    return Result

def split_obs(obs):
    cutoff = len(obs)%(2*GEN_SEGM_LEN)
    obs = obs[cutoff:]
    #obs = obs.reshape(GEN_SEGM_LEN, -1, 2).transpose((1,0,2))
    return obs.reshape(-1, 2*GEN_SEGM_LEN)

def build_agent():
    model = Input(2)
    model = LReLU(Affine(model, 32))
    model = LReLU(Affine(model, 32))
    model = Affine(model, 1)

    opt = Adam(
        np.random.randn(model.n_params),
        lr=0.01, # 0.05 / 0.01 / 0.02
        horizon=20, # 40 / 10 / 20 / 5
    )
    def sgd_step(traj):
        model.load_params(opt.get_value())
        quiet_actions, backprop = model.evaluate(traj.o)
        grad = ((traj.a - quiet_actions).T * traj.r.T).T
        opt.apply_gradient(backprop(grad))

    def randomize_policy():
        model.load_params(opt.get_value()+((np.random.randn(model.n_params)*0.2)))

    model.randomize_policy = randomize_policy
    model.policy = model
    model.sgd_step = sgd_step
    return model

def save_plot(file_name, trajs, *,
        xs=lambda t: t.o[:,0],
        ys=lambda t: t.o[:,1],
        color=lambda t: ["b", "r"][np.argmax(t.a[0])]):
    import matplotlib.pyplot as plt
    plt.clf()
    for t in trajs:
        plt.plot(
            xs(t), ys(t),
            color=color(t), alpha=0.2, linewidth=2, zorder=1
        )
    plt.gcf().axes[0].set_ylim([-0.075, 0.075])
    plt.gcf().axes[0].set_xlim([-1.25, 0.5])
    plt.gcf().set_size_inches(10, 8)
    plt.gcf().savefig(file_name, dpi=100)

def traj_scorer():
    imagination = build_vae()
    old_agent_trajs = []
    def pick_segment(obs):
        start = np.random.randint(len(obs) - GEN_SEGM_LEN + 1)
        return obs[start:start+GEN_SEGM_LEN,:].reshape(2*GEN_SEGM_LEN)
    def imagine(how_many):
        generated_obs = [go.reshape(-1, 2) for go in imagination.generate(how_many)]
        return [Trajectory(go, [[1,0]]*GEN_SEGM_LEN) for go in generated_obs]
    def frolic():
        nonlocal old_agent_trajs
        old_agent_trajs = old_agent_trajs[-128:]
        agent_obs = np.array([pick_segment(at.o) for at in old_agent_trajs])
        imagination.train(agent_obs)
    def curious(obs):
        obs = split_obs(obs)
        rewards = imagination.DKL(obs)
        def c(r):
            result = []
            for i in range(len(r)):
                result.append(rewards[min(i//GEN_SEGM_LEN, len(rewards)-1)])
            return result
        return c

    class Result:
        def score(agent_traj):
            old_agent_trajs.append(agent_traj)
            for _ in range(4):
                frolic()
            return agent_traj.modified(rewards=curious(agent_traj.o))
        def plot(file_name, transform=lambda x: x):
            generated_trajs = imagine(50)
            tagged_trajs = [Trajectory(agent_traj.o, [[0,1]]*len(agent_traj)) for agent_traj in old_agent_trajs[-20:]]
            forplot = generated_trajs + tagged_trajs
            forplot = [fp.modified(observations=transform) for fp in forplot]
            save_plot(file_name, forplot)
    return Result

def curiosity(world):
    log_dir = "__car"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    world = NormalizedObservations(world)
    agent = build_agent()
    scorer = traj_scorer()

    curNormalize = RunningNormalize(horizon=10)

    def unnormalize(obs):
        return (obs * world.get_std()) + world.get_mean()

    for ep in range(2000):
        agent.randomize_policy()
        agent_traj = episode(world, agent.policy, max_steps=200)

        agent_traj_curio = scorer.score(agent_traj)

        if (ep % 20) == 0:
            scorer.plot(log_dir + "/%04d.png"%ep, unnormalize)
            np.savez(log_dir + "/%04d.npz"%ep, params=agent.get_params(), norm_mean=world.get_mean(), norm_std=world.get_std())

        print(bar(np.mean(agent_traj_curio.r), 100.))

        agent_traj_curio = agent_traj_curio.discounted(horizon=200)
        agent_traj_curio = agent_traj_curio.modified(rewards=curNormalize)
        agent.sgd_step(agent_traj_curio)

def run():
    world = gym.make("MountainCarContinuous-v0")

    if len(sys.argv) >= 2:
        old_render = world.render
        def betterRender():
            import time
            old_render()
            time.sleep(0.01)
        world.render = betterRender
        agent = build_agent()
        for fn in sys.argv[1:]:
            pars = np.load(fn)
            agent.load_params(pars["params"])
            def seeingAgent(inps):
                inps = inps - pars["norm_mean"]
                inps = inps / pars["norm_std"]
                return agent(inps)
            episode(world, seeingAgent, render=True)
    else:
        curiosity(world)

if __name__ == "__main__":
    run()
