import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(env_name='CartPole-v1', hidden_sizes=[32], baseline_size = [32], lr=1e-2, 
          baseline_lr=1e-2, epochs=50, n_step=2, batch_size=5000, render=None):
    
    loss = []
    reward = []
    totlen = []
    # rendering
    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name,render_mode = render)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts]) # input obs, output action
    V_estimator = mlp(sizes=[obs_dim]+baseline_size+[1]) # can also be baseline, input obs, output E[return]

    # convert list to tensor
    def tot(list):
        return torch.FloatTensor(np.array(list))
    
    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * (weights-V_estimator(obs))).mean()

    def baseline_loss(obs, weights):
        return nn.MSELoss()(V_estimator(obs).squeeze(),weights)
    # make optimizer
    optimizer_policy = Adam(logits_net.parameters(), lr=lr)
    optimizer_baseline = Adam(V_estimator.parameters(), lr=baseline_lr)

    # for training policy
    def update_policy():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs, _ = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep
        # render first episode of each epoch

        # collect experience by acting in the environment with current policy
        while True:

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(tot(obs))
            obs, rew, done, _, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)
            if len(ep_rews)>10000:
                done = True
            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len
                # reset episode-specific variables
                (obs,_), done, ep_rews = env.reset(), False, []

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer_policy.zero_grad()
        batch_loss = compute_loss(obs=tot(batch_obs),
                                  act=tot(batch_acts),
                                  weights=tot(batch_weights)
                                  )
        batch_loss.backward()
        optimizer_policy.step()
        return batch_loss.detach().numpy(), (batch_obs, batch_acts, batch_weights, batch_rets, batch_lens, ep_rews)

    def update_baseline(inp):
        optimizer_baseline.zero_grad()
        (batch_obs, batch_acts, batch_weights, batch_rets, batch_lens, ep_rews) = inp
        b_loss = baseline_loss(obs =tot(batch_obs),
                               weights=tot(batch_weights)
                               ) 
        b_loss.backward()
        optimizer_baseline.step()
    # training loop
    for i in range(epochs):
        batch_loss, data = update_policy()
        if i % n_step ==0:
            update_baseline(data)
        loss += [batch_loss]
        reward += [np.mean(data[3])]
        totlen += [np.mean(data[4])]
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, reward[-1], totlen[-1]))
    return logits_net,(loss, reward, totlen)