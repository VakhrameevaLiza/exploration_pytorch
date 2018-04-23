import torch
import numpy as np


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_cummulative_returns(r, gamma=1):
    import scipy.signal
    r = np.array(r)
    assert r.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], r[::-1], axis=0)[::-1]


def rollout(env, agent, max_pathlength=2500, n_timesteps=50000):
    paths = []

    total_timesteps = 0
    while total_timesteps < n_timesteps:
        obervations, actions, rewards, probs_for_actions = [], [], [], []
        mus, logvars, policies = [], [], []
        obervation = env.reset()
        for _ in range(max_pathlength):
            if agent.discrete_type:
                action, action_prob, probs = agent.act(obervation)
                policies.append(probs)
            else:
                action, action_prob, mu, logvar = agent.act(obervation)
                mus.append(mu)
                logvars.append(logvar)

            obervations.append(obervation)
            actions.append(action)
            probs_for_actions.append(action_prob)

            obervation, reward, done, _ = env.step(action)
            rewards.append(reward)
            total_timesteps += 1
            if done or total_timesteps==n_timesteps:
                path = {"observations": np.array(obervations),
                        "policies":  np.array(policies),
                        "mus": np.array(mus),
                        "logvars": np.array(logvars),
                        "probs_for_actions": np.array(probs_for_actions),
                        "actions": np.array(actions),
                        "rewards": np.array(rewards),
                        "cumulative_returns":get_cummulative_returns(rewards),
                       }
                paths.append(path)
                break
    return paths


def linesearch(f, x, fullstep, max_kl):
    """
    Linesearch finds the best parameters of neural networks in the direction of fullstep contrainted by KL divergence.
    :param: f - function that returns loss, kl and arbitrary third component.
    :param: x - old parameters of neural network.
    :param: fullstep - direction in which we make search.
    :param: max_kl - constraint of KL divergence.
    :returns:
    """
    max_backtracks = 10
    loss, _, = f(x)
    for stepfrac in .5**np.arange(max_backtracks):
        xnew = x + stepfrac * fullstep
        new_loss, kl = f(xnew)
        actual_improve = new_loss - loss
        if kl.data.numpy()<=max_kl and actual_improve.data.numpy() < 0:
            x = xnew
            loss = new_loss
    return x


from numpy.linalg import inv
def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    """
    This method solves system of equation Ax=b using iterative method called conjugate gradients
    :f_Ax: function that returns Ax
    :b: targets for Ax
    :cg_iters: how many iterations this method should do
    :residual_tol: epsilon for stability
    """
    p = b.clone()
    r = b.clone()
    x = torch.zeros(b.size())
    rdotr = torch.sum(r*r)
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / (torch.sum(p*z) + 1e-8)
        x += v * p
        r -= v * z
        newrdotr = torch.sum(r*r)
        mu = newrdotr / (rdotr + 1e-8)
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x
