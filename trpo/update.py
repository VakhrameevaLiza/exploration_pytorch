import torch
from torch.autograd import Variable

from trpo.loss import get_loss, get_discrete_kl, get_normal_kl
from trpo.utils import conjugate_gradient, linesearch
from trpo.utils import get_flat_params_from, set_flat_params_to


def update_step(agent, optimizer, observations, actions,
                cummulative_returns, old_probs_for_actions, max_kl,
                old_policies=None, old_mu=None, old_logvar=None):
    """
    This function does the TRPO update step
    :param: observations - batch of observations
    :param: actions - batch of actions
    :param: cummulative_returns - batch of cummulative returns
    :param: old_probs - batch of probabilities computed by old network
    :param: max_kl - controls how big KL divergence may be between old and new policy every step.
    :returns: KL between new and old policies and the value of the loss function.
    """
    # Here we prepare the information
    observations = Variable(torch.FloatTensor(observations))
    if agent.discrete_type:
        actions = torch.LongTensor(actions)
    else:
        actions = Variable(torch.FloatTensor(actions))

    cummulative_returns = Variable(torch.FloatTensor(cummulative_returns))
    old_probs_for_actions = Variable(torch.FloatTensor(old_probs_for_actions))
    if old_policies is not None:
        old_policies = Variable(torch.FloatTensor(old_policies))
    if old_mu is not None and old_logvar is not None:
        old_mu = Variable(torch.FloatTensor(old_mu))
        old_logvar = Variable(torch.FloatTensor(old_logvar))


    # Here we compute gradient of the loss function
    loss, value_loss = get_loss(agent, observations, actions,
                                cummulative_returns, old_probs_for_actions)

    optimizer.zero_grad()
    value_loss.backward()
    optimizer.step()

    grads = torch.autograd.grad(loss, agent.policy.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    def Fvp(v):
        # Here we compute Fx to do solve Fx = g using conjugate gradients
        # We actually do here a couple of tricks to compute it efficiently
        if agent.discrete_type:
            kl = get_discrete_kl(agent, observations, old_policies)
        else:
            kl = get_normal_kl(agent, observations, (old_mu, old_logvar))

        grads = torch.autograd.grad(kl, agent.policy.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, agent.policy.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * 0.1

    # Here we solveolve Fx = g system using conjugate gradients
    stepdir = conjugate_gradient(Fvp, -loss_grad, 10)

    # Here we compute the initial vector to do linear search
    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)

    # Here we get the start point
    prev_params = get_flat_params_from(agent.policy)

    def get_loss_kl(params):
        # Helper for linear search
        set_flat_params_to(agent.policy, params)
        loss = get_loss(agent, observations, actions, cummulative_returns, old_probs_for_actions)[0]
        if agent.discrete_type:
            kl = get_discrete_kl(agent, observations, old_policies)
        else:
            kl = get_normal_kl(agent, observations, (old_mu, old_logvar))
        return loss, kl

    # Here we find our new parameters
    new_params = linesearch(get_loss_kl, prev_params, fullstep, max_kl)

    # And we set it to our network
    set_flat_params_to(agent.policy, new_params)

    return get_loss_kl(new_params)
