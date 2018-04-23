import os, shutil
import matplotlib.pyplot as plt
import numpy as np


def plot_and_log_images(state_action_count, all_q_values, t, folder):
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(121)
    s_a_visitations_plot = plt.imshow(state_action_count / state_action_count.sum(), vmin=0, vmax=1)
    plt.colorbar()
    plt.title('S-A Visitations')
    plt.subplot(122)
    q_values_plot = plt.imshow(all_q_values)
    plt.colorbar()
    plt.title('Q values')
    plt.savefig(folder + '/{}.png'.format(t))
    plt.close(fig)


def plot_q_func_and_visitations(episode_state_action_count,
                  state_action_count, all_q_values, num_episodes, t, dir_img
                  ):
    h, w = 2,5
    fig = plt.figure(figsize=(16,6))

    ax1 = plt.subplot2grid((h,w), (0,0), colspan=2)
    n=state_action_count.shape[0]
    bar_locations = np.arange(n)
    plt.bar(bar_locations, state_action_count.sum(axis=1), color='gray', alpha=0.75)
    plt.title('Total states count')

    ax2 = plt.subplot2grid((h,w), (0,2), rowspan=2)
    plt.imshow(episode_state_action_count)# / episode_state_action_count.sum(), vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Episode S-A Visitations')

    ax2 = plt.subplot2grid((h,w), (0,3), rowspan=2)
    plt.imshow(state_action_count)#/ state_action_count.sum(), vmin=0, vmax=1)
    plt.colorbar()
    plt.title('S-A Visitations')


    ax3 = plt.subplot2grid((h,w), (0,4), rowspan=2)
    plt.imshow(all_q_values)#, vmax=int(np.maximum(10, all_q_values).max()))
    plt.colorbar()
    plt.title('Q-values')

    ax4 = plt.subplot2grid((h,w), (1,0), colspan=2)
    n=episode_state_action_count.shape[0]
    bar_locations = np.arange(n)
    plt.bar(bar_locations, episode_state_action_count.sum(axis=1), color='gray', alpha=0.75)
    plt.title('Episode states count')


    plt.savefig(dir_img + '/episode:{}, step:{}.png'.format(num_episodes, t))
    plt.close(fig)


def plot_q_func_and_visitations_bridge(episode_state_action_count,
                                       state_action_count, all_q_values, num_episodes, t, dir_img
                                      ):
    num_states, num_actions = episode_state_action_count.shape
    ids = np.arange(num_states // 3) * 3 + 1
    h, w = 2,3
    fig = plt.figure(figsize=(10,6))

    ax1 = plt.subplot2grid((h,w), (0,0), rowspan=2)
    plt.imshow(episode_state_action_count[ids])# / episode_state_action_count.sum(), vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Episode S-A Visitations')

    ax2 = plt.subplot2grid((h,w), (0,1), rowspan=2)
    plt.imshow(state_action_count[ids])#/ state_action_count.sum(), vmin=0, vmax=1)
    plt.colorbar()
    plt.title('S-A Visitations')

    ax3 = plt.subplot2grid((h,w), (0,2), rowspan=2)
    plt.imshow(all_q_values[ids])#, vmax=int(np.maximum(10, all_q_values).max()))
    plt.colorbar()
    plt.title('Q-values')

    plt.savefig(dir_img + '/episode:{}, step:{}.png'.format(num_episodes, t))
    plt.close(fig)



def plot_q_func_and_visitations_and_policy(total_states_cnt, episode_state_action_count,
                  state_action_count, all_q_values, all_policy_probs, num_episodes, t, dir
                  ):
    h, w = 2,6
    fig = plt.figure(figsize=(20,6))
    #fig = plt.figure()
    ax1 = plt.subplot2grid((h,w), (0,0), colspan=2)
    n=total_states_cnt.shape[0]
    bar_locations = np.arange(n)
    plt.bar(bar_locations, total_states_cnt, color='gray', alpha=0.75)
    plt.title('Total states count')

    ax2 = plt.subplot2grid((h,w), (0,2), rowspan=2)
    plt.imshow(episode_state_action_count)# / episode_state_action_count.sum(), vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Episode S-A Visitations')

    ax2 = plt.subplot2grid((h,w), (0,3), rowspan=2)
    plt.imshow(state_action_count)#/ state_action_count.sum(), vmin=0, vmax=1)
    plt.colorbar()
    plt.title('S-A Visitations')


    ax3 = plt.subplot2grid((h,w), (0,4), rowspan=2)
    plt.imshow(all_q_values)#, vmax=int(np.maximum(10, all_q_values).max()))
    plt.colorbar()
    plt.title('Q-values')

    ax3 = plt.subplot2grid((h,w), (0,5), rowspan=2)
    plt.imshow(all_policy_probs, vmin=0, vmax=1)#, vmax=int(np.maximum(10, all_q_values).max()))
    plt.colorbar()
    plt.title('Policy probs')

    ax4 = plt.subplot2grid((h,w), (1,0), colspan=2)
    n=episode_state_action_count.shape[0]
    bar_locations = np.arange(n)
    plt.bar(bar_locations, episode_state_action_count.sum(axis=1), color='gray', alpha=0.75)
    plt.title('Episode states count')

    plt.savefig(dir + '/episode:{}, step:{}.png'.format(num_episodes, t))
    plt.close(fig)

