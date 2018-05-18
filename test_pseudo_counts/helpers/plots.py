import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from matplotlib import rc

font = {'family': 'Verdana',
        'weight': 'normal'}
rc('font', **font)

from .get_pseudo_counters import get_counters

colors = ['red', 'green', 'blue', 'orange', 'pink']


def smooth(arr, alpha):
    smothed = []
    for i, x in enumerate(arr):
        if i == 0:
            smothed.append(x)
        else:
            smothed.append((1-alpha)*smothed[-1] + alpha*x)
    return np.array(smothed)


def plot_learning_history_with_pgs(logs, pgs,
                                   title='Онлайн обучение',
                                   smooth_pg=True, alpha=0.1,
                                   red_lines=[], filename=None):
    ll_log, kld_log = logs[0], logs[1]

    num_plots = 3
    if smooth_pg:
        num_plots += 1

    plt.figure(figsize=(num_plots*5, 5))

    plt.subplot(1, num_plots, 1)
    plt.plot(ll_log)
    for line in red_lines:
        plt.axvline(x=line, color='red', linewidth=1)
    plt.title('Лог-Правдоподобие ({})'.format(title), fontsize=15)
    plt.ylabel('Лог-Правдоподобие', fontsize=15)
    plt.xlabel('шаги обучения', fontsize=15)
    plt.grid()
    
    plt.subplot(1, num_plots, 2)
    plt.plot(kld_log)
    for line in red_lines:
        plt.axvline(x=line, color='red', linewidth=1)
    plt.title('KL-дивергенция ({})'.format(title), fontsize=15)
    plt.ylabel('KL-дивергенция', fontsize=15)
    plt.xlabel('шаги обучения', fontsize=15)
    plt.grid()

    if smooth_pg:
        plt.subplot(1, num_plots, 3)
        plt.plot(smooth(pgs, alpha), label='smoothed({})'.format(alpha))
        for line in red_lines:
            plt.axvline(x=line, color='red', linewidth=1)
        plt.axhline(y=0, color='r', linewidth=2)
        plt.ylim(-2, 10)
        plt.title('Прирост лог-правдоподобия({})'.format(title), fontsize=15)
        plt.ylabel('PG', fontsize=15)
        plt.xlabel('шаги обучения', fontsize=15)
        plt.grid()
        plt.legend(loc='upper right')
        non_smothed_id = 4
    else:
        non_smothed_id = 3

    plt.subplot(1, num_plots, non_smothed_id)
    plt.plot(pgs)
    for line in red_lines:
        plt.axvline(x=line, color='red', linewidth=1)
    plt.axhline(y=0, color='r', linewidth=2)
    plt.ylim(-2, 10)
    plt.title('Прирост лог-правдоподобия({})'.format(title), fontsize=15)
    plt.ylabel('PG', fontsize=15)
    plt.xlabel('шаги обучения', fontsize=15)
    plt.grid()
    
    if filename is not None:
        plt.savefig(filename)
        plt.close()
        

def plot_learning_history(logs, mu=None, std=None, title='Обучение по батчам', filename=None):
    
    ll_log, kld_log = logs[0], logs[1]
    n_plots = 2 if mu is None else 3
        
    plt.figure(figsize=(5*n_plots, 5))

    plt.subplot(1, n_plots, 1)
    plt.plot(ll_log)
    plt.title('Лог-правдоподобие ({})'.format(title), fontsize=15)
    plt.ylabel('Лог-правдоподобие', fontsize=15)
    plt.xlabel('шаги обучения', fontsize=15)
    plt.grid()
    
    plt.subplot(1, n_plots, 2)
    plt.plot(kld_log)
    plt.title('KL-дивергенция ({})'.format(title), fontsize=15)
    plt.ylabel('KL-дивергенция', fontsize=15)
    plt.xlabel('шаги обучения', fontsize=15)
    plt.grid()

    if mu is not None and std is not None:
        ax= plt.subplot(1, n_plots, n_plots)
        e = Ellipse([0,0], 6, 6)
        e.set_alpha(0.75)
        e.set_facecolor('white')
        e.set_edgecolor('black')
        ax.add_artist(e)

        for i in range(len(mu)):
            e = Ellipse(mu[i], 6*std[i][0], 6*std[i][1])
            e.set_alpha(0.25)
            e.set_facecolor('gray')
            ax.add_artist(e)

            plt.plot(mu[i][0], mu[i][1], '.', markersize=15,
                     color=colors[i], alpha=0.8, label='class {}'.format(int(i+1)))
        plt.legend(loc='lower right')
        plt.grid()
        plt.title('Распределение скрытых переменных', fontsize=16)
        plt.xlim((-4,4))
        plt.ylim((-4,4))    
    
    if filename is not None:
        plt.savefig(filename)
        plt.close()


def plot_point_with_std(mu, std, title='Распределение скрытых переменных'):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, aspect='equal')

    e = Ellipse([0, 0], 6, 6)
    e.set_alpha(0.75)
    e.set_facecolor('white')
    e.set_edgecolor('black')
    ax.add_artist(e)

    for i in range(len(mu)):
        e = Ellipse(mu[i], 6 * std[i][0], 6 * std[i][1])
        e.set_alpha(0.25)
        e.set_facecolor('gray')
        ax.add_artist(e)

        plt.plot(mu[i][0], mu[i][1], '.', markersize=15,
                 color=colors[i], alpha=0.8)
    plt.grid()
    plt.title(title, fontsize=16)
    plt.xlim((-4, 4))
    plt.ylim((-4, 4))


def plot_smoothed_pgs(pgs, img_name=None):
    plt.figure(figsize=(18, 5))
    y_lims = [0.1, 0.01]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.plot(1 / (np.arange(len(pgs)) + 1), label='1/n', linewidth=4)

        plt.plot(smooth(pgs, 0.1),
                 label='smoothed(0.1) PG', linewidth=2)

        plt.plot(smooth(pgs, 0.1) / (np.arange(len(pgs)) + 1) ** 0.5,
                 label='smoothed(0.1) PG/sqrt(n)', linewidth=2)

        plt.plot(smooth(pgs, 0.1) / (np.arange(len(pgs)) + 1),
                 label='smoothed(0.1) PG/n', linewidth=2)

        plt.ylim((0, y_lims[i]))
        plt.xlabel('steps', fontsize=15)
        plt.ylabel('PG', fontsize=15)
        plt.grid()
        plt.legend(loc='upper right', fontsize=15)

        plt.title('PG in comparison with 1/n, scale: y_max={}'.format(y_lims[i]), fontsize=15)
    if img_name is not None:
        plt.savefig(img_name)
        plt.close()


def plot_counters(real_counters, pseudo_counters):
    if len(real_counters.shape) == 2 and len(pseudo_counters.shape) ==  2:
        num_classes = real_counters.shape[0]
        plt.figure(figsize=(7*num_classes, 7))
        for cl in range(num_classes):
            plt.subplot(1, num_classes, cl+1)
            plt.plot(real_counters[cl], label='real_counters')
            plt.plot(pseudo_counters[cl], label='pseudo_counters')
            plt.legend(loc='upper right', fontsize=15)
            plt.title('Counters: class {}'.format(cl+1), fontsize=15)
            plt.xlabel('steps', fontsize=15)
            plt.ylabel('counters', fontsize=15)
            plt.ylim(0, real_counters.shape[1])
            plt.grid()
    else:
        plt.plot(real_counters, label='real_counters')
        plt.plot(pseudo_counters, label='pseudo_counters')
        plt.legend(loc='upper right', fontsize=15)
        plt.title('Counters', fontsize=15)
        plt.ylim(0, real_counters.shape[0])
        plt.grid()


def plot_pseudo_counters_with_alpha_and_degree(pgs, schedule, num_classes,
                                               num_classes_to_plot=None,
                                               alpha_range=[0.01, 0.1, 0.25],
                                               degree_range=[0, 0.5, 1],
                                               img_name=None,
                                               ):
    if num_classes_to_plot is None:
        num_classes_to_plot = num_classes
    else:
        num_classes_to_plot = min(num_classes_to_plot, num_classes)
    n = len(alpha_range)
    plt.figure(figsize=(7 * n, 7 * num_classes))

    real_counters, _, _ = get_counters(schedule, pgs, num_classes)
    for cl in range(num_classes_to_plot):
        for i, alpha in enumerate(alpha_range):

            plt.subplot(num_classes, n, cl*n+i+1)
            plt.plot(real_counters[cl], label='real_counters', linewidth=4)

            for j, degree in enumerate(degree_range):
                smoothed_pgs = smooth(pgs, alpha) / (np.arange(len(pgs)) + 1) ** degree
                _, pseudo_counters, _ = get_counters(schedule, smoothed_pgs, num_classes)
                plt.plot(pseudo_counters[cl], label='degree={}'.format(degree))
                plt.grid()
                plt.xlabel('steps', fontsize=15)
                plt.ylabel('counters: class {}'.format(cl+1), fontsize=15)
                plt.title('Pseudo counters, alpha={}'.format(alpha), fontsize=15)
                plt.legend(loc='upper right', fontsize=15)
                plt.ylim(0, 2 * real_counters.max())
    if img_name is not None:
        plt.savefig(img_name)
        plt.close()
