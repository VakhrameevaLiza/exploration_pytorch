import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

font = {'family': 'Verdana',
        'weight': 'normal'}
rc('font', **font)


arr = np.random.rand(1000)

plt.hist(arr, bins=20)
fontsize = 20
plt.xlabel("Коэффициент корреляции Пирсона", fontsize=fontsize)
plt.ylabel("Количество эпизодов", fontsize=fontsize)
plt.title("Корреляция между E-счетчиками и наблюдаемыми счетчиками", fontsize=fontsize)
plt.grid()
plt.show()