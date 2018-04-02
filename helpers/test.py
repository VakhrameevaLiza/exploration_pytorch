import numpy as np
import matplotlib.pyplot as plt

def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)

h, w = 2,5



plt.figure(figsize=(12,5))
ax1 = plt.subplot2grid((h,w), (0,0), colspan=1)
ax1 = plt.subplot2grid((h,w), (0,1), colspan=2)

n=5
bar_locations = np.arange(n)
data = np.arange(n)
plt.bar(bar_locations, data, color='gray', alpha=0.75)
plt.title('Total states count')

ax1 = plt.subplot2grid((h,w), (0,3), rowspan=2)
ax1 = plt.subplot2grid((h,w), (0,4), rowspan=2)
ax1 = plt.subplot2grid((h,w), (1,0), rowspan=1)
ax1 = plt.subplot2grid((h,w), (1,1), colspan=2)
#ax2 = plt.subplot2grid((h,w), (1,0), colspan=2)
#ax3 = plt.subplot2grid((h,w), (1, 2), rowspan=2)
#ax4 = plt.subplot2grid((h,w), (2, 0))
#ax5 = plt.subplot2grid((h,w), (2, 1))

#plt.suptitle("subplot2grid")
make_ticklabels_invisible(plt.gcf())
plt.show()