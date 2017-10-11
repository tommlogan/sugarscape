'''
Plot the sugar scape model after it has simulated
Tom Logan
'''

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import code
from time import sleep
from matplotlib.animation import FuncAnimation

'''
plot the sugarscape simulation
'''
def importData():
    # import the results
    patches = np.load('result/patches.npz')['data']
    with open('result/genetic.npy', 'rb') as fid:
        turtle_genetic = pickle.load(fid)
    #
    with open('result/turtles.npy', 'rb') as fid:
        sim = pickle.load(fid)
    #
    return patches, sim


# import 
patches, sim = importData()

time_steps = patches.shape[0]
size = patches.shape[1]

# Create new Figure and an Axes which fills it.
fig = plt.figure()
ax = fig.add_axes(xlim = (0, size), ylim = (0, size), frameon=False)

# objects
objs = [plt.imshow(patches[0], interpolation="none", cmap = 'gray'),
        plt.plot([],[], linestyle='None', marker='^', color = 'y', markersize=10)[0]]

def init():
    objs[0].set_data([[]])
    objs[1].set_data([],[])
    return objs

def animate(i):
    objs[0].set_data(patches[i])
    x = sim.x[sim.time == i]
    y = sim.y[sim.time == i]
    objs[1].set_data(y,x)
    return objs

animation = FuncAnimation(fig, animate, init_func = init, interval=10, frames = time_steps, blit = True)
animation.save('ants.gif', dpi=80, writer='imagemagick')
'''
Note - to use the gif saver you need to install imagemagick and modify matplotlib's config file
for windows, an installer for imagemagick is here https://www.imagemagick.org/script/download.php#windows
make sure you tick the box to install "converter"

if you get an error still about this installation, you need to change matplotlib's matplotlibrc
instructions are here  https://stackoverflow.com/questions/25140952/matplotlib-save-animation-in-gif-error (not the top answer)

'''
plt.show()
