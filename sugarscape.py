'''
Sugarscape
Python 3
Object-oriented program
Agent-based model

Tom M Logan
www.tomlogan.co.nz

The sugarscape model as described in
Growing Artificial Societies
by Axtell & Epstein
'''

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot
from time import sleep
from scipy import signal
import pandas as pd
import cProfile
import pstats
import sqlite3
import code
import pickle

def main():
    '''
    Runs the sugar scape model.
    '''
    
    # the size of the board
    size = 50
    time_steps = 500

    # population of agents
    turtle_num = 10 #400

    # create the board
    environ = Environment(size, 5)

    # init the turtles
    turts = Turtles(size, turtle_num, 6, 6)

    # record initial
    history = History(environ, turts)

    # simulate the game of life
    for t in range(1, time_steps + 1):
        # if all the turtles are dead, stop
        if len(turts.id) == 0:
            break
        #
        # grow patches
        environ.grow()
        #
        # move turtles
        turts.move(environ)
        #
        # record state
        history.record(environ, turts, t)

    # save simulation
    history.save()

class Environment:
    '''
    A 2D area explored by animal, containing 2D qualities of sugar level, sugar capacity, and sugar regeneration.
    '''

    def __init__(self, size, capacity):
        '''
        Initializes the landscape and patches
        Inputs:
            size
            capacity
            alpha - growback rule
        '''
        
        # Initialize the board size
        self.size = size
        if type(size) == int:
            # if integer provided, return square environment
            self.size = (size, size) 
        # self.size = np.array(size, dtype=int)

        # patch properties:
        # self.capacity = np.ones(self.size) * capacity # uniform capacity
        self.capacity = np.random.randint(0, capacity + 1, size = self.size) # random capacity
        # the amount of sugar in each patch is random
        self.level = np.random.randint(0, capacity + 1, size = self.size)
        # ensure that the amount of sugar in a patch doesn't exceed its capacity:
        self.level = np.minimum.reduce([self.level,self.capacity])

        # growback
        # self.growback = alpha


    def grow(self):
        '''
        sugar levels grow back
        '''
        # immediate full growback
        # self.level = self.capacity

        # increase by 1 each time
        # self.level = np.minimum.reduce([self.level + 1, self.capacity])

        # no growth
        self.level = self.level


class Turtles:
    '''
    Turtles with position in environment, amount of sugar, it's metabolism, and vision
    Methods and instances for turtles
    '''

    def __init__(self, size, turtle_num, metabolism_max, vision_max):
        '''
        Instances for the turtles
        Inputs:
            position - randomly generated from size of environment
        '''
        
        # list of turtles
        self.id = np.array(range(turtle_num))

        # randomly generate position
        self.x = np.random.randint(0, size, turtle_num)
        self.y = np.random.randint(0, size, turtle_num)
        
        # characteristics
        self.metabolism = np.random.randint(1, metabolism_max + 1, turtle_num)
        self.vision = np.random.randint(1, vision_max + 1, turtle_num)
        self.holding = np.random.randint(5, 25, turtle_num)


    def move(self, environ):
        '''
        Loop through the turtles and move them
        '''
        for i in self.id:
            # move turtle
            self.movement(i, environ)
            # eat
            self.holding[i] = self.holding[i] - self.metabolism[i] + environ.level[self.x[i], self.y[i]]
            # update patch sugar level
            environ.level[self.x[i], self.y[i]] = 0
            # die
            if self.holding[i] <= 0:
                self.id = self.id[self.id != i]



    def movement(self, i, environ):
        '''
        Movement rule:
            look as far as vision permits in the cardinal directions
            identify site(s) with max sugar
            go to closest site with max sugar
            collect all the sugar
        '''
        # turtle properties
        x, y = self.x[i], self.y[i]
        view = self.vision[i] + 1
        patches = environ.level.copy()
        size = patches.shape[0]

        # remove occupied patches from consideration
        occupied_x = self.x[self.id[self.id != i]]
        occupied_y = self.y[self.id[self.id != i]]
        try:
            patches[occupied_x, occupied_y] = -1
        except:
            code.interact(local = locals())

        # code.interact(local = locals())
        # mask environ with kernel and set -1
        mask = np.ones(environ.size, dtype = bool)
            # set the vision region to false
        mask[x, max(y - view, 0) : min(size, y + view)] = False
        mask[max(x - view, 0) : min(size, x + view), y] = False
            # set everything outside vision to -1
        patches[mask] = -1

        # find index of maximum
        # if statement when len > 1
        idx = np.where(patches == np.amax(patches))
        # import code
        # code.interact(local = locals())
        if len(idx[0]) > 1:
            # choose closest
            dist_x = abs(idx[0] - x)
            dist_y = abs(idx[1] - y)
            dist = dist_x + dist_y
            sites = np.where(dist == np.amin(dist))
            # if there are multiple min dist sites equally attractive, pick randomly
            if len(sites[0]) > 1:
                # choose a random interval 
                r = np.random.randint(0,len(sites[0]))
                try:
                    sites = sites[0][r]
                except:
                    code.interact(local = locals())
            # choose the idx
            idx = (idx[0][sites], idx[1][sites])
                            
        # move
        self.x[i] = int(idx[0])
        self.y[i] = int(idx[1])



class History:
    '''
    Stores the historical states
    '''
    def __init__(self, environ, turts):
        '''
        create the historical records
        '''
        # reshape levels and save
        self.patches = np.reshape(environ.level, (1,) + environ.level.shape)
        # prepare the df for the time step
        self.turtles_time = self.turtle_properties_df(turts, time = 0)
        # add the genetic df
        genetic_df = pd.DataFrame({'id' : turts.id, 
                                'metabolism':turts.metabolism,
                                'vision':turts.vision})
        genetic_df = genetic_df.set_index('id')
        self.turtles_genetic = genetic_df

    def record(self, environ, turts, time):
        '''
        record the states
        '''
        # record sugar level
        self.record_environment(environ)

        # record turtle position and state
        self.turtles_time = self.turtles_time.append(self.turtle_properties_df(turts, time))



    def record_environment(self, environ):
        '''
        record the environment state
        # append the environment to the historical
        '''
        # make the 2d array 3d
        environ = np.reshape(environ.level, (1,) + environ.level.shape)

        # append to history
        self.patches = np.append(self.patches, environ, axis=0)


    def turtle_properties_df(self, turts, time):
        '''
        turn the turtle properties to a df
        '''
        # create data dic
        d = {'id' : turts.id, 
        'time' : np.ones(len(turts.id)) * int(time), 
        'x' : turts.x[turts.id], 
        'y' : turts.y[turts.id], 
        'holding' : turts.holding[turts.id],
        # 'alive' : np.ones(len(turts.id), dtype = bool)
        }
        # put into df
        try: 
            turtTime_df = pd.DataFrame(d)
        except:
            code.interact(local = locals())
        # turtTime_df = turtTime_df.set_index(['id', 'time'])
        
        return turtTime_df


    def save(self):
        '''
        pickle the results
        '''
        # save the array
        np.savez_compressed('result/patches', data=self.patches)

        # save the pandas dataframes
        with open('result/genetic.npy', 'wb') as fid:
            pickle.dump(self.turtles_genetic, fid, protocol = 2)
        with open('result/turtles.npy', 'wb') as fid:
            pickle.dump(self.turtles_time, fid, protocol = 2)


if __name__ == '__main__':
    main()