#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:25:34 2022

@author: mischasch
"""

import PyMoCa as mc, numpy as np

sim = mc.Simulation('test')

#create new parameter
sim.addParameter(mc.Param('par1', s = np.ones(100)))

#do again
sim.addParameter(mc.Param('par1', s = 3 *np.ones(100)))

#try to add new element by direcg access
#sim['new'] = mc.Param('par1', s = np.ones(100))