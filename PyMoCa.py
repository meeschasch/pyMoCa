#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 18:49:00 2021

@author: mischasch

"""

#imports
import pandas as pd, scipy.stats as stats, numpy as np, re, math, os, abc
import matplotlib.pyplot as plt, seaborn as sns, pickle, warnings
from pathlib import Path
from matplotlib import style
from abc import ABC, abstractmethod

#plot presets
#plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.dpi': 100})
#plt.rcParams.update({'legend.loc': 'best'})
style.use('seaborn')

#parent class for simulation elements
class simulation_element(ABC):
    def __init__(self, name, s, unit = '-'):
        '''        
        constructor. see documentation in subclasses
        '''
        
        self.name = name
        self.unit = unit
        self._s = s
        self.figsize = (6.4*0.8, 4.8*0.8)#standard figszie in jupyter notebook
        self.savefig = False
        self.savedir = None
        
# =============================================================================
#  Non-abstract methods       
# =============================================================================imp
        
    def compute_running_average(self):
        '''
        computes the running average of an elements .s attribute.
       
        Returns
        -------
        ra : np.array
             running average
       
        '''
        ra = np.zeros(len(self.s))
       
        i = 0
        while i+1 <= len(self.s):
             
             ra[i] = np.mean(self.s[0:i+1])
             i = i+1
       
        return ra

    def compute_running_se(self):
           
         
        """
        computes the running standard error (std / sqrt(n)) of an elements .s attribute

       
        Returns
        -------
        array of sane length as input with the running standard error values
       
        """
        rs = np.zeros(len(self.s))
       
        i = 1
        while i+1 <= len(self.s):
             
             rs[i] = np.std(self.s[0:i+1]) / np.sqrt(i)
             i = i+1
       
        return rs
# =============================================================================
# Properties
# =============================================================================
    @property
    def summary(self):
        return self.compile_summary()
    
    @property 
    def running_average(self):
        return self.compute_running_average()
    
    @property
    def running_se(self):
        return self.compute_running_se()
    
    #s
    @property
    def s(self):
        return self._s
    
    @s.setter
    def s(self, newvalue):
        #check type
        if newvalue is not None and not isinstance(newvalue, np.ndarray):
            raise ValueError('.s attribute must be a numpy array')
            
        self._s = newvalue
            
        
    
# =============================================================================
# Abstract methods that will be overridden in subclasses        
# =============================================================================
    @abstractmethod
    def plot(self):
        ...
    @abstractmethod
    def compile_summary(self):
        ... 

    
        
class parameter(simulation_element):
    def __init__(self, name, s = None, unit = '-', dist = None, hist_data = None):
        '''
        constructor for new parameter object.

        Parameters
        ----------
        name : string
            parameter name.
        s : np.array of length nmc, optional
            numerical simulation values The default is None.
        unit : string, optional
            the parameters physical unit. The default is '-'.
        dist : scipy.stats, optional
            the statistical distribution of the parameter values. The default is None.
        hist_data : np.array of any length, optional
            historical values of the parameter. The default is None.

        '''
        #initialise parent class
        super().__init__(name, s, unit)
        
        
        #attributes of derived class
        self._dist = dist
        self._hist_data = hist_data
        
    #dist
    @property
    def dist(self):
        return self._dist
    
    @dist.setter
    def dist(self, newvalue):       
        if newvalue is not None and not isinstance(newvalue, stats._distn_infrastructure.rv_frozen):
            raise ValueError('.dist attribute must be a scipy.stats element')
            
    @property
    def hist_data(self):
        return self._hist_data
    
    @hist_data.setter
    def hist_data(self, newvalue):
        #check type
        if newvalue is not None and not isinstance(newvalue, np.ndarray):
            raise ValueError('.hist_data attribute must be a numpy array')
            
        self._s = newvalue
        
        
    def plot(self, mode = 'dhs'):
        """
        creates and shows a plot of the parameter. Use the mode parameter to choose what to include in the plot.


        Parameters
        ----------
        mode : string
            Plots the parameter. Default: 'dhs'. The submitted string can be a combination of the followiung characters:
            'd': distribution
            'h': historic data
            's': simulation data

        """
        parameter_visualizer(self, mode).plot()
        
        return
    
    def compile_summary(self):
        """
        compiles a summary table containing a statistical description of the input parameter.
        All data available (simulation, historical, distribution) are included.

        Returns
        -------
        pandas.DataFrame containing the description

        """

        # create pd.DataFrame for table
        indexs = ['n / fit', 'unit', 'mean', 'std', 'min', 'max', 'P10', 'P50', 'P90']
                
        t= pd.DataFrame(index = indexs)
    
        #fill values for the parameter distribution
        if self.dist is not None:
            t.loc['n / fit', 'fit'] = self.dist.dist.name
            t.loc['mean', 'fit'] = self.dist.mean()
            t.loc['std', 'fit'] = self.dist.std()
            t.loc['min', 'fit'] = self.dist.ppf(0)
            t.loc['max', 'fit'] = self.dist.ppf(1)
            t.loc['P90', 'fit'] = self.dist.ppf(0.1)
            t.loc['P50', 'fit'] = self.dist.ppf(0.5)
            t.loc['P10', 'fit'] = self.dist.ppf(0.9)
        
        #fill values for the historical data
        if self.hist_data is not None:
           #t['data'] = t.loc[:,'fit'] #create column for data
           t.loc['mean','data'] = np.mean(self.hist_data)
           t.loc['std','data'] = np.std((self.hist_data))
           t.loc['min','data'] = min(self.hist_data)
           t.loc['max','data'] = max(self.hist_data)
           t.loc['n / fit','data'] = self.hist_data.size
           t.loc['P90','data'] = np.quantile(self.hist_data, 0.1)
           t.loc['P50','data'] = np.quantile(self.hist_data, 0.5)
           t.loc['P10','data'] = np.quantile(self.hist_data, 0.9)

        #fill values for the simulation data
        if self.s is not None:
           #t['sim'] = t.loc[:,'fit'] #create column for data
           t.loc['mean','sim'] = np.mean(self.s)
           t.loc['std','sim'] = np.std((self.s))
           t.loc['min','sim'] = min(self.s)
           t.loc['max','sim'] = max(self.s)
           t.loc['n / fit','sim'] = self.s.size
           t.loc['P90','sim'] = np.quantile(self.s, 0.1)
           t.loc['P50','sim'] = np.quantile(self.s, 0.5)
           t.loc['P10','sim'] = np.quantile(self.s, 0.9)
           
        t.loc['unit', :] = self.unit
           
        return t.transpose()
            
    def realise(self, nmc, overwrite_existing = False):
        """
        Draws nmc numbers of the parameters distribution and sets the set as self.s

        Parameters
        ----------
        nmc : int
            numnber of realisations (must be >= 1)
        overwrite_existing: If True, existing simulation data will be overwritten. Default: False.

        Returns
        -------
        None

        """
        if (self.s is None) and (self.dist is None):
            raise Exception('Parameter '+ self.name + ' has neither a distribution nor simulation data and is hence unusable.')
            
        if (self.s is None) | ((self.s is not None) & (overwrite_existing)):
            try:
                self.s = self.dist.rvs(size = nmc)
            except:
                raise Exception('Realisation failed')
        
    
class result(simulation_element):
    def __init__(self, name, s = None, unit = '-'):

        super().__init__(name, s, unit)
        
    def plot(self, showfig = True):
        """
        plots a histogram of the results .s attribute

        """
        result_visualizer(self).plot()
    
    
    def compile_summary(self):
        """
        compiles a summary table containing a statistical description of the result.

        Returns
        -------
        pandas.DataFrame containing the description

        """

        # create pd.DataFrame for table
        indexs = ['n', 'unit', 'mean', 'std', 'min', 'max', 'P90', 'P50', 'P10']
                
        t= pd.DataFrame(columns = ['result'], index = indexs)
    
        
        t.loc['mean','result'] = np.mean(self.s)
        t.loc['std','result'] = np.std((self.s))
        t.loc['min','result'] = min(self.s)
        t.loc['max','result'] = max(self.s)
        t.loc['n','result'] = self.s.size
        t.loc['P90','result'] = np.quantile(self.s, 0.1)
        t.loc['P50','result'] = np.quantile(self.s, 0.5)
        t.loc['P10','result'] = np.quantile(self.s, 0.9)
        t.loc['unit', 'result'] = self.unit
           
        return t.transpose()
                  
        
class Simulation:
    def __init__(self, name, savefigs = False, nmc = 100, savedir = '', figsize = (6.4*0.8, 4.8*0.8)):
        """
        creates a new simulation object.

        Parameters
        ----------
        name : string
            name of the simulation
        nmc: int
            number of realisations
        safedir: Path to where plots are stored. Leave blank if 
        savefigs: If True, all generated plots are saved in savedir
        figsize: widthxlength in inches (applies to most figures)

        Returns
        -------
        None.

        """
        #self.name = name
        self.parameters = []
        self.results = []

        
        #dictionary with all necessary settings
        self.settings = {
            'name': name,
            'nmc': nmc,
            'savedir': savedir,
            'savefigs': savefigs,
            'figsize': figsize}
        
        if savedir != '': 
            self.__create_savedirs() #create structure of directories to save plots
        
    def __getitem__(self, index):
        '''
        lets the user access parameters and result of a simulation using simulation_name[attribute name]
        '''
        if index in self.parameter_names:
            for param in self.parameters:
                if param.name == index:
                    return param
        elif index in self.result_names:
            for result in self.results:
                if result.name == index:
                    return result
        else:
            raise ValueError(index + ' not in parameters or results')
            
    def __setitem__(self, key, newvalue):
        #only continue if new item is a result or a paraeter
        if not isinstance(newvalue, (parameter, result)):
            raise ValueError('New item must bei either a parameter or a result')
         
        #adding new elements using __setitem__ not allowed
        if newvalue.name not in self.parameters + self.results:
            raise ValueError('Use addParameter() or addresult() to add new elements')
            
        if isinstance(newvalue, parameter): #it's a parameter
            #remove existing entry
            if key in self.parameter_names: #parameter already exists
                for par in self.parameters:
                    if par.name == key:
                        warnings.warn('Parameter already exsists and is replaced with the new one')
                        self.parameters.remove(par) #remove existing parameter
            
                #add new
                self.parameters.append(newvalue)
            
        elif isinstance(newvalue, result):
            #remove existing entry
            if key in self.result_names:
                for res in self.results:
                    if res.name == key:
                        warnings.warn('result already exsists and is replaced with the new one')
                        self.results.remove(res)
            
                #add new
                self.results.append(newvalue)

            
        
            
    def __create_savedirs(self):
        '''
        Creates some directories within savedir, where plots can later be stored.

        '''
        dirs = ['parameters', 'results', 'qc and sensitivity']
        
        for diri in dirs:
            path = self.settings['savedir'] / diri
            if not os.path.isdir(path):
                os.makedirs(path)
        
    
    #properties
    @property
    def parameter_names(self):
        return [par.name for par in self.parameters]
    
    @property
    def result_names(self):
        return [res.name for res in self.results]
    
    @property
    def summary(self):
        return self.assemble_summary()
    
    @property
    def dump(self):
        return self.assemble_dump()
    
    @property
    def ciom_p10(self):
        return simulation_analyzer(self).compute_ciom()[0]
        
    @property
    def ciom_p90(self):
        return simulation_analyzer(self).compute_ciom()[1]
    
    @property
    def pearsonr(self):
        return simulation_analyzer(self).correlate()[0]
    @property
    def spearmansr(self):
        return simulation_analyzer(self).correlate()[1]
# =============================================================================
# Methods
# =============================================================================
    
    def add(self, simulation_element):
        '''
        adds a simulation element (parameter or result) to the simulation.

        Parameters
        ----------
        simulation_element : simulation_element
            element to add.

        Returns
        -------
        None.

        '''
        try:
            self.validate_simulation_element(simulation_element)
        except:
            raise ValueError('ELement could not be added')
        
        #pass on simulation wide parameters to simulation elements
        simulation_element.figsize = self.settings['figsize'] 
        simulation_element.savefig = self.settings['savefigs']
        simulation_element.savedir = self.settings['savedir']
        
        if isinstance(simulation_element, parameter):
            self.parameters.append(simulation_element)
        elif isinstance(simulation_element, result):
            self.results.append(simulation_element)

    def remove(self, name):
        '''
        removes an element (identified by its name) from the simulation.

        Parameters
        ----------
        remove_element : simulation_element
            element to remove.

        Returns
        -------
        None.

        '''
        if name in self.parameter_names:
            for p in self.parameters:
                if p.name == name:
                    self.parameters.remove(p)
                    
        elif name in self.result_names:
            for r in self.results:
                if r.name == name:
                    self.results.remove(r)
        else:
            raise Exception('No element with this name in the simulation')
        
    def validate_simulation_element(self, new_element):
        '''
        validates a new simulation element before added / altered. If the validation failes, an error is raised.

        Parameters
        ----------
        simulation_element : simulation_element
            element to add.

        Returns
        -------
        None

        '''
        #is simulation_element?
        if not isinstance(new_element, simulation_element):
            raise ValueError('Only parameters and results can be added to a simulation.')
            
        #shape of s (if given)
        if (new_element.s is not None) and (len(new_element.s) != (self.settings['nmc'])):
            raise ValueError('The provided simulation data must be a vector of length nmc.')
            
        #element already present?
        if new_element.name in self.parameter_names + self.result_names:
            raise ValueError('Element of that name already present in simulation.')
            
        
        
    def realise_parameter_sets(self, overwrite_existing = False):
        """
        draws nmc values of the distribution of each parameter. 

        Parameters
        ----------
        nmc : int
            number of realisations
        overwrite_existing: if True, existing parameter values (.s attribute) will be kept, otherwise overwritten. Default: True

        Returns
        -------
        None.

        """
        
        
        #print('Realising parameter sets wtih {} values'.format(self.settings['nmc']))
        for param in self.parameters:
            s = param.realise(self.settings['nmc'], overwrite_existing = overwrite_existing)
                
        
    def assemble_summary(self):
        """
        Statistical description of all simulation parameters and results

        Returns
        -------
        pandas.DataFrame with parameters and results as rows and properties as columns

        """
        #compose dataframe
        cols =  ['name', 'type', 'unit', 'mean', 'std', 'min', 'max', 'P90', 'P50', 'P10']
        s = pd.DataFrame(columns = cols)
        s.set_index('name', inplace = True)

        #add parameters
        for param in self.parameters:
            s.loc[param.name, 'type'] = 'parameter'
            s.loc[param.name, 'unit'] = param.unit
            s.loc[param.name, 'mean'] = np.mean(param.s)
            s.loc[param.name, 'std'] = np.std(param.s)
            s.loc[param.name, 'min'] = min(param.s)
            s.loc[param.name, 'max'] = max(param.s)
            s.loc[param.name, 'P90'] = np.quantile(param.s, 0.1)
            s.loc[param.name, 'P50'] = np.quantile(param.s, 0.5)
            s.loc[param.name, 'P10'] = np.quantile(param.s, 0.9)
            
        #add results
        for result in self.results:
            s.loc[result.name, 'type'] = 'result'
            s.loc[result.name, 'unit'] = result.unit
            s.loc[result.name, 'mean'] = np.mean(result.s)
            s.loc[result.name, 'std'] = np.std(result.s)
            s.loc[result.name, 'min'] = min(result.s)
            s.loc[result.name, 'max'] = max(result.s)
            s.loc[result.name, 'P90'] = np.quantile(result.s, 0.1)
            s.loc[result.name, 'P50'] = np.quantile(result.s, 0.5)
            s.loc[result.name, 'P10'] = np.quantile(result.s, 0.9)        

                         
        return s
    
    def assemble_dump(self):
        '''
        returns: pandas DataFrame containing all parameters and all results for each realisation
        '''
        colsp = [f'{parami.name} [{parami.unit}]' for parami in self.parameters] +  \
            [f'{parami.name} [{parami.unit}]' for parami in self.results]
            
        
        
        dump = pd.DataFrame(columns = np.hstack([self.parameter_names, self.result_names]))
        
        for column in self.parameter_names + self.result_names:
            dump[column] = self[column].s
            
        dump.columns = colsp
        
        return dump
    
# =============================================================================
# Methods that point to other classes where simulation elements are analyzed
# and plotted
# =============================================================================
    
    def convergence(self, plot = True):
        """
        evaluate the convergeence of an element in a simulation. Convergence is indicated by the reduction of 
        the standard error of the running average.

        Parameters
        ----------
        plot : bool, optional
            show a plot of the evolution of all parameter means

        Returns
        -------
        None.

        """
        simulation_visualizer(self).plot_convergence(plot)
     
  
    def plot_pairplot(self, subset_par = None, subset_res = None, focus = False):
        """
        

        Parameters
        ----------
        subset_par : list, optional
            List of parameter names that should be plotted. If empty, all parameters are considered. The default is None.
        subset_res : list, optional
            ist of results names that should be plotted. If empty, all results are considered. The default is None.
        focus: narrow the x- and y lims of the plots to the 90% uncertainty intervals

        Returns
        -------
        None.

        """

        simulation_visualizer(self).plot_pairplot(subset_par, subset_res, focus)
            
    def plot_tornado(self, limit_parameters = True):
        '''
        creates a tornado plot for each result, showing sensitivity on all parameters.
        Sensitivities are computed as change in output mean.
        
        Parameters
        ----------
        limit_parameters : bool, optional
            only show parameters in tornado plot whose ciom is larger than the standard error of the results mean
        
        '''
        simulation_visualizer(self).plot_tornado(limit_parameters)
          
    def show_parameters(self, mode = 'sdh', showtable = True, showfig = True):
        """
        shows a plot and a summary table for each parameter

        Parameters
        ----------
        showtable: If True, shows summary table after each plot
        showfig: If True (Default), shows each plot
        mode : string, optional
            Plots the parameter. Default: 'd'. The submitted string can be a combination of the followiung characters:
            'd': distribution
            'h': historic data
            's': simulation data

        Returns
        -------
        None.

        """
        for param in self.parameters:
            fig = param.plot(mode)
            
            if showtable:
                r = param.summary
                display(r)
                
            if self.settings['savefigs']:
                file = 'par_' + param.name + '.png'
                fig.savefig(Path(self.settings['savedir']) / 'parameters' / file, bbox_inches = 'tight', dpi = 250)
                
            
    def show_results(self, showtable = True, showfig = True):
        """
        shows a plot and a summary table for each result
        
        Parameters
        --------
        showtable: If True, shows summary table after each plot
        showfig: If True (Default), shows each plot

        Returns
        -------
        None.

        """
        for result in self.results:
            fig = result.plot(showfig = showfig)
            if showtable:
                display(result.summary)
                
            if self.settings['savefigs']:
                file = 'res_' + result.name + '.png'
                fig.savefig(Path(self.settings['savedir']) / 'results' / file, bbox_inches = 'tight', dpi = 250)
        
    def savesim(self):
        """
        Saves the simulation in a pickle on the savedir

        """
        file = self.settings['savedir'] / (self.name  + '.pickle')
        with open(file, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        
class parameter_visualizer():
    def __init__(self, parameter: simulation_element, mode: str = 'dhs') -> None:
        '''
        

        Parameters
        ----------
        element : simulation_element
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        '''
        self.parameter = parameter
        self.mode = mode
        
    def plot(self):
        """
        creates and shows a plot of the parameter. Use the mode parameter to choose what to include in the plot.


        Parameters
        ----------
        mode : string
            Plots the parameter. Default: 'dhs'. The submitted string can be a combination of the followiung characters:
            'd': distribution
            'h': historic data
            's': simulation data
        showfig: If True, plot is showed

        """
        
        #cehck whether mode is valid
        if re.search('[^dhs]', self.mode) is not None:
            raise ValueError('mode string contains characters other than dhs')
        
        fig, ax = plt.subplots()
        fig.set_size_inches(*self.parameter.figsize)
        
    
        
        if self.parameter.hist_data is not None:
            xmin = min(min(self.parameter.hist_data), self.parameter.dist.ppf(0.01)) * 0.9
            xmax = max(max(self.parameter.hist_data), self.parameter.dist.ppf(0.99)) * 1.1
        elif self.parameter.dist is not None:
            xmin = self.parameter.dist.ppf(0.01) * 0.9
            xmax = self.parameter.dist.ppf(0.99) * 1.1
        else:
            xmin = min(self.parameter.s) * 0.9
            xmax = max(self.parameter.s) * 1.1
                
        
        x = np.linspace(xmin, xmax, 100)
        #nbins = 100
        
        #print distribution
        if 'd' in self.mode:
            #pdf
            if self.parameter.dist is not None:
                ax.plot(x, self.parameter.dist.pdf(x), color = 'lightcoral', lw = 2, label = 'fit')
           
        #print historical data
        if (self.parameter.hist_data is not None)& ('h' in self.mode):
           #histogram
           ax.hist(self.parameter.hist_data, alpha=0.9, density=True, color = 'deepskyblue', label = 'observed')
           
        #print simulation data
        if (self.parameter.s is not None) & ('s' in self.mode):
            ax.hist(self.parameter.s, alpha=0.2, density=True, color = 'firebrick', label = 'simulation')
            
        #plot 80% CI for dist and hist_data
        ylim = plt.ylim()
        
        if (self.parameter.dist is not None) &('d' in self.mode):
            #80%CI fit
            ax.hlines(ylim[1]*1.1, xmin = self.parameter.dist.ppf(0.1), xmax = self.parameter.dist.ppf(0.9), color = 'lightcoral', linewidth = 3)
            ax.hlines(ylim[1]*1.1, xmin = self.parameter.dist.ppf(0.0001), xmax = self.parameter.dist.ppf(0.9999), color = 'lightcoral', linewidth = 3, linestyle = '--')
            p = [self.parameter.dist.ppf(0.0001),
                 self.parameter.dist.ppf(0.1),
                 self.parameter.s.mean(),
                 self.parameter.dist.ppf(0.9),
                 self.parameter.dist.ppf(0.9999)]
            ax.scatter(p, np.ones(len(p)) * ylim[1]*1.1, edgecolors = 'lightcoral', c = 'None', linewidths = 2)
        
        if (self.parameter.hist_data is not None)& ('h' in self.mode):
            #80% CI historical data
           ax.hlines(ylim[1]*1.2, xmin = np.quantile(self.parameter.hist_data, 0.1), xmax = np.quantile(self.parameter.hist_data, 0.9) , color = 'blue',  linewidth = 3)
           ax.hlines(ylim[1]*1.2, xmin = self.parameter.hist_data.min(), xmax = self.parameter.hist_data.max(), color = 'blue',  linewidth = 3, linestyle = '--')

           p = [self.parameter.hist_data.min(),
                np.quantile(self.parameter.hist_data, 0.1),
                self.parameter.hist_data.mean(),
                np.quantile(self.parameter.hist_data, 0.9),
                self.parameter.hist_data.max()]
           ax.scatter(p, np.ones(len(p)) * ylim[1]*1.2, edgecolors = 'blue', c = 'None', linewidths = 2)
            
        ax.set_title(f"Parameter: {self.parameter.name} [{self.parameter.unit}]")
        ax.set_xlim(xmin, xmax)
        ax.legend(loc = 7)
        ax.grid(True)
        
        if self.parameter.savefig:
            file = 'par_' + self.parameter.name + '.png'
            fig.savefig(Path(self.parameter.savedir) / 'parameters' / file, bbox_inches = 'tight', dpi = 250)
            
            
        
        plt.show()
        
        return
        
class result_visualizer():
    def __init__(self, result: result) -> None:
        self.result = result
        
    def plot(self):
        """
        plots a histogram of the results .s attribute

        Returns
        -------
        None.

        """
        
        xmin = min(self.result.s) * 0.9
        xmax = max(self.result.s) * 1.1
        fig, ax = plt.subplots()
        fig.set_size_inches(*self.result.figsize)  # 2*14 cm breit, 5.51, 3.9
        
        #histogram
        ax.hist(self.result.s, alpha=0.9, density=True, label = 'modelled', color = 'forestgreen')
        
        #80%CI
        ylim = plt.ylim()
        ax.hlines(ylim[1]*1.1, xmin = np.quantile(self.result.s, 0.1), xmax = np.quantile(self.result.s, 0.9), label = 'min, P90, mean, P10, max', color = 'forestgreen', linewidth= 3, alpha = 0.9)       
        ax.hlines(ylim[1]*1.1, xmin = self.result.s.min(), xmax = self.result.s.max(), color = 'forestgreen', linewidth= 3, alpha = 0.9, linestyle = '--')       

        p = [self.result.s.min(),
             np.quantile(self.result.s, 0.1),
             self.result.s.mean(),
             np.quantile(self.result.s, 0.9),
             self.result.s.max()]
        ax.scatter(p, np.ones(len(p)) * ylim[1]*1.1, edgecolors = 'forestgreen', c = 'None', linewidths = 2)
        
        ax.set_title(f"Result: {self.result.name} [{self.result.unit}]")
        ax.set_xlim(xmin, xmax)
        #ax.legend(loc = (1.04,0.5), frameon = True)
        ax.grid(True)

        plt.show()
        
        if self.result.savefig:
            file = 'par_' + self.result.name + '.png'
            fig.savefig(Path(self.result.savedir) / 'parameters' / file, bbox_inches = 'tight', dpi = 250)
        
        return 
        
class simulation_visualizer():
    
    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation
        
    def plot_pairplot(self, subset_par = None, subset_res = None, focus = False):
            """
            
    
            Parameters
            ----------
            subset_par : list, optional
                List of parameter names that should be plotted. If empty, all parameters are considered. The default is None.
            subset_res : list, optional
                ist of results names that should be plotted. If empty, all results are considered. The default is None.
            focus: narrow the x- and y lims of the plots to the 90% uncertainty intervals
    
            Returns
            -------
            None.
    
            """
    
            parameter_names = subset_par if subset_par is not None else self.simulation.parameter_names
            result_names = subset_res if subset_res is not None else self.simulation.result_names
            
            fig, ax = plt.subplots(len(parameter_names), len(result_names), sharex = 'col', sharey = 'row')
            fig.set_size_inches(2*5.51, len(parameter_names) * 2*5.51 / len(result_names))  # 2*14 cm breit
            
            if (len(parameter_names) <= 1) | (len(result_names) <= 1):
                ax = [ax]
            
            x = 0 #parameter rows
            y = 0 #result columns
            for param_name in parameter_names: #iterate through parameter rows
                 x = 0 #first result column
                 for res_name in result_names: #iterate through result columns
                     #plot data
                      
                     
                     #highest 10% 
                     _filter_90 = self.simulation[param_name].s >= np.quantile(self.simulation[param_name].s, 0.9)
                     ax[y][x].scatter(x = self.simulation[res_name].s[_filter_90], y = self.simulation[param_name].s[_filter_90], label = str(x) + '/' + str(y), 
                                       facecolors = 'lightcoral', edgecolors = 'none', s = 2)
                     
                     #lowest 10%
                     _filter_10 = self.simulation[param_name].s <= np.quantile(self.simulation[param_name].s, 0.1)
                     ax[y][x].scatter(x = self.simulation[res_name].s[_filter_10], y = self.simulation[param_name].s[_filter_10], label = str(x) + '/' + str(y), 
                                       facecolors = 'deepskyblue', edgecolors = 'none', s = 2)
                     #rest
                     _filter_rest = np.invert(_filter_10 + _filter_90)
                     ax[y][x].scatter(x = self.simulation[res_name].s[_filter_rest], y = self.simulation[param_name].s[_filter_rest], label = str(x) + '/' + str(y), 
                                       facecolors = 'grey', edgecolors = 'none', s = 2)
                      #plot mean of both attributes
                      #first, fix x and ylim
                     ax[y][x].set_xlim(ax[y][x].get_xlim())
                     ax[y][x].set_ylim(ax[y][x].get_ylim())
                     ax[y][x].hlines(self.simulation[param_name].s.mean(), xmin = ax[y][x].get_xlim()[0], xmax = ax[y][x].get_xlim()[1], color = 'black', lw = 0.7)
                     ax[y][x].vlines(self.simulation[res_name].s.mean(), ymin = ax[y][x].get_ylim()[0], ymax = ax[y][x].get_ylim()[1], color = 'black', lw = 0.7)
                    
                     if y == len(parameter_names)-1:
                         ax[y][x].set_xlabel(self.simulation[res_name].name, rotation = 45)
                     if x == 0:
                         ax[y][x].set_ylabel(self.simulation[param_name].name)
                           
                       
                      
                     ax[y][x].grid(visible = True)
                      
                     if focus:
                        ax[y][x].set_xlim([np.quantile(self.simulation[res_name].s, 0.05), np.quantile(self.simulation[res_name].s, 0.95)]) 
                        ax[y][x].set_ylim([np.quantile(self.simulation[param_name].s, 0.05), np.quantile(self.simulation[param_name].s, 0.95)]) 
                   
                     x += 1
                 y += 1
                 
            if self.simulation.settings['savefigs']:
                file = 'stairplot' + '.png'
                fig.savefig(Path(self.simulation.settings['savedir']) / 'qc and sensitivity' / file, bbox_inches = 'tight', dpi = 250)
            
    def plot_tornado(self, limit_parameters = True):
        '''
        creates a tornado plot for each result, showing sensitivity on all parameters.
        Sensitivities are computed as change in output mean.
        
        Parameters
        ----------
        limit_parameters : bool, optional
            only show parameters in tornado plot whose ciom is larger than the standard error of the results mean
        
        '''
        
        #span from p10 to p90
        span = ((self.simulation.ciom_p90 - self.simulation.ciom_p10)
                .abs())
        
        for result in self.simulation.results:
         
            fig, ax = plt.subplots()
            fig.set_size_inches(self.simulation.settings['figsize'])
            
            
            ax.set_title(f"Result: {result.name} [{result.unit}]")
            #compute standard error of result 
            std = np.std(result.s)
            ste = std / np.sqrt(self.simulation.settings['nmc'])
            
            #sorting order descending from span p90 - p10
            sorting_order = (span
                     .loc[(span.index == result.name)] 
                     .T
                     .query('`{}` > {}*{}'.format(result.name, 2 if limit_parameters else 0, ste)))
            
            sorting_order = (sorting_order
                     .assign(old_index = np.arange(len(sorting_order)))
                     .sort_values(by = result.name, ascending = False)
                     .reset_index()
                     .sort_values(by = 'old_index', ascending = True))
            
            sorting_order_index = sorting_order.index.to_numpy()
            
            #high parameter values
            ciom_p10t = (self.simulation.ciom_p10
                .loc[self.simulation.ciom_p10.index == result.name]
                .loc[:, sorting_order.iloc[:,0]]
                .T
                .assign(sorting_order = sorting_order_index)
                .sort_values(by = 'sorting_order'))
            
            #low parameter values            
            ciom_p90t = (self.simulation.ciom_p90
                .loc[self.simulation.ciom_p90.index == result.name]
                .loc[:, sorting_order.iloc[:,0]]
                .T
                .assign(sorting_order = sorting_order_index)
                .sort_values(by = 'sorting_order'))


            
            names = ciom_p10t.index
            
            ciom_p10t = np.squeeze(ciom_p10t[result.name].to_numpy())
            ciom_p90t = np.squeeze(ciom_p90t[result.name].to_numpy())
            
            base_value = result.s.mean()
            
            #compute left parameter for barh (P10)
            left = np.ones(len(ciom_p10t))
            left[ciom_p10t < 0] = base_value + ciom_p10t[ciom_p10t < 0]
            left[ciom_p10t >= 0] = base_value
            
            
            #plot ciom_p10
            ax.barh(y = np.flip(np.arange(0,stop = len(ciom_p10t))), width = abs(ciom_p10t) , left =  left, tick_label = names, alpha = 0.7, color = 'lightcoral', label = 'high')
            
            #compute left parameter for barh (P90)
            left = np.ones(len(ciom_p90t))
            left[ciom_p90t < 0] = base_value + ciom_p90t[ciom_p90t < 0]
            left[ciom_p90t >= 0] = base_value
            
            
            #plot ciom_p90
            ax.barh(y = np.flip(np.arange(0,stop = len(ciom_p90t))), width = abs(ciom_p90t) , left =  left,  tick_label = names, alpha = 0.7, color = 'deepskyblue', label = 'low')     
            
            #plot standard error of result in both directions
            ax.axvspan(base_value - ste, base_value + ste, color = 'white', alpha = 0.5, label = 'standard error')
            
            #plot baseline
            ax.vlines(x = base_value, ymin = ax.get_ylim()[0],  ymax = ax.get_ylim()[1], ls = '--', color = 'black', label = 'base value')
    
            plt.box(True)

            ax.set_xlim(base_value - (base_value - ax.get_xlim()[0]) * 1.1, ax.get_xlim()[1]) #pyplot cuts off the plot on the left side...
            plt.legend(frameon = True, loc = 4)
            
            plt.show()
            
            if self.simulation.settings['savefigs']:
                file = 'tornado' + result.name + '.png'
                fig.savefig(Path(self.simulation.settings['savedir']) / 'results' / file, bbox_inches = 'tight', dpi = 250)    

    def plot_convergence(self, plot = True):
            """
            evaluate the convergeence of an element in a simulation. Convergence is indicated by the reduction of 
            the standard error of the running average.
    
            Parameters
            ----------
            plot : bool, optional
                show a plot of the evolution of all parameter means
    
            """
            for result in self.simulation.results:
                #print
                print('Element {name}: Standard error of the mean after the last iteration is {se_abs: .2f} or {se_rel: .2f} %'.format(name = result.name, 
                                                                                                                          se_abs = result.running_se[-1], 
                                                                                                                          se_rel = result.running_se[-1] / result.running_average[-1] * 100))
                
            #plot 
            if plot:
                n = math.ceil(len(self.simulation.results) / 2)
                fig, ax = plt.subplots(n, 2, squeeze=True, sharex = True)
                ax = ax.flatten()
                fig.set_size_inches(2*5.51, n/2*5)  # 2*14 cm breit
            
                i = 0
            
                for result in self.simulation.results:
            
                    rsp = result.running_average + result.running_se
                    rsn = result.running_average - result.running_se
            
                    # plot result of each iteration
                    ax[i].plot(result.s, alpha=0.4, color='grey')
                    
                    # plot running average
                    ax[i].plot(result.running_average, color = 'blue')
                    
                    #plot area between -se and +se
                    ax[i].fill_between(np.arange(0,len(result.s),1), rsn, rsp, facecolor = 'blue', alpha = 0.6)
                    
                    #textbox: standard error of mean
                    ax[i].set_title('Result: ' + result.name)
                    fig.tight_layout()
                    
                    ax[i].set_xlabel('iteration')
                    ax[i].set_ylabel('value')
                    
                    
            
                    i = i+1
                    
                    if self.simulation.settings['savefigs']:
                        file = 'parameter convergence.png'
                        fig.savefig(Path(self.simulation.settings['savedir']) / 'qc and sensitivity' / file, bbox_inches = 'tight', dpi = 250)    
class simulation_analyzer():
    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        
    def compute_ciom(self):
             """
             
             correlates every result with every parameter regarding change in output mean for the P10 and P90+
             of the parameter.
             
             Parameters
             ----------
    
        
             Returns
             -------
             ciom_P10: Pandas dataframe with results as columns, parameters as rows, change in output mean of result for the lowest 10% of the result values
             ciom_P90: Pandas dataframe with results as columns, parameters as rows, change in output mean of result for the highest 10% of the result values
        
             """
    
        
             ciom_P10 = pd.DataFrame([], index = self.simulation.result_names, columns = self.simulation.parameter_names)
             ciom_P90 = pd.DataFrame([], index = self.simulation.result_names, columns = self.simulation.parameter_names)
             
             for result in self.simulation.results:
                  
                  #merge result and parameters
                  d = np.vstack([result.s, np.vstack([param.s for param in self.simulation.parameters])]).T
                  d = pd.DataFrame(d, columns = np.hstack(['res', self.simulation.parameter_names]))
                  
                  #sort by each parameter, compute result mean for Q10 and Q90
                  mean_p10 = [np.mean(d.loc[(d[param.name] <= np.quantile(d[param.name], 0.9)), 'res']) for param in self.simulation.parameters]
                  
                  mean_p90 = [np.mean(d.loc[(d[param.name] >= np.quantile(d[param.name], 0.1)), 'res']) for param in self.simulation.parameters]
                  
                  ciom_P10.loc[ciom_P90.index == result.name, :] = mean_p90 - d['res'].mean()
                  ciom_P90.loc[ciom_P10.index == result.name, :] = mean_p10 - d['res'].mean()
                  
             return ciom_P10, ciom_P90
       
    def correlate(self):
         """
         correlates every parameter to every result (using spearman and pearson correlation)

         Returns
         -------
         r2 : pd.DataFrame (rows:parameters, cols: results, values: R²)
         spearmansr : pd.DataFrame (rows:parameters, cols: results, values: spearmans rho)
         coeff1 : pd.DataFrame (rows:parameters, cols: results, values: intercept)
         coeff2 : pd.DataFrame (rows:parameters, cols: results, values: slope)

         """
         r2 = []
         spearmansr = []
         pearsonr = []
         coeff1 = []
         coeff2 = []
         for param in self.simulation.parameters:
            #fit parameter vs. result for each result (array of n(ResultVal) tupels)
            pfit_t = [np.polyfit(param.s, res.s, 1, full = True) for res in self.simulation.results]
            
            
            #determine spearman rho
            spearmansr_t = np.array([stats.spearmanr(param.s, res.s)[0] for res in self.simulation.results])
            pearsonr_t = np.array([stats.pearsonr(param.s, res.s)[0] for res in self.simulation.results])
            #get linear regression coefficients
            
            coeff2_t = np.array([pfit[0][0] for pfit in pfit_t])
            coeff1_t = np.array([pfit[0][1] for pfit in pfit_t])
            
            
        
               
              #add to coeff1
            if isinstance(coeff1, np.ndarray):
                 coeff1 = np.vstack([coeff1, coeff1_t])
            else:
                 coeff1 = coeff1_t
             
            #add to coeff2
            if isinstance(coeff2, np.ndarray):
                 coeff2 = np.vstack([coeff2, coeff2_t])
            else:
                 coeff2 = coeff2_t
                 
            #add to spearmansr
            if isinstance(spearmansr, np.ndarray):
                 spearmansr = np.vstack([spearmansr, spearmansr_t])
            else:
                 spearmansr = spearmansr_t
    
            #add to pearsonr
            if isinstance(pearsonr, np.ndarray):
                 pearsonr = np.vstack([pearsonr, pearsonr_t])
            else:
                 pearsonr = pearsonr_t
               
         
         #convert r2, spearmansr, pearsonr, coeff1, coeff2 to pd.DataFrame 
         #coeff1 = pd.DataFrame(coeff1, index = param_names, columns=(res_names)) 
         #coeff2 = pd.DataFrame(coeff2, index = param_names, columns=(res_names))  
         spearmansr = pd.DataFrame(spearmansr, index = self.simulation.parameter_names, columns=(self.simulation.result_names))
         pearsonr = pd.DataFrame(pearsonr, index = self.simulation.parameter_names, columns=(self.simulation.result_names))   
           
         return pearsonr, spearmansr#, coeff1, coeff2    
    
