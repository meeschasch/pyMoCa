#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 18:49:00 2021

@author: mischasch

TODO:
    - units (check)
    - properties for simulation (e.g. P-Fälle, savefigs etc.) (check)
    - parameter and result inherit from "element" (nicht sinnvoll?)
    - P Fälle dynamisch
    - convergence: text im plot
"""

#imports
import pandas as pd, scipy.stats as stats, numpy as np, re, math, os
import matplotlib.pyplot as plt, seaborn as sns, pickle, warnings
from pathlib import Path
from matplotlib import style

#plot presets
#plt.rcParams.update({'font.size': 14})ß
plt.rcParams.update({'figure.dpi': 100})
plt.rcParams.update({'legend.loc': 'best'})
style.use('seaborn')

#parent class for simulation elements
class simulation_element():
    def __init__(self, name, s, unit = '-'):
        '''
        

        Parameters
        ----------
        name : TYPE
            DESCRIPTION.
        s : TYPE
            DESCRIPTION.
        unit : TYPE, optional
            DESCRIPTION. The default is '-'.

        Returns
        -------
        None.

        '''
        self.name = name
        self.unit = unit
        self.s = s
        self.figsize = (2, 1.618*2) #standard figszie in jupyter notebook
        self.savefig = False
        self.savedir = None
        
# =============================================================================
#  General methods       
# =============================================================================
    # def evaluate_convergence(self):
    #     """
    #     evaluates the convergence of a simulation element by computing and setting the attributes self.runningAverage and 
    #     self.runningStandardError. 

    #     Returns
    #     -------
    #     None.

    #     """
    #     #check ihf simulation data is available
    #     if self.s is None:
    #         raise Exception('No simulated data available to analyse')
        
    #     self.running_average = self.compute_running_average(self.s)
    #     self.running_se = self.compute_running_se(self.s)
        
    #     print('Element: {} ... Standard Error of the mean after the last iteration is {:.2f} or {:.2f} %%'
    #           .format(self.name, self.running_se[-1],
    #                   self.running_se[-1] / self.running_average[-1]*100))
        
    def compute_running_average(self):
        '''
        
       
        Parameters
        ----------
        array : np.array
             DESCRIPTION: array of numbers on which to compute the running average
        Returns
        -------
        ra : TYPE
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
        computes a running standard error (std / sqrt(n))
       
            Parameters
        ----------
        array : np.array
             list in correct order for which to calculate the running quantile
        quantile : float
             quantile to compute
       
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
    
# =============================================================================
# Methods that will be overridden in subclasses        
# =============================================================================
    def plot(self):
        return
    
    def compile_summary(self):
        return 

    
        
class parameter(simulation_element):
    def __init__(self, name, s = None, unit = '-', dist = None, hist_data = None):
        '''
        

        Parameters
        ----------
        name : TYPE
            DESCRIPTION.
         : TYPE
            DESCRIPTION.
        s : TYPE, optional
            DESCRIPTION. The default is None.
        unit : TYPE, optional
            DESCRIPTION. The default is '-'.
        dist : TYPE, optional
            DESCRIPTION. The default is None.
        hist_data : TYPE, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        #initialise parent class
        super().__init__(name, s, unit)
        
        
        #attributes of derived class
        self.dist = dist
        self.hist_data = hist_data
        
        
    def plot(self, mode = 'dhs', showfig = True):
        """
        creates and shows a plot of the parameter. In any case, the theoretical PDF of the distribution is displayed. 
        If available, historical data as well as randomly drawn data for the MC simulation is displayed.

        Parameters
        ----------
        mode : string
            Plots the parameter. Default: 'd'. The submitted string can be a combination of the followiung characters:
            'd': distribution
            'h': historic data
            's': simulation data
        showfig: If True, plot is showed
            
        Returns
        -------
        None.

        """
        
        #cehck whether mode is valid
        if re.search('[^dhs]', mode) is not None:
            raise ValueError('mode string contains characters other than dhs')
        
        fig, ax = plt.subplots()
        fig.set_size_inches(*self.figsize)
        
    
        
        if self.hist_data is not None:
            xmin = min(min(self.hist_data), self.dist.ppf(0.01)) * 0.9
            xmax = max(max(self.hist_data), self.dist.ppf(0.99)) * 1.1
        elif self.dist is not None:
            xmin = self.dist.ppf(0.01) * 0.9
            xmax = self.dist.ppf(0.99) * 1.1
        else:
            xmin = min(self.s) * 0.9
            xmax = max(self.s) * 1.1
                
        
        x = np.linspace(xmin, xmax, 100)
        #nbins = 100
        
        #print distribution
        if 'd' in mode:
            #pdf
            if self.dist is not None:
                ax.plot(x, self.dist.pdf(x), color = 'lightcoral', lw = 2, label = 'fit')
           
        #print historical data
        if (self.hist_data is not None)& ('h' in mode):
           #histogram
           ax.hist(self.hist_data, alpha=0.9, density=True, color = 'blue', label = 'observed')
           
        #print simulation data
        if (self.s is not None) & ('s' in mode):
            ax.hist(self.s, alpha=0.2, density=True, color = 'firebrick', label = 'simulation')
            
        #plot 80% CI for dist and hist_data
        ylim = plt.ylim()
        
        if (self.dist is not None) &('d' in mode):
            #80%CI fit
            ax.hlines(ylim[1]*1.1, xmin = self.dist.ppf(0.1), xmax = self.dist.ppf(0.9), label = 'min, P90, mean, P10, max - sim.', color = 'lightcoral', linewidth = 3)
            ax.hlines(ylim[1]*1.1, xmin = self.dist.ppf(0.0001), xmax = self.dist.ppf(0.9999), color = 'lightcoral', linewidth = 3, linestyle = '--')
            p = [self.dist.ppf(0.0001),
                 self.dist.ppf(0.1),
                 self.s.mean(),
                 self.dist.ppf(0.9),
                 self.dist.ppf(0.9999)]
            ax.scatter(p, np.ones(len(p)) * ylim[1]*1.1, edgecolors = 'lightcoral', c = 'None', linewidths = 2)
        
        if (self.hist_data is not None)& ('h' in mode):
            #80% CI historical data
           ax.hlines(ylim[1]*1.2, xmin = np.quantile(self.hist_data, 0.1), xmax = np.quantile(self.hist_data, 0.9), label = 'min, P90, mean, P10, max - obs.', color = 'blue',  linewidth = 3)
           ax.hlines(ylim[1]*1.2, xmin = self.hist_data.min(), xmax = self.hist_data.max(), color = 'blue',  linewidth = 3, linestyle = '--')

           p = [self.hist_data.min(),
                np.quantile(self.hist_data, 0.1),
                self.hist_data.mean(),
                np.quantile(self.hist_data, 0.9),
                self.hist_data.max()]
           ax.scatter(p, np.ones(len(p)) * ylim[1]*1.2, edgecolors = 'blue', c = 'None', linewidths = 2)
            
        ax.set_title(f"Parameter: {self.name} [{self.unit}]")
        ax.set_xlim(xmin, xmax)
        #ax.legend(loc = (1.04,0.5), frameon = True)
        ax.grid(True)
        
        if self.savefig:
            file = 'par_' + self.name + '.png'
            fig.savefig(Path(self.savedir) / 'parameters' / file, bbox_inches = 'tight', dpi = 250)
            
            
        
        plt.show()
        
        return #fig
    
    def compile_summary(self):
        """
        compiles a summary table conatining a statistical description of the input parameter.
        If available, balso the historical data is described.

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
        0 if succesful, if nmc <1 1, the function returns -1.

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
        plots the result

        Returns
        -------
        None.

        """
        
        xmin = min(self.s) * 0.9
        xmax = max(self.s) * 1.1
        fig, ax = plt.subplots()
        fig.set_size_inches(*self.figsize)  # 2*14 cm breit, 5.51, 3.9
        
        #histogram
        ax.hist(self.s, alpha=0.9, density=True, label = 'modelled', color = 'forestgreen')
        
        #80%CI
        ylim = plt.ylim()
        ax.hlines(ylim[1]*1.1, xmin = np.quantile(self.s, 0.1), xmax = np.quantile(self.s, 0.9), label = 'min, P90, mean, P10, max', color = 'forestgreen', linewidth= 3, alpha = 0.9)       
        ax.hlines(ylim[1]*1.1, xmin = self.s.min(), xmax = self.s.max(), color = 'forestgreen', linewidth= 3, alpha = 0.9, linestyle = '--')       

        p = [self.s.min(),
             np.quantile(self.s, 0.1),
             self.s.mean(),
             np.quantile(self.s, 0.9),
             self.s.max()]
        ax.scatter(p, np.ones(len(p)) * ylim[1]*1.1, edgecolors = 'forestgreen', c = 'None', linewidths = 2)
        
        ax.set_title(f"Result: {self.name} [{self.unit}]")
        ax.set_xlim(xmin, xmax)
        #ax.legend(loc = (1.04,0.5), frameon = True)
        ax.grid(True)

        plt.show()
        
        if self.savefig:
            file = 'par_' + self.name + '.png'
            fig.savefig(Path(self.savedir) / 'parameters' / file, bbox_inches = 'tight', dpi = 250)
        
        return #fig
    
    
    def compile_summary(self):
        """
        compiles a summary table conatining a statistical description of the result.

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
        safeplots: If True, all generated plots are saved in savedir
        figsize: widthxlength in inches (applies to most figures)

        Returns
        -------
        None.

        """
        #self.name = name
        self.__parameters = []
        self.__results = []

        
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
            for param in self.__parameters:
                if param.name == index:
                    return param
        elif index in self.result_names:
            for result in self.__results:
                if result.name == index:
                    return result
        else:
            raise ValueError(index + ' not in parameters or results')
            
    def __setitem__(self, key, newvalue):
        #only continue if new item is a result or a paraeter
        if not isinstance(newvalue, (parameter, result)):
            raise ValueError('New item must bei either a parameter or a result')
         
        #adding new elements using __setitem__ not allowed
        if newvalue.name not in self.__parameters + self.__results:
            raise ValueError('Use addParameter() or addresult() to add new elements')
            
        if isinstance(newvalue, parameter): #it's a parameter
            #remove existing entry
            if key in self.parameter_names: #parameter already exists
                for par in self.__parameters:
                    if par.name == key:
                        warnings.warn('Parameter already exsists and is replaced with the new one')
                        self.__parameters.remove(par) #remove existing parameter
            
                #add new
                self.__parameters.append(newvalue)
            
        elif isinstance(newvalue, result):
            #remove existing entry
            if key in self.result_names:
                for res in self.__results:
                    if res.name == key:
                        warnings.warn('result already exsists and is replaced with the new one')
                        self.__results.remove(res)
            
                #add new
                self.__results.append(newvalue)

            
        
            
    def __create_savedirs(self):
        dirs = ['parameters', 'results', 'qc and sensitivity']
        
        for diri in dirs:
            path = self.settings['savedir'] / diri
            if not os.path.isdir(path):
                os.makedirs(path)
        
        
    def set_savedir(self, savedir):
        self.settings['savedir'] = Path(savedir)
        self.__create_savedirs()
        
        return
    
    #properties
    @property
    def parameter_names(self):
        return [par.name for par in self.__parameters]
    
    @property
    def result_names(self):
        return [res.name for res in self.__results]
    
    @property
    def summary(self):
        return self.__summary()
    
    def add(self, simulation_element):
        '''
        

        Parameters
        ----------
        simulation_element : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        try:
            self.__validate_simulation_element(simulation_element)
        except:
            raise ValueError('ELement could not be added')
        
        #pass on simulation wide parameters to simulation elements
        simulation_element.figsize = self.settings['figsize'] 
        simulation_element.savefig = self.settings['savefigs']
        simulation_element.savedir = self.settings['savedir']
        
        if isinstance(simulation_element, parameter):
            self.__parameters.append(simulation_element)
        elif isinstance(simulation_element, result):
            self.__results.append(simulation_element)

    def remove(self, name):
        '''
        removes an element (identified by its name) from the simulation.

        Parameters
        ----------
        remove_element : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if name in self.parameter_names:
            for p in self.__parameters:
                if p.name == name:
                    self.__parameters.remove(p)
                    
        elif name in self.result_names:
            for r in self.__results:
                if r.name == name:
                    self.__results.remove(r)
        else:
            raise Exception('No element with this name in the simulation')
        
    def __validate_simulation_element(self, new_element):
        '''
        validates a new simulation element before added / altered.

        Parameters
        ----------
        simulation_element : TYPE
            DESCRIPTION.

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
            
    # def addParameter(self, param):
    #     """
    #     add a parameter to the simulation.

    #     Parameters
    #     ----------
    #     param : mcoo.Param
    #         Dthe parameter

    #     Returns
    #     -------
    #     None

    #     """
    #     #check if param is an mcoo.Param object
    #     if not isinstance(param, Param):
    #         raise ValueError('Parameter is not an mcoo.Param object')
        
    #     #check if already exists
    #     if param.name in self.result_names + self.parameter_names:
    #         raise ValueError('Element of that name already present')
        
    #     self.__parameters.append(param)
        
        
        
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
        for param in self.__parameters:
            s = param.realise(self.settings['nmc'], overwrite_existing = overwrite_existing)
                
        
                
    # def addResult(self, result):
    #     """
    #     adds a result to the simulation.

    #     Parameters
    #     ----------
    #     name : string
    #         name of the result
    #     results : array of numbers
    #         array of results. Length of vector must equal lenght of parameters.
            
    #     Returns
    #     -------
    #     None.
    #     """
        
    #     if not isinstance(result, Result):
    #         raise ValueError('Result is not an mcoo.Param object')
            
    #     #check if already exists
    #     if result.name in np.hstack([self.parameter_names, self.result_names]):
    #         raise ValueError('Element of that name already present')
            
    #     if not len(result.s) == self.settings['nmc']:
    #         raise Exception('Length of result does not match simulation number of cases')
            
    #     self.__results.append(result)
        
    #     #TODO: list result_namesx
        
        
    def __summary(self):
        """
        Statistical description of all simulation parameters and results

        Returns
        -------
        pandas.DataFrame with parameters and results as rows and properties as columns

        """
        #compose dataframe
        cols =  ['name', 'type', 'unit', 'mean', 'std', 'min', 'max', 'P90', 'P50', 'P10']
        # index = [param.name for param in self.parameters] + [result.name for result in self.results]
        # sets = InputParamVal + ResultVal
        # types = ['parameter' if seti[0] in [par[0] for par in InputParamVal] else 'result' for seti in sets]
        s = pd.DataFrame(columns = cols)
        s.set_index('name', inplace = True)

        #add parameters
        for param in self.__parameters:
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
        for result in self.__results:
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
    
    def dump(self):
        '''
        returns: pandas DataFrame containing all parameters and all results for each realisation
        '''
        colsp = [parami.name for parami in self.__parameters]
        colsr= [resi.name for resi in self.__results]
        
        dump = pd.DataFrame(columns = np.hstack([colsp, colsr]))
        
        for column in dump.columns:
            dump[column] = self[column].s
        return dump
    
    def convergence(self, plot = True):
        """
        evaluate the convergeence of an element in a simulation. Convergence is indicated by the reduction of 
        the standard error of the running average.

        Parameters
        ----------
        plot : bool, optional
            show a plot of the evolution of all parameter means
        text : bool, optional
            describe the convergence of all parameters in text

        Returns
        -------
        None.

        """
        for result in self.__results:
            #print
            print('Element {name}: Standard error of the mean after the last iteration is {se_abs: .2f} or {se_rel: .2f} %'.format(name = result.name, 
                                                                                                                      se_abs = result.running_se[-1], 
                                                                                                                      se_rel = result.running_se[-1] / result.running_average[-1] * 100))
            
        #plot 
        if plot:
            n = math.ceil(len(self.__results) / 2)
            fig, ax = plt.subplots(n, 2, squeeze=True, sharex = True)
            ax = ax.flatten()
            fig.set_size_inches(2*5.51, n/2*5)  # 2*14 cm breit
        
            i = 0
        
            for result in self.__results:
        
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
                
                if self.settings['savefigs']:
                    file = 'parameter convergence.png'
                    fig.savefig(Path(self.settings['savedir']) / 'qc and sensitivity' / file, bbox_inches = 'tight', dpi = 250)
                
    def sensitivity(self, pairplot = True, tornado_ciom = True):
        """
        

        Parameters
        ----------
        pairplot : boolean, optional
            plot a pairplot of all parameter / result combination. The default is True.
        tornado_ciom : boolean, optional
            plot a tornade diagram based on change in output mean correlation. The default is True.

        Returns
        -------
        Two pandas.DataFrames with preconfigured display style
        separmansr: pd.DataFrme containing Spearman's correlation coefficient between parameters and results.
        pearsonr:pd.DataFrme containing Pearson's correlation coefficient between parameters and results.

        """

        #get R² and linear fit parameters between all parameters and results
        self.pearsonr, self.spearmansr, self.coeff1, self.coeff2= self._correlate_r()
        
        #get change in output mean P10 and P90 matrixes
        self.ciom_p10, self.ciom_p90 = self._corelate_ciom()
        
        #Plot a Pairplot
        if pairplot:
             self._plot_pairplot()
             
        if tornado_ciom:
              self._plotTornado_ciom()
        
        #adjust display style and return pd.DataFrames for correlation coefficients
        #pandas display options
        pd.options.display.precision = 2
        cm = sns.color_palette("vlag", as_cmap=True)
        
        self.pearsonr.style.background_gradient(cmap=cm) 
        self.spearmansr.style.background_gradient(cmap=cm) 
        
        
        return self.spearmansr, self.pearsonr
    
    
    def correlate(self):
         """
         correlates every parameter to every result

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
         for param in self.__parameters:
            #fit parameter vs. result for each result (array of n(ResultVal) tupels)
            pfit_t = [np.polyfit(param.s, res.s, 1, full = True) for res in self.__results]
            
            
            #determine spearman rho
            spearmansr_t = np.array([stats.spearmanr(param.s, res.s)[0] for res in self.__results])
            pearsonr_t = np.array([stats.pearsonr(param.s, res.s)[0] for res in self.__results])
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
               
               
               
               
               

         res_names = [res.name for res in self.__results]
         param_names = [param.name for param in self.__parameters]
         
         #convert r2, spearmansr, pearsonr, coeff1, coeff2 to pd.DataFrame 
         coeff1 = pd.DataFrame(coeff1, index = param_names, columns=(res_names)) 
         coeff2 = pd.DataFrame(coeff2, index = param_names, columns=(res_names))  
         spearmansr = pd.DataFrame(spearmansr, index = param_names, columns=(res_names))
         pearsonr = pd.DataFrame(pearsonr, index = param_names, columns=(res_names))   
           
         return pearsonr, spearmansr, coeff1, coeff2
     

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

    
         ciom_P10 = pd.DataFrame([], index = self.result_names, columns = self.parameter_names)
         ciom_P90 = pd.DataFrame([], index = self.result_names, columns = self.parameter_names)
         
         for result in self.__results:
              
              #merge result and parameters
              d = np.vstack([result.s, np.vstack([param.s for param in self.__parameters])]).T
              d = pd.DataFrame(d, columns = np.hstack(['res', self.parameter_names]))
              
              #sort by each parameter, compute result mean for Q10 and Q90
              mean_p10 = [np.mean(d.loc[(d[param.name] <= np.quantile(d[param.name], 0.9)), 'res']) for param in self.__parameters]
              
              mean_p90 = [np.mean(d.loc[(d[param.name] >= np.quantile(d[param.name], 0.1)), 'res']) for param in self.__parameters]
              
              ciom_P10.loc[ciom_P90.index == result.name, :] = mean_p90 - d['res'].mean()
              ciom_P90.loc[ciom_P10.index == result.name, :] = mean_p10 - d['res'].mean()
              
         return ciom_P10, ciom_P90
     
   
    def plot_pairplot(self, subset_par = None, subset_res = None, focus = True):
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

        parameter_names = subset_par if subset_par is not None else self.parameter_names
        result_names = subset_res if subset_res is not None else self.result_names
        
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
                 _filter_90 = self[param_name].s >= np.quantile(self[param_name].s, 0.9)
                 ax[y][x].scatter(x = self[res_name].s[_filter_90], y = self[param_name].s[_filter_90], label = str(x) + '/' + str(y), 
                                   facecolors = 'lightcoral', edgecolors = 'none', s = 2)
                 
                 #lowest 10%
                 _filter_10 = self[param_name].s <= np.quantile(self[param_name].s, 0.1)
                 ax[y][x].scatter(x = self[res_name].s[_filter_10], y = self[param_name].s[_filter_10], label = str(x) + '/' + str(y), 
                                   facecolors = 'deepskyblue', edgecolors = 'none', s = 2)
                 #rest
                 _filter_rest = np.invert(_filter_10 + _filter_90)
                 ax[y][x].scatter(x = self[res_name].s[_filter_rest], y = self[param_name].s[_filter_rest], label = str(x) + '/' + str(y), 
                                   facecolors = 'grey', edgecolors = 'none', s = 2)
                  #plot mean of both attributes
                  #first, fix x and ylim
                 ax[y][x].set_xlim(ax[y][x].get_xlim())
                 ax[y][x].set_ylim(ax[y][x].get_ylim())
                 ax[y][x].hlines(self[param_name].s.mean(), xmin = ax[y][x].get_xlim()[0], xmax = ax[y][x].get_xlim()[1], color = 'black', lw = 0.7)
                 ax[y][x].vlines(self[res_name].s.mean(), ymin = ax[y][x].get_ylim()[0], ymax = ax[y][x].get_ylim()[1], color = 'black', lw = 0.7)
                
                 if y == len(parameter_names)-1:
                     ax[y][x].set_xlabel(self[res_name].name, rotation = 45)
                 if x == 0:
                     ax[y][x].set_ylabel(self[param_name].name)
                       
                   
                  
                 ax[y][x].grid(visible = True)
                  
                 if focus:
                    ax[y][x].set_xlim([np.quantile(self[res_name].s, 0.05), np.quantile(self[res_name].s, 0.95)]) 
                    ax[y][x].set_ylim([np.quantile(self[param_name].s, 0.05), np.quantile(self[param_name].s, 0.95)]) 
               
                 x += 1
             y += 1
             
        if self.settings['savefigs']:
            file = 'stairplot' + '.png'
            fig.savefig(Path(self.settings['savedir']) / 'qc and sensitivity' / file, bbox_inches = 'tight', dpi = 250)
            
    def plot_tornado(self):
        '''
        creates a tornado plot for each result, showing sensitivity on all parameters.
        Sensitivities are computed as change in output mean.
        '''
        ciom_p10, ciom_p90 = self.compute_ciom()
        
        #span from p10 to p90
        span = ((ciom_p90 - ciom_p10)
                .abs())
        
        for result in self.__results:
         
            fig, ax = plt.subplots()
            fig.set_size_inches(self.settings['figsize'])
            
            
            ax.set_title(f"Result: {result.name} [{result.unit}]")

            #sorting order descending from span p90 - p10
            sorting_order = (span
                     .loc[span.index == result.name]
                     .T
                     .assign(old_index = np.arange(span.shape[1]))
                     .sort_values(by = result.name, ascending = False)
                     .reset_index()
                     .sort_values(by = 'old_index', ascending = True)
                     .index
                     .to_numpy())
            
            #high parameter values
            ciom_p10t = (ciom_p10
                .loc[ciom_p10.index == result.name]
                .T
                .assign(sorting_order = sorting_order)
                .sort_values(by = 'sorting_order'))
            
            #low parameter values            
            ciom_p90t = (ciom_p90
                .loc[ciom_p90.index == result.name]
                .T
                .assign(sorting_order = sorting_order)
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
            
            #compute and plot standard error of result in both directions
            std = np.std(result.s)
            ste = std / np.sqrt(self.settings['nmc'])
            ax.axvspan(base_value - ste, base_value + ste, color = 'white', alpha = 0.5, label = 'standard error')
            
            #plot baseline
            ax.vlines(x = base_value, ymin = ax.get_ylim()[0],  ymax = ax.get_ylim()[1], ls = '--', color = 'black', label = 'base value')
    
            plt.box(True)

            ax.set_xlim(base_value - (base_value - ax.get_xlim()[0]) * 1.1, ax.get_xlim()[1]) #pyplot cuts off the plot on the left side...
            plt.legend(frameon = True, loc = 4)
            
            plt.show()
            
            if self.settings['savefigs']:
                file = 'tornado' + result.name + '.png'
                fig.savefig(Path(self.settings['savedir']) / 'results' / file, bbox_inches = 'tight', dpi = 250)
            
          
    def show_parameters(self, mode = 'sdh', showtable = True, showfig = True):
        """
        creates a plot for each parameter.

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
        for param in self.__parameters:
            fig = param.plot(mode, showfig = showfig)
            
            if showtable:
                r = param.summary
                display(r)
                
            if self.settings['savefigs']:
                file = 'par_' + param.name + '.png'
                fig.savefig(Path(self.settings['savedir']) / 'parameters' / file, bbox_inches = 'tight', dpi = 250)
                
            
    def show_results(self, showtable = True, showfig = True):
        """
        creates a plot for each result
        
        Parameters
        --------
        showtable: If True, shows summary table after each plot
        showfig: If True (Default), shows each plot

        Returns
        -------
        None.

        """
        for result in self.__results:
            fig = result.plot(showfig = showfig)
            if showtable:
                display(result.summary)
                
            if self.settings['savefigs']:
                file = 'res_' + result.name + '.png'
                fig.savefig(Path(self.settings['savedir']) / 'results' / file, bbox_inches = 'tight', dpi = 250)
        
    def savesim(self):
        """
        

        Returns
        -------
        None.

        """
        file = self.settings['savedir'] / (self.name  + '.pickle')
        with open(file, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        

        
        
        
