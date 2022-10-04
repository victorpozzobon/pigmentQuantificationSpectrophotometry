# -*- coding: utf-8 -*-
"""
PSO nested in a loop with increasing number of wavelength / components

@author: Victor Pozzobon
Article
Lutein, violaxanthin, and zeaxanthin spectrophotometric quantification: a machine learning approach
Pozzobon, V., & Camarena-Bernard, C. (2022).
Journal of Applied Phycology, pp-pp.

"""
### Loading modules ###
import time
startTime = time.time()
startTime_string = time.localtime()

import numpy as np
import multiprocessing as mp
from PSO_GA_AssistantFunctions import processLargeLoader, saveBest, ObjectiveFunction
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt # <= to plot the results
plt.close('all')
import os
import sys

#%%% --- Parameters --- ###

description = "PSO screening of the wavelength, run multiple times in a row"
screeningMode = True # if False, it does only one run, should be used to identify the best wavelengths once their number is determined
nrunTotal = 100           # number of run for the sreening
nbWLVector = np.arange(1,10+1,1) # including up to 10 wabvelengths
nbParticleOneTime = 1000000 

#%%% --- Data loading --- ###
TrainData = np.load("Data/Pigment_Spectra_train.npy")
ValidData = np.load("Data/Pigment_Spectra_valid.npy")
meanStdBest = []

if not screeningMode:
    nrunTotal = 1
    nbWLVector = [6]
else:
    os.system('rm -r Screening') # <= Linux command, please modify if you are running under windows
    os.system('mkdir Screening')


# Loop repeating the procedure nrunTotal times
for nrun in range(1, nrunTotal+1):
    if screeningMode:
        os.system('mkdir Screening/Screening_' + str(nrun))
    
    # Testing the different number of wavelength (1 to 10), if screeningMode = True
    for nbWL in nbWLVector:
        startTime_string = time.localtime()
        # Tuning PSO
        nbParticle = nbWL * 5000                # Number of particles, I know it should be a power function, but ... way too expensive // PSO
        if not screeningMode: nbParticle = nbParticleOneTime
        nbParam = nbWL                      # Number of paramenters (= nb activated wavelength here)
        plageUp   = (461+461-3-100) * np.ones(nbParam)  # Upper bound of the range 
        plageDown = 0 * plageUp            # Lower bound of the range 
        nbitermax = 10000                 # Maximum number of iteration 
        c1 = 0.6                        # Cognitive swarm parameter
        c2 = 0.6                        # Social swarm parameter 
        inertie = 0.8                   # Intertia  //  overriden by chaotic one
        depMax = 46.1                        # Velocity capping 
                                            # ex: 100 => 1/100 of the range
        sampling = "random"             # Search-space initial sampling method : "uniform" (not available here), "random"
        plotPeriod = 1000               # Plot results every plotPeriod 
        procNb = 8                      # Number of tread available for parallel runs
        maxStagnation = 50              # Number of iretation which stagnate before stopping the search
        
        # Some verification and declaration
        if min(plageUp - plageDown) < 0:
            print ("Check ranges!")
            sys.exit()
            
        maxObj = 1e2                     # Unlikely to be met maximum of the cost function
        Improvement = maxObj             # Storage to compute improvement from one iteration to the other
        OldMax = maxObj                  # Former best results
        StagnationCounter = 0            # Counter for stagnation
        
        #%%% --- Initialization --- ###
        
        ### PSO Initialization ###
        # Columns of m contains the p paramater and the cost function value f(p) in row p+1
        x = np.ones((nbParam + 1 + 1, nbParticle + 1))       # each colums is a particle
                                                             # the last columns is the swarm best
                                                             # de toutes les particules et itération
                                                             # line nbParam+1 = best value
                                                             # line nbParam+2 = unique ID
                                                             
        xbest = maxObj * np.ones((nbParam + 1, nbParticle))  # save the location associated to each particle best result
                                                             # and the associated cost function value
                                                                                                               
        # Initializing velocity
        v = np.ones((nbParam, nbParticle))               # Velocity of each particle
        for i in range(0, nbParam):
            v[i,:] = (plageUp - plageDown)[i] / depMax * 2 * (0.5-np.random.rand(nbParticle))
            
        x[nbParam, :] = maxObj                               # Initializing cost function
        
        # Setting first particle as best, for graphical purposes
        x[:, nbParticle] = x[:,0]
        
        if sampling == "random":
            for j in range(0, nbParticle):
                for i in range(0, nbParam):
                    if i == 0:
                        lowerBound = 0
                    else:
                        lowerBound = x[i-1, j]
                    upperBound = (461+461-3-100) - nbParam + i + 1
                    x[i, j] = lowerBound + int((upperBound - lowerBound) * np.random.rand()) # initialisation aléatoire des particules
                    
        # Unique ID affectation   
        for i in range(0, nbParticle):
            x[nbParam+1, i] = i 
    
        #%%% --- Starting first screening --- ###
        print('\nIteration: 0 -> evaluating the objective function at the swarm initial positions')
        
        # Before the first run, compute the starting point performances
        results = []
        def accumulateResults(currentResult):
            results.append(currentResult)
            
        # Filling the pool
        pool_PSO = mp.Pool(processes=procNb)
        
        # Splitting m over the different processors
        x_split = np.array_split(x[:,:-1], procNb, axis=1)
        
        for j in range(0, len(x_split)):
            # Cost function computation in parallel
            pool_PSO.apply_async(processLargeLoader, args=(x_split[j], TrainData, ValidData), callback=accumulateResults)
        pool_PSO.close()
        pool_PSO.join()
        
        # Organising the results
        for j in range(0, len(results)):
            for k in range(0, len(results[j])):
                x[nbParam, int(results[j][k][1])] = results[j][k][0]
                
        # Processing the results  
        for j in range(0, nbParticle):
            # Saving first iteration as particle best
            xbest[:,j] = x[:-1,j]
            # Saving if beats swarm best
            if x[nbParam, j] < x[nbParam, nbParticle]:
                x[:, nbParticle] = x[:,j]
                
        #%%% --- Starting PSO --- ###
        for i in range(0, nbitermax):
            print('\nIteration: ' + str(i+1))
        
            # Define an output queue
            # Building up results agglomerator
            results = []
            def accumulateResults(currentResult):
                results.append(currentResult)
              
            # Moving PSO particle  
            for j in range(0, nbParticle):
                
                # Computing chaotic random inertia
                z1 = np.random.uniform(0, 1)
                z2 = np.random.uniform(0, 1)
                inertie = 0.5 * z1 + 4 * 0.5 * z2 * (1-z2)
                
                # Moving each particle
                for k in range(0, nbParam):
                    # Velocity calculation  
                    v[k, j] =   inertie * v[k, j] + \
                                c1 * np.random.rand() * (xbest[k, j] - x[k,j]) + \
                                c2 * np.random.rand() * (x[k, nbParticle] - x[k,j])
                        
                      
                    # Capping of the velocity          
                    if v[k, j] > (plageUp[k] - plageDown[k]) / depMax:
                        v[k, j] = (plageUp[k] - plageDown[k]) / depMax
                    if v[k, j] < -(plageUp[k] - plageDown[k]) / depMax:
                        v[k, j] = -(plageUp[k] - plageDown[k]) / depMax
                        
                    # Updating each parameter
                    x[k,j] = int(x[k,j]) + int(v[k,j])
                    
                    # Checking that the parameters are within the range
                    if k == 0:
                        lowerBound = 0
                        if nbParam >= 2:
                            upperBound = x[k+1,j] # old value as boundary, k+1 not provessed yet
                    
                    elif k == nbParam -1:
                        upperBound = (461+461-3-100)
                        if nbParam >= 2:
                            lowerBound = x[k-1,j] - int(v[k-1,j]) # old value as boundary
                    else:
                        upperBound = x[k+1,j] # old value as boundary, k+1 not provessed yet
                        lowerBound = x[k-1,j] - int(v[k-1,j]) # old value as boundary
                        
                    if x[k, j] > upperBound:
                        x[k, j] = upperBound
                    if x[k, j] < lowerBound:
                        x[k, j] = lowerBound
                        
            # Filling the pool
            pool_PSO = mp.Pool(processes=procNb)
            x_split = np.array_split(x[:,:-1], procNb, axis=1)
            
            for j in range(0, len(x_split)):
                # Cost function computation in parallel
                pool_PSO.apply_async(processLargeLoader, args=(x_split[j], TrainData, ValidData), callback=accumulateResults)
            pool_PSO.close()
            pool_PSO.join()
        
            # Organising the results
            for j in range(0, len(results)):
                for k in range(0, results[j].shape[0]):
                    x[nbParam, int(results[j][k][1])] = results[j][k][0]
                    
            # Processing the results  
            for j in range(0, nbParticle):
                # Saving if beats particle best
                if x[nbParam, j] < xbest[nbParam, j]:
                    xbest[:,j] = x[:-1,j]
                # Saving if beats swarm best
                if x[nbParam, j] < x[nbParam, nbParticle]:
                    x[:, nbParticle] = x[:,j]
                    
            # Carrying on or stopping here?
            Improvement = np.floor(abs(OldMax - x[nbParam, nbParticle]) / (abs(OldMax) + 1e-15) * 100000)
            OldMax = x[nbParam, nbParticle]
            
            # Printing out infos
            print('Improvement: ' + str(Improvement/1000) + ' %')
            print('Best score : {:f}'.format(x[nbParam, nbParticle]))
            print('Current best : ')
            for l in range(0, nbParam):
                print("Parameter " + str(l+1) + ": " + "{:d}".format(int(x[l, nbParticle])) + " (min : " + "{:d}".format(int(np.min(x[l, :]))) + ", max : " + "{:d}".format(int(np.max(x[l, :]))) + ", limits : " + str(plageDown[l]) + " / " + str(plageUp[l]) + " )")
        
            if Improvement <= 0:
                StagnationCounter = StagnationCounter + 1
            else:
                StagnationCounter = 0
        
            print("Stagnation: " + str(StagnationCounter))
            
            # Just to make life easier
            bestParticleString = ""
            for k in range(0, nbParam-1):
                bestParticleString += "{:.6g}, ".format(x[k, nbParticle])
            bestParticleString += "{:.6g}".format(x[nbParam-1, nbParticle])
            print("Best particle (for copy/paste): " + bestParticleString)
            
            # Saving the results
            if not screeningMode: 
                saveBest(x, nbParam, nbParticle, i, plageDown, plageUp, StagnationCounter, maxStagnation, description, startTime_string)
                meanStdBest.append([np.average(x[nbParam, 0:nbParticle]), np.std(x[nbParam, 0:nbParticle]), x[nbParam, nbParticle], np.min(x[nbParam, 0:nbParticle]), np.max(x[nbParam, 0:nbParticle])] )
                np.savetxt("Mean_Std_Best.txt", np.array(meanStdBest))     
            else: 
                np.savetxt("Screening/Screening_" + str(nrun) + "/Swarm_Best_" + str(nbWL) + ".txt", x[:-1, nbParticle])
            
            if StagnationCounter > maxStagnation:
                break

endTime = time.time()
print('Execution time : {:3.2f} h '.format((endTime - startTime)/3600))
