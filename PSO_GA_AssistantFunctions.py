# -*- coding: utf-8 -*-
"""
Functions associatd to the PSO algorithm aiming at identifing the relevant number of wavelength

@author: Victor Pozzobon
Article
Lutein, violaxanthin, and zeaxanthin spectrophotometric quantification: a machine learning approach
Pozzobon, V., & Camarena-Bernard, C. (2022).
Journal of Applied Phycology, pp-pp.

"""

import numpy as np # <= to deal with vector and matrices

'''
Saving the best results in a file
'''
def saveBest(x, nbParam, nbParticle, i, plageDown, plageUp, StagnationCounter, maxStagnation, description, startTime_string):
    # recreating the file name
    savefileTimeStamp = str(startTime_string.tm_year) + "_" + str(startTime_string.tm_mon) + "_" + str(startTime_string.tm_mday) + "_" + \
        str(startTime_string.tm_hour) + "_" + str(startTime_string.tm_min)+ "_" + str(startTime_string.tm_sec)
    savefileName = "Save_PSO_GA_" + savefileTimeStamp + ".txt"

    # Loading the file
    
    f = open(savefileName, "w")
    # Header
    f.write("PSO_GA savefile")
    f.write("\n")
    f.write("\nDescription: " + description)
    
    # Printing out infos
    f.write("\nNumber of iteration: " + str(i+1))
    f.write('\nBest score : {:f}'.format(x[nbParam, nbParticle]))
    # Printing info
    f.write('\nBest : ')
    for l in range(0, nbParam):
        f.write("\nParameter " + str(l+1) + ": " + "{: .3g}".format(x[l, nbParticle]) + " (min : " + "{: .3g}".format(np.min(x[l, :])) + ", max : " + "{: .3g}".format(np.max(x[l, :])) + ", limits : " + str(plageDown[l]) + " / " + str(plageUp[l]) + " )")

    f.write("\nStagnation: " + str(StagnationCounter) + "/" + str(maxStagnation))
    
    # Just to make life easier
    bestParticleString = ""
    for k in range(0, nbParam-1):
        bestParticleString += "{:.6g}, ".format(x[k, nbParticle])
    bestParticleString += "{:.6g}".format(x[nbParam-1, nbParticle])
    f.write("\nBest particle (for copy/paste): " + bestParticleString)
    f.close()
    
    return 0

''' 
Cost / Loss / Objective function
'''
def ObjectiveFunction(p, TrainData, ValidData, n_com = 10):
    from sklearn.cross_decomposition import PLSRegression
    import numpy as np
    
    # p is the indeces of the activated wavelengths
    # Creating a vector with all the wavelengths. 1 = activated, 0 = not activated
    pprime = np.zeros(461+461-3-100)
    for index in p:
        pprime[int(index)] = 1

    # Number of components for PLS, cannot be below number of paramter
    n_component_actual = np.min([n_com, len(p)])
    
    # Score to be returned
    Gap = 0
    
    # Scaler
    # dimensions = ["Chlorophyll a", "Chlorophyll b", "Zeaxanthin", "Lutein", "Violaxanthin"] # Just to remind you 
    scale_dimensions = [4.495, 1.86, 0.0798, 0.595, 0.063]  # Actual relative weigth
    
    # Tries the procedure, may crash if parameters are really not well-suited
    try:
        # Train and validate in cross validation
        X_calib = TrainData[:-5, :][pprime>0.5]
        X_valid = ValidData[:-5, :][pprime>0.5]
        Y_calib = TrainData[-5:, :]
        Y_valid = ValidData[-5:, :] 
        pls = PLSRegression(n_components = n_component_actual)
        pls.fit(np.transpose(X_calib), np.transpose(Y_calib))
        Y_pred = pls.predict(np.transpose(X_valid))
        Y_valid = np.transpose(Y_valid)
        Y_pred = (Y_pred + np.abs(Y_pred)) / 2 

        for i in range(0, len(Y_pred)):
            for dimension in range(0, len(scale_dimensions)):
                Gap += ((Y_pred[i, dimension] - Y_valid[i, dimension]) / scale_dimensions[dimension])**2 

    except Exception as e:
        Gap = 1e2
        # print("Crash") # Commented to silent the warning
        pass
    return Gap
    
'''
Launching Ojective function of each particle of the batch
Argument to cascade towards objective function
'''
def processLargeLoader(p, TrainData, ValidData):
    import numpy as np
    nb_particle_slip = p.shape[1]
    results = np.zeros([nb_particle_slip, 2])
    results[:, 0] = p[-2, :] # by default, new Gap = old Gap, avoid returning 0
    for i in range(0, nb_particle_slip):
        Gap = ObjectiveFunction(p[0:-2, i], TrainData, ValidData)
        results[i, :] = Gap, p[-1, i]

    return results
