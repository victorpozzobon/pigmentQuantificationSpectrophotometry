# -*- coding: utf-8 -*-
"""
Validation, plotting results and getting the equations

@author: Victor Pozzobon

Article
Nitrate and nitrite as mixed source of nitrogen for Chlorella vulgaris: fast nitrogen quantification using spectrophotometer and machine learning.
https://www.springer.com/journal/10811
Pozzobon, V., Levasseur, W., Guerin, C. & Perre, P. (2021). 
Journal of Applied Phycology, 1-9.
"""
#### --- Clear all --- ###
Clear = False
if Clear:
    from IPython import get_ipython
    get_ipython().magic('reset -sf') 
#### --- --------- --- ###

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
plt.close('all')


# Loading the best set of wavelengths
nbParam = 461+461-3-100
m = np.zeros(nbParam)
ActivatedWL = [42, 255, 289, 322, 336, 391]
ActivatedWL = np.genfromtxt("Screening/Screening_" + str(87) + "/Swarm_Best_" + str(7) + ".txt")[:-1]
for i in ActivatedWL:
    m[int(i)] = 1
dimensions = ["Chlorophyll a", "Chlorophyll b", "Zeaxanthin", "Lutein", "Violaxanthin"]
n_dim = len(dimensions)

# Training PLS
Data_train = np.load("Data/Pigment_Spectra_train.npy")
X = Data_train[:-5, :][m>0.5]
Y = Data_train[-5:, :]

pls = PLSRegression(n_components = len(ActivatedWL))
pls.fit(np.transpose(X), np.transpose(Y))

# Validating
Data_valid = np.load("Data/Pigment_Spectra_valid.npy")
X = Data_valid[:-5, :][m>0.5]
Y = Data_valid[-5:, :]
Y_pred = pls.predict(np.transpose(X))
Y = np.transpose(Y)

# Saving data for reuse or plot in another software
np.savetxt('Results/Data_validation.txt', Y)
np.savetxt('Results/Data_prediction.txt', Y_pred)

########################
# Plotting the results #
########################
nrow = 3
ncol = n_dim
k_plot = 1
plt.figure(figsize=(12, 6*n_dim)) 


# First bissector
for i in range(0, n_dim):
    plt.subplot(nrow, ncol, k_plot)
    
    # Bounding the graphs
    mingraph = 0.9 * np.min([np.min(Y_pred[:,i]), np.min(Y[:,i])])
    maxgraph = 1.1 * np.max([np.max(Y_pred[:,i]), np.max(Y[:,i])])
    plt.xlim([mingraph, maxgraph])
    plt.ylim([mingraph, maxgraph])
    
    mingraph = 0
    plt.plot([mingraph, maxgraph], [mingraph, maxgraph], '-b')
    # plt.legend(["First bisector"])
    # plt.title("First bisector plot " + dimensions[i])
    plt.title(dimensions[i])
    plt.xlabel("PSO-GA-PLS estimations (mg/l)")
    plt.ylabel("IC measurements (mg/l)")
    
    # Labeling with the numbers
    number = True
    if number:
        for k in range(0, len(Y_pred[:,0])):
            label = str(k)
            plt.annotate(label, # this is the text
                         (Y_pred[k,0],Y[k,0]), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,0), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center
    plt.plot(Y_pred[:,i], Y[:,i], 'x', color = 'r')
    k_plot += 1
    
# Absolute comparison
for i in range(0, n_dim):
    plt.subplot(nrow, ncol, k_plot)
    plt.plot(Y[:,i], 'o')
    plt.plot(Y_pred[:,i], 'x', color = 'r')
    # plt.legend(["IC measurements", "PSO-GA-PLS estimation"])
    # plt.title("Direct comparison - On the validation part of the dataset - " + dimensions[i])
    plt.xlabel("Index (-)")
    plt.ylabel("Concentration (mg/l)")
    k_plot += 1

# Error plot
for i in range(0, n_dim):
    plt.subplot(nrow, ncol, k_plot)
    n, bins, patches = plt.hist(x=Y_pred[:,i] - Y[:,i], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Error value (mg/l)')
    plt.ylabel('Count (-)')
    k_plot += 1

# Saving the graphs
plt.tight_layout()    
plt.savefig("Results/Results.png", dpi= 600)

#########################
# Computing some values #
#########################
print("Various value assessing quantification efficiency:")
for i in range(0, n_dim):
    print("\nMean " + dimensions[i] + " error: {:3.3f} mg/l".format(np.average(Y_pred[:,i] - Y[:,i])))
    print("STD " + dimensions[i] + " error: {:3.3f} mg/l".format(np.std(Y_pred[:,i] - Y[:,i])))

    # Relative errors
    err_perc = 0
    k = 0
    for j in range(0, len(Y[:,i])):
        if Y[j,i] > 0:
            err_perc += (Y_pred[j,i] - Y[j,i]) / Y[j,i] * 100
            k +=1
    print("Abs % " + dimensions[i] + " error: {:3.3f} %".format(err_perc/k))


    # Removing outliers (should not impact results)
    mask= np.abs(Y_pred[:,i] - Y[:,i]) < 0.5
    print("STD " + dimensions[i] + " error without outlier: {:3.3f} mg/l".format(np.std(Y_pred[mask,i] - Y[mask,i])))

    blank = []
    for j in range(0, len(Y[:,i])):
        if Y[j,0] < 0.01:
            blank.append(Y_pred[j,i])
        
    print(dimensions[i] + " Blank avg: {:3.3f} mg/l".format(np.average(blank)))
    print(dimensions[i] + " Blank std: {:3.3f} mg/l".format(np.std(blank)))
    print(dimensions[i] + " LoD: {:3.3f} mg/l".format(np.average(blank) + 3 * np.std(blank)))
    print(dimensions[i] + " LoQ: {:3.3f} mg/l".format(np.average(blank) + 10 * np.std(blank)))



##################################
# Obtaining the actual equations #
##################################

# Using symbolic calculation to get an "esay to read" output
import sympy as sym
from sympy.physics.vector import ReferenceFrame, dot
from sympy.matrices import MatrixSymbol, Transpose

# Creating list of symbolic variable
list_var = []
k = 0
for i in range(0, int(np.sum(m[:-2]))):
    list_var.append('A_' + str(int(ActivatedWL[k])))
    k += 1

xsym = sym.Matrix(list_var)
coef = sym.Matrix(pls.coef_)
xmean = sym.Matrix(pls._x_mean)
xstd = sym.Matrix(pls._x_std)
ymean = sym.Matrix(pls._y_mean)

X1 = xsym-xmean
for i in range(0, len(X1)):
    X1[i] = X1[i] / xstd[i]

expression = Transpose(X1) @ coef+Transpose(ymean)
from sigfig import round
print("\nSymbolic:")
print('Equation, then coefficients for each species with adequate precision')
k = 0
for chemical in dimensions:
    print("\n" + chemical + " = ")
    print(expression[k])
    constantValue = expression[k]
    stringEq = "[" + chemical + "] = " 
    for i in range(0, len(xsym)):
        print("For " + str(xsym[i]) + ': {:3.3f}'.format(expression[k].coeff(xsym[i])))
        stringEq += str(round(expression[k].coeff(xsym[i]), 4)).rstrip("0") + " " + str(xsym[i]).replace("_", "_{") + "nm} "
        constantValue = constantValue.coeff(xsym[i], 0) # extracting the only value that does not depend on any variable
    print("For constant: {:3.3f}".format(constantValue))
    stringEq += " " + str(round(constantValue, 4)).rstrip("0")
    print(stringEq)
    k = k + 1    
k=k-1


# # Validation 
# Data_valid = np.load("Data/Pigment_Spectra_valid.npy")
# X = Data_valid[:-5, :][m>0.5]
# Y = Data_valid[-5:, :]
# Y_pred = pls.predict(np.transpose(X))
# X = np.transpose(X)
# Y = np.transpose(Y)

# for sample in range(0, len(X)*0 + 1):
#     k = 0
#     for chemical in dimensions:
#         print(Y_pred[sample, k])
#         val = 0
#         constantValue = expression[k]
#         for coeff in range(0, len(xsym)):
#             val += X[sample, coeff] * expression[k].coeff(xsym[coeff])
#             constantValue = constantValue.coeff(xsym[coeff], 0)
#         val += constantValue
#         print(val)
#         k = k + 1 
        
# k=k-1




#### Comparing to Whilshire
# dimensions = ["Chlorophyll a", "Chlorophyll b", "Zeaxanthin", "Lutein", "Violaxanthin"]
# X = Data_valid[:-5, :][m>0.5]
# Y = Data_valid[-5:, :]
# Y_pred = pls.predict(np.transpose(X))
# Y = np.transpose(Y)

# Willshire 
plt.ion()
offset = 340

# Remove 8
# Y_pred = np.delete(Y_pred, 8, 0)
# Y = np.delete(Y, 8, 0)
# Data_valid = np.delete(Data_valid, 8, 1)

npoint = len(Y_pred)
Ca = np.zeros(npoint)
Cb = np.zeros(npoint)
Cxc = np.zeros(npoint)

for i in range(0, npoint):
    Abs = Data_valid[:, i]
    idx666 = 666-offset
    A666 = np.max(Abs[idx666-0:idx666+1])
    idx653 = 653-offset
    A653 = Abs[idx653]
    idx470 = 470-offset
    A470 = np.max(Abs[idx470-0:idx470+1])
    idx750 = 750-offset
    A750 = Abs[idx750]

    # A666 = Data_valid[666-offset, :]
    # A653 = Data_valid[653-offset, :]
    # A470 = Data_valid[470-offset, :]
    # A750 = Data_valid[750-offset, :]
    
    # 3 wavelength prediction
    Ca[i] = 15.65*(A666-A750) - 7.34*(A653-A750)
    Cb[i] = -11.21*(A666-A750) + 27.05*(A653-A750)
    
    # Ca[i] = 16.5169*(A666-A750) - 8.0962*(A653-A750)
    # Cb[i] = -12.1688*(A666-A750) + 27.4405*(A653-A750)
    
    
    Cxc[i] = (1000.*(A470-A750) - 2.86*Ca[i] -129.2*Cb[i])/221.
x = np.arange(0, len(Ca))+1


xx = np.arange(0, len(Abs)) + offset
plt.figure()
plt.plot(xx, Abs)

nsample = len(Ca)
import time
time.sleep(1)
# Y[8, 0] = 1e-9
# Y[8, 1] = 1e-9
Y_pred = (Y_pred + abs(Y_pred)) /2 
print("\n\nComparing:")
plt.figure()
ax1 = plt.subplot(1,3,1)
ax1.bar(x, Ca, width = 0.25)
ax1.bar(x+0.25, Y[:, 0], width = 0.25)    
ax1.bar(x+0.5, Y_pred[:, 0], width = 0.25)  
print("\nChl a:")
print("3 wavelength average relative error: {:3.3f} %".format(np.nanmean((Ca-Y[:, 0])/Y[:, 0]*100)))
print("3 wavelength MSE: {:3.3f} mg²/L²".format(np.sum((Ca-Y[:, 0])**2)/nsample))
print("Proposed model average relative error: {:3.3f} %".format(np.nanmean((Y_pred[:, 0]-Y[:, 0])/Y[:, 0]*100)))
print("Proposed model MSE: {:3.3f} mg²/L²".format(np.sum((Y_pred[:, 0]-Y[:, 0])**2)/nsample))
  
ax2 = plt.subplot(1,3,2)
ax2.bar(x, Cb, width = 0.25)
ax2.bar(x+0.25, Y[:, 1], width = 0.25) 
ax2.bar(x+0.5, Y_pred[:, 1], width = 0.25) 
print("\nChl b:")
print("3 wavelength average relative error: {:3.3f} %".format(np.nanmean((Cb-Y[:, 1])/Y[:, 1]*100)))
print("3 wavelength MSE: {:3.3f} mg²/L²".format(np.sum((Cb-Y[:, 1])**2)/nsample))
print("Proposed model average relative error: {:3.3f} %".format(np.nanmean((Y_pred[:, 1]-Y[:, 1])/Y[:, 1]*100)))
print("Proposed model MSE: {:3.3f} mg²/L²".format(np.sum((Y_pred[:, 1]-Y[:, 1])**2)/nsample))

ax3 = plt.subplot(1,3,3)
ax3.bar(x, Cxc, width = 0.25)
ax3.bar(x+0.25, np.sum(Y[:, 2:], axis=1), width = 0.25) 
ax3.bar(x+0.5, np.sum(Y_pred[:, 2:], axis=1), width = 0.25)
print("\nCxc:")
print("3 wavelength average relative error: {:3.3f} %".format(np.nanmean((Cxc-np.sum(Y[:, 2:], axis=1))/np.sum(Y[:, 2:], axis=1)*100)))
print("3 wavelength MSE: {:3.3f} mg²/L²".format(np.sum((Cxc-np.sum(Y[:, 2:], axis=1))**2)/nsample))
print("Proposed model average relative error: {:3.3f} %".format(np.nanmean((np.sum(Y_pred[:, 2:], axis=1)-np.sum(Y[:, 2:], axis=1))/np.sum(Y[:, 2:], axis=1)*100)))
print("Proposed model MSE: {:3.3f} mg²/L²".format(np.sum((np.sum(Y_pred[:, 2:], axis=1)-np.sum(Y[:, 2:], axis=1))**2)/nsample))

print("Absolute")
print("3 wavelength average relative error: {:3.3f} %".format(np.nanmean(abs(Cxc-np.sum(Y[:, 2:], axis=1))/np.sum(Y[:, 2:], axis=1)*100)))
print("3 wavelength MSE: {:3.3f} mg²/L²".format(np.sum((Cxc-np.sum(Y[:, 2:], axis=1))**2)/nsample))
print("Proposed model average relative error: {:3.3f} %".format(np.nanmean(abs(np.sum(Y_pred[:, 2:], axis=1)-np.sum(Y[:, 2:], axis=1))/np.sum(Y[:, 2:], axis=1)*100)))
print("Proposed model MSE: {:3.3f} mg²/L²".format(np.sum((np.sum(Y_pred[:, 2:], axis=1)-np.sum(Y[:, 2:], axis=1))**2)/nsample))

import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import pandas as pd

fig = go.Figure()
x = np.arange(0, 16)+1 # for bar chart abscissa
bar_width = 0.25
bar_xoffset = 0.0001
colorRoster = [7, 5, 4]

# Bar
k=0
fig.add_trace(go.Bar(
    x=x + (k - 1) * bar_width,
    width = bar_width,
    y=np.sum(Y[:, 2:], axis=1),
    name="HPLC",
    marker_color=px.colors.qualitative.Prism[1],
)
)
k=1
fig.add_trace(go.Bar(
    x=x + (k - 1) * bar_width,
    width = bar_width,
    y=np.sum(Y_pred[:, 2:], axis=1),
    name="Proposed correlations",
    marker_color=px.colors.qualitative.Prism[5],
)
)
k=2
fig.add_trace(go.Bar(
    x=x + (k - 1) * bar_width,
    width = bar_width,
    y=Cxc,
    name="Wellburn correlations",
    marker_color=px.colors.qualitative.Prism[4],
)
)

fig.update_layout(
    yaxis=dict(
        title_text="Lutein + Zexanthin + Violaxanthin <br /> concentration (mg/L)",
        tickmode="array",
        titlefont=dict(size=18),
        tickfont=dict(size=14),
        range = [0, 1.6]
        ),
    xaxis=dict(
        # title_text=x_axis_label,
        # ticktext=x_text_label,
        title_text="Sample index (-)",
        tickvals=x,
        tickmode="array",
        titlefont=dict(size=18),
        tickfont=dict(size=14)
    ),
    margin=dict(l=20, r=20, t=20, b=20),
    font=dict(
        color="black"
    )
)
fig.update_layout(legend=dict(
    orientation="v",
    yanchor="top",
    y=0.975,
    xanchor="left",
    x=0.025,
    font = dict(size = 18)
    
))
fig.update_layout(xaxis_tickangle=0)

fig.write_image("Fig_SumCaro.png", engine="kaleido", width=600, height=600, scale = 4)
fig.write_image("Fig_SumCaro.pdf", engine="kaleido", width=600, height=600, scale = 1)