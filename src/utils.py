######################################################
#
# Useful utility functions
#

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

#####################################################
# Function to calculate normalised dot product
#

def dotprod(vec1,vec2):

    # Normalise
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)

    # Calc
    if n1 > 0.0 and n2 > 0.0:
        dot = np.dot(vec1/n1,vec2/n2)
    else:
        dot = 0.0

    return dot
    
#####################################################
# Function to build and test a linear classifier
#

def testmodel(X,Y,X_tst,Y_tst):

    # Construct decoder and get predictions
    Xc      = sm.add_constant(X,has_constant='add')
    Xc_tst  = sm.add_constant(X_tst,has_constant='add')
    model   = sm.OLS(Y,Xc)
    results = model.fit()
    Yp = results.predict(Xc_tst)

    # How many correct classifications
    sum = 0
    for i in range(len(Y_tst)):
        sum += Y_tst[i,np.argmax(Yp[i,:])]

    return sum/len(Y_tst)

#####################################################
# Function to build and test a linear predictor (not a strict classifier)
#

def testmodel_arb(X,Y,X_tst,Y_tst):

    # Construct decoder and get predictions
    Xc      = sm.add_constant(X,has_constant='add')
    Xc_tst  = sm.add_constant(X_tst,has_constant='add')
    model   = sm.OLS(Y,Xc)
    results = model.fit()
    Yp = results.predict(Xc_tst)

    # How many correct predictions
    sum = 0
    for i in range(len(Y_tst)):
        sum += dotprod(Y_tst[i,:],Yp[i,:])

    return sum/len(Y_tst)

#####################################################
# Function to build and test a linear binary classifier
#

def testmodel_bin(X,y,X_tst,y_tst):

    # Construct decoder and get predictions
    Xc      = sm.add_constant(X,has_constant='add')
    Xc_tst  = sm.add_constant(X_tst,has_constant='add')
    model   = sm.OLS(y,Xc)
    results = model.fit()
    yp = np.round(results.predict(Xc_tst))

    # How many correct classifications
    sum = np.sum(y_tst==yp)

    return sum/len(y_tst)

#####################################################
# Function to create fit line
#

def fitlin(x,y,nb,pctl):
    minx = min(x); maxx = max(x); stp = (maxx-minx)/nb; minx += stp/2
    fx = []; fy = []
    for i in range(nb):
        fx.append(minx)
        #fy.append(np.median(y[np.logical_and(x>=minx-stp/2,x<=minx+stp/2)]))
        xrange = np.logical_and(x>=minx-stp/2,x<=minx+stp/2)
        if np.any(xrange):
            if pctl:
                fy.append(np.percentile(y[xrange],[10,90]))
            else:
                fy.append(np.median(y[xrange]))
        else:
            if pctl:
                fy.append(np.array([0,0]))
            else:
                fy.append(0)
        minx += stp
    return(np.array(fx),np.array(fy))

#####################################################
# Function to perform PCA (from askpython.com/python/examples/principal-component-analysis)
#

def PCA(X,num_components):
     
    #Step-1
    X_meaned = X - np.mean(X,axis=0)
    #Step-2
    cov_mat = np.cov(X_meaned,rowvar=False)
    #Step-3
    eigen_values,eigen_vectors = np.linalg.eigh(cov_mat)
    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
    #Step-6
    X_reduced = np.dot(eigenvector_subset.transpose(),X_meaned.transpose() ).transpose()
    # Done
    return X_reduced

#####################################################
# Function to build and test a logistic classifier
#

def testmodel_logit(X,y,X_tst,y_tst):

    try:
        # Build model and fit data
        #X       = sm.add_constant(X,has_constant='add')
        #X_tst   = sm.add_constant(X_tst,has_constant='add')
        log_reg = sm.Logit(y,X).fit()
        # Get predictions
        yp = np.round(log_reg.predict(X_tst))

        # How many correct classifications
        sum = np.sum(y_tst==yp)
    except:
        sum = 0
        
    return sum/len(y_tst)

#####################################################
# Function to build and test a LM classifier
#

def testmodel_LM(X,y,X_tst,y_tst):

    # Build model and fit data
    X       = sm.add_constant(X,has_constant='add')
    X_tst   = sm.add_constant(X_tst,has_constant='add')
    results = sm.GLM(y,X).fit()
    # Get predictions
    yp = np.round(results.predict(X_tst))
    
    # How many correct classifications
    sum = np.sum(y_tst==yp)

    return sum/len(y_tst)

#####################################################
