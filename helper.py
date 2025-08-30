
import numpy as np
import pandas as pd

def calculateROCPoints(df):
    
    nN = np.sum(df.label==0)
    nP = np.sum(df.label==1)

    # init fpr and tpr columns
    df.fpr=np.nan
    df.tpr=np.nan

    # loop through each examples (sorted by score) and calculate tpr and fpr for with that score as the threshold 
    prevScore = -1
    prevLabel = -1
    for i in range(0,df.shape[0]):

        threshold = df.loc[i,'score']
        
        # find indexes with examples predicted as true for this threshold
        i_pred = np.where(df.score >= threshold, 1, 0)
        
        # calculate fpr and tpr
        label = df.label[i]
        fp = np.sum((i_pred==1) & (df.label == 0))
        tp = np.sum((i_pred==1) & (df.label == 1))
        
        df.loc[i, 'fpr'] = fp / nN
        df.loc[i, 'tpr'] = tp / nP


    # past the highest score and no examples classified as true 
    numrows = df.shape[0]
    df.loc[numrows,'fpr'] = 0
    df.loc[numrows,'tpr'] = 0
    return(df)


def lowerEnvelope(df, piP=np.nan):

    # use class distribution in data unless piP is specified
    if np.isnan(piP):
        nN = np.sum(df.label==0)
        nP = np.sum(df.label==1)
        piP = nP/(nP+nN)
    
    piN = 1-piP

    # get best loss for each cost
    loss_costs = []
    for c in np.arange(0,1.01, 0.01):
        
        #loss = 2*(c*pi_1*(1-df['tpr']) + (1-c)*pi_0*df['fpr'])
        loss = 2*((1-c)*piP*(1-df['tpr']) + c*piN*df['fpr'])
        
        minLoss = min(loss)

        # the point with lowest loss doesn't have c==t
        #print(c, ': ', df.score[np.argmin(loss)])

        loss_cost = {'cost':c, 'loss':minLoss}
        loss_costs.append(loss_cost)

    dfLossCost = pd.DataFrame(loss_costs)

    return(dfLossCost)

def lowerEnvelopeFlip(df, piP=np.nan):

    # use class distribution in data unless piP is specified
    if np.isnan(piP):
        # labels are flipped
        nN = np.sum(df.label==1)
        nP = np.sum(df.label==0)
        piP = nP/(nP+nN)
    
    piN = 1-piP

    # get best loss for each cost
    loss_costs = []
    for c in np.arange(0,1.01, 0.01):
        
        # flipping means tpr becomes 1-fpr and fpr becomes 1-tpr
        loss = 2*((1-c)*piP*(df['fpr']) + c*piN*(1-df['tpr']))
        
        minLoss = min(loss)

        # the point with lowest loss doesn't have c==t
        #print(c, ': ', df.score[np.argmin(loss)])

        loss_cost = {'cost':c, 'loss':minLoss}
        loss_costs.append(loss_cost)

    dfLossCost = pd.DataFrame(loss_costs)

    return(dfLossCost)


def lowerEnvelopeSkew(df):

    # use class distribution in data unless piP is specified

    nN = np.sum(df.label==0)
    nP = np.sum(df.label==1)
    piP = nP/(nP+nN)    
    piN = 1-piP

    # get best loss for each cost
    loss_costs = []
    for skew in np.arange(0,1.01, 0.01):
        
        loss = (1-skew)*(1-df['tpr']) + skew*df['fpr']
        
        minLoss = min(loss)

        # the point with lowest loss doesn't have c==t
        #print(c, ': ', df.score[np.argmin(loss)])

        loss_cost = {'skew':skew, 'loss':minLoss}
        loss_costs.append(loss_cost)

    dfLossCost = pd.DataFrame(loss_costs)

    return(dfLossCost)


def lowerEnvelopeNB(df, piP=np.nan, brierCosts=False):

    # use class distribution in data unless piP is specified
    if np.isnan(piP):
        nN = np.sum(df.label==0)
        nP = np.sum(df.label==1)
        piP = nP/(nP+nN)
    
    piN = 1-piP
    n = nN+nP

    # get best loss for each cost
    nb_costs = []
    for c in np.arange(0,1.01, 0.01):
        
        if (brierCosts==False):
            nb = piP*df['tpr'] - (c/(1-c))*piN*df['fpr']
        else:
            nb = 2*(1-c)*piP*df['tpr'] - 2*c*piN*df['fpr']

        maxNB = max(nb)

        nb_cost = {'cost':c, 'nb':maxNB}
        nb_costs.append(nb_cost)

    dfNBCost = pd.DataFrame(nb_costs)

    return(dfNBCost)


def lowerEnvelopeNBFlip(df, piP=np.nan):

    # use class distribution in data unless piP is specified
    if np.isnan(piP):
        # labels are flipped
        nN = np.sum(df.label==1)
        nP = np.sum(df.label==0)
        piP = nP/(nP+nN)
    
    piN = 1-piP
    n = nN+nP

    # get best loss for each cost
    nb_costs = []
    for c in np.arange(0,1.01, 0.01):
        
        # flipping means tpr becomes 1-fpr and fpr becomes 1-tpr
        nb = piP*(1-df['fpr']) - (c/(1-c))*piN*(1-df['tpr'])
        

        maxNB = max(nb)

        # the point with lowest loss doesn't have c==t
        #print(c, ': ', df.score[np.argmin(loss)])

        nb_cost = {'cost':c, 'nb':maxNB}
        nb_costs.append(nb_cost)

    dfNBCost = pd.DataFrame(nb_costs)

    return(dfNBCost)



def lowerEnvelopeDCALoss(df, piP=np.nan):

    # use class distribution in data unless piP is specified
    if np.isnan(piP):
        nN = np.sum(df.label==0)
        nP = np.sum(df.label==1)
        piP = nP/(nP+nN)
    
    piN = 1-piP

    # get best loss for each cost
    loss_costs = []
    for c in np.arange(0,1.01, 0.01):
        
        #loss = 2*(c*pi_1*(1-df['tpr']) + (1-c)*pi_0*df['fpr'])
        #loss = 2*((1-c)*piP*(1-df['tpr']) + c*piN*df['fpr'])
        loss = piP*(1-df['tpr']) + (c/(1-c))*piN*df['fpr']

        minLoss = min(loss)

        # the point with lowest loss doesn't have c==t
        #print(c, ': ', df.score[np.argmin(loss)])

        loss_cost = {'cost':c, 'loss':minLoss}
        loss_costs.append(loss_cost)

    dfLossCost = pd.DataFrame(loss_costs)

    return(dfLossCost)
