
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


def lowerEnvelope(df):

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