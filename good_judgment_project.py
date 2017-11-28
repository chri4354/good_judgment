# %% import 
import sys
sys.path.insert(0, '/data/myfunctions/')
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp
from metacognition import type2roc
import theano.tensor as tt
import pymc3 as pm
import pickle

# Additional details for the Good Judgment Project data repository.
# 
# 
# ## Field Details
# Many of the field (column) names are common to a range of files in the repository.  Some key value definitions are
# 
# 
# ```
# # IFP identifier
# ifp_id        1000-8999       Base number assigned to question
# 
# 
# # User identifier
# user_id       00001 - 05999 Year 1 GJP
#               06000 - 06999 Year 2 GJP
#               07000 - 15999 Year 3 GJP
#               17500+        Year 4 GJP
# 
# # User condition assignment
# ctt           
#               Individuals
#               1a   = Individual w/o training (all years)
#               1b   = Individual w/ probability training (all years)
#               1c   = Individual w/ scenario training (year 1)
#               1h   = Individual w/ training; Hybrid-Accountability (year 4)              
#               1n   = MOOF platform with NCBP scoring (year 4)
#               1p   = Individual w/ training; Process-Accountability (year 4)
#               1r1  = MOOF raters (individuals) (year 4)
#               1u   = MOOF platform untrained individuals [no train](year 4) 
#               1z   = MOOF platform standard participant (year 4)
#               
#               Individuals who could see crowd information
#               2a   = Crowd information w/o training (year 1) 
#               2b   = Crowd information w/ probability training (year 1)
#               2c   = Crowd information w/ scenario training (year 1)
# 
#               Prediction Markets
#               3a   = Lumenogic Prediction Market (year 2)
#               3b   = Lumenogic Prediction Market w/ training (year 2)
#               3b1  = Lumenogic Prediction Market (year 3) w/ pretty much no training (same market as 3b2)
#               3b2  = Lumenogic Prediction Market (year 3) w/ some training (same market as 3b1)
#               3e   = Lumenogic Prediction Market for non-citizens / overflow / experimental (year 3)              
#               3f   = Inkling Control Prediction Market (year 3 & 4)
#               3g1  = Inkling Batch Auction w/o Training (year 4)
#               3g2  = Inkling Batch Auction w/ Training (year 4)
#               3s   = Inkling Super Market (year 4)
#               3txx = Inkling Prediction Market Teams, xx = team_id (year 4)
# 
#               Teams (xx = team_id)
#               4axx = Teams without training (year 1 & 2)              
#               4bxx = Teams with training(all years); Outcome Accountability (year 4)
#               4cxx = Teams with scenario training (year 1)
#               4dxx = Teams with training and facilitators (year 3)
#               4hx  = Teams with training; Hybrid Accountability (year 4)
#               4px  = Teams with training; Process Accountability (year 4)
#               4uxx = Team size experiment with smaller teams (year 4)
#               4wxx = Team size experiment with larger teams (year 4)
#               
#               Superforecasters (xx = team_id)
#               5bxx = Superteams with training (year 2)
#               5dxx = Superteams with training and facilitators (year 3)
#               5sxx = Superteams with training; Outcome Accountability (year 4)
# 
# 
# # forecast identifiers
# 
# forecast_id   unique integer identifier within year
# 
# fcast_type    0 = new, first forecast on an IFP by a user
#               1 = update, subsequent forecast by a user
#               2 = affirm, update to a forecast with no change in value
#               4 = withdraw (probabilities show last standing, individual scoring stops after this date)
# 
# # IFP question type
# 
# q_type        0 = regular binomial or multinomial
#               1 = cIFP, Answer Option 1 
#               2 = cIFP, Answer Option 2
#               3 = cIFP, Answer Option 3
#               4 = cIFP, Answer Option 4
#               5 = cIFP, Answer Option 5
#               6 = Ordered Multinomial
#               
# q_text        Question Text
# q_desc        "More Info" and formal resolution criteria
# 
# # Current IFP question status
# q_status      Voided - not counted for scoring
#               Closed - formally resolved and scored)
# 
# 
# # Other IFP properties              
# date_start    YYYY-MM-DD, date question opened for forecasts
# date_suspend  YYYY-MM-DD, date question suspended on platforms
# date_to_close YYYY-MM-DD, end date as specified in question
# date_closed   YYYY-MM-DD HH:MM:SS, date and time question declared closed by MITR
# outcome       "a"-"e" as resolved
# short_title   main short title
# 
# 
# ```

# # Import data # 

# In[3]:


IFP = pd.read_csv('ifps.csv', encoding='latin-1')
IFP.head()


# Now we import the individual differences table

# In[4]:


ID = pd.read_csv('all_individual_differences.tab',sep="\t")
ID.head()


# # Evaluate calibration #

# Define functions to report correct response.

# In[5]:


# return correct option
def correct_answer(ifp_id):
    # Given an IFP, return its correct answer
    mask = IFP['ifp_id'] == ifp_id
    corr = IFP['outcome'][mask]
    return corr.values.item()

# return number of options for a particular ifp
def no_opts(ifp_id):
    # Given an IFP, return its correct answer
    mask = IFP['ifp_id'] == ifp_id
    no = IFP['n_opts'][mask]
    return no.values.item()


# Now we import the forecasts table

# In[11]:


forecasts = pd.read_csv('survey_fcasts.yr1.tab',sep="\t")
forecasts.head()


# In[12]:


# parallelise for loop to return the correct outcome vector 
# (dims: DF.shape[0] x 1)
inputs = range(forecasts.shape[0])
serier = pd.DataFrame()
def processInput(i):
    return correct_answer(forecasts['ifp_id'][i])
 
num_cores = mp.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)

# append result to existing data frame DF
newcol = pd.Series(results)
forecasts_ = forecasts.copy()
forecasts_.insert(10,'outcome',newcol)


# Define accuracy function and create data frame encoding accuracy and 
# confidence vectors.

# In[13]:


# define accuracy (binary) and brier vectors
def accuracy_idx(corr, ans, p):
    # Given correct answer, and probability estimate, return squared error
    err2 = ((corr == ans) - p)**2 # Incorrect:(0-p)**2 vs Correct:(1-p)**2
    # Define binary accuracy: 0=greater probability put on the wrong answer
    if corr == ans:
        acc = p > .5
    elif corr != ans:
        acc = p <= .5
    else:
        acc = np.nan
    return acc, err2

def performance(outc, ans, val, sid, ifpid):
    """ Return a performance dataframe, where accuracy is defined as:
    is the estimated probability of the true event greater than
    other estimated probabilities?"""
    baseline = 1 / no_opts(ifpid)
    if outc == ans:
        acc = val > baseline
        conf = np.around([np.abs(val - baseline) / baseline], decimals = 2).item()
    else: 
        acc = np.nan
        conf = np.nan
    return ifpid, sid, acc, conf


# In[14]:

# run performance function on each row of forecasts_
serier = pd.DataFrame()
def processInput(i):
    return performance(forecasts_['outcome'][i], forecasts_['answer_option'][i], forecasts_['value'][i],
                         forecasts_['user_id'][i], forecasts_['ifp_id'][i])

results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)


# In[15]:

# assign results to PERF dataframe
perf = pd.DataFrame(results,columns = ['ifp_id','user_id','acc','conf'])
PERF = perf.dropna(axis=0, how='any')


# Set all confidence values bigger than 1 to one [HACK]

# In[16]:

P = PERF.copy()
P.loc[P['conf']>1,'conf'] = 1
P['conf'] = P['conf'] * 100


# In[17]:

# group entries by user id
g = P.groupby(['user_id'])
serier = pd.DataFrame()
def processInput(i,group):
    return i,type2roc(group['acc'],group['conf'],100).item()

results = Parallel(n_jobs=num_cores)(delayed(processInput)(i,group) \
                   for i,group in g)


AUROC1 = pd.DataFrame(results,columns =['user_id','ArocII'])


#%% Plot AROC

AUROC1.hist(column='ArocII',bins=10)
plt.show()


#%% Brier Scores #
# In[21]:


def compute_brier(prob, truth):
    """
    Computes Brier scores according to 
    
    Inputs:
    prob: dims = n_options x 1
    truth: dims = n_options x 1
    
    Outputs: 
    """
    
    if np.shape(prob) != np.shape(truth):
        Warning('Input dimensions must be consistent!')
    
    return np.sum((truth - prob)**2)

def processInput(i,group):
    return i,compute_brier(group['value'], \
                           group['answer_option']==group['outcome'])

#%% Compute and plot Brier

g = forecasts_.groupby(['forecast_id'])
results = Parallel(n_jobs=num_cores-2)(
        delayed(processInput)(i,group) for i,group in g)
Brier = pd.DataFrame(results, columns=['forecast_id', 'brier'])

Brier.hist()
plt.show()

#%% Forecast correlation vs. accuracy of the average
#%% define function to compute correlation
def bayes(x,opts):
    p1 = x.values[0,]
    p2 = x.values[1,]
    return pd.DataFrame((p1 * p2) / np.sum(p1*p2), index=opts)

def compute_corr(i, subji):
    print(chr(27) + "[2J")
    # progression bar
    print('|',int(i / np.size(participants) * 100) * '*',
           int(100 - i / np.size(participants) * 100) * '-','|')
    R=[]
    A=[]
    for j,subjj in enumerate(participants[i+1:]):
        # find overlapping questions between subject i and j
        overlap = set(
                forecasts_['ifp_id'][forecasts_['user_id'] == subji]) & set(
                        forecasts_['ifp_id'][forecasts_['user_id'] == subjj])
        x = []
        y = []
        a_avg = []
        a_bays = []
        for q in overlap:
            # forecast probas from subject i and j
            x_ = forecasts_['value'][(
                    forecasts_['ifp_id']==q)&(forecasts_['user_id']==subji)]
            y_ = forecasts_['value'][(
                    forecasts_['ifp_id']==q)&(forecasts_['user_id']==subjj)]
            
            # take only most recent forecast on the last option
            x_=np.array(x_)[-1]
            y_=np.array(y_)[-1]
            
            # number of options in question q
            nopts = IFP['n_opts'][IFP['ifp_id']==q].item()
            # preprocess data 
            #(some participants provide multiple responses to the same 
            #questions)
            #while np.size(x_) != np.size(y_):
            #    if np.size(x_) < np.size(y_):
            #        x_ = np.append(x_,x_[-nopts:])
            #    elif np.size(y_) < np.size(x_):
            #        y_ = np.append(y_,y_[-nopts:])
                
            # append question forecasts
            x = np.append(x,x_)
            y = np.append(y,y_)
            # compute accuracy of the average prediction
            a_avg = np.append(a_avg,
                          compute_brier(
                                  np.mean([x_,y_]),
                                  forecasts_.answer_option.unique(
                                          )[nopts-1]== \
                                  IFP.outcome[IFP.ifp_id==q]))

            # compute accuracy of the Bayesian aggregation FIX!
            a_bays = np.append(a_bays,
                          compute_brier(
                                  bayes([x_,y_], nopts),
                                  forecasts_.answer_option.unique(
                                          )[nopts-1]== \
                                  IFP.outcome[IFP.ifp_id==q]))
        
        # compute correlation coeff
        if len(overlap) <= 2:
            R = np.append(R,np.array(-999))
            A = np.append(A,np.array(-999)) # pearson's R
        else:
            R = np.append(R,np.corrcoef(x,y)[0,1]) # pearson's R
            A = np.append(A,np.mean(a_avg)) # pearson's R
    return R, A

# %% define distance functions
def distance(p1,p2):
    """
    Compute distance between 2 forecasts on multinomial problems.
    
    Inputs:
        p1 = forecasts array for subj 1 (dims: n_opts  x 1)
        p2 = forecasts array for subj 2 (dims: n_opts  x 1)
    Outputs:
        d = distance metric (scalar)
    """
        
    return np.mean((p1 - p2) ** 2)

# compute distance metric
def compute_distance(i, subji):
    print(chr(27) + "[2J")
    # progression bar
    print('|',int(i / np.size(participants) * 100) * '*',
           int(100 - i / np.size(participants) * 100) * '-','|')
    
    D=[]
    A=[]
    for j,subjj in enumerate(participants[i+1:]):
        # find overlapping questions between subject i and j
        overlap = set(
                forecasts_['ifp_id'][forecasts_['user_id'] == subji]) & set(
                        forecasts_['ifp_id'][forecasts_['user_id'] == subjj])
        
        # initialise variables
        d = np.array([])
        a = np.array([])
        for q in overlap:
            # forecast probas from subject i and j
            p1 = forecasts_[['value','answer_option','outcome']][(
                    forecasts_['ifp_id']==q)&(forecasts_['user_id']==subji)]
            p2 = forecasts_[['value','answer_option','outcome']][(
                    forecasts_['ifp_id']==q)&(forecasts_['user_id']==subjj)]
            
            # number of options in question q
            nopts = IFP['n_opts'][IFP['ifp_id']==q].item()
#             preprocess data 
#            (some participants provide multiple responses to the same 
#            questions)
            while p1.shape[0] != p2.shape[0]:
                if p1.shape[0] < p2.shape[0]:
                    p1 = p1.append(p1[-nopts:])
                elif np.size(p2) < np.size(p1):
                    p2 = p2.append(p2[-nopts:])
                
            # appen distance 
            d = np.append(d,distance(
                    p1.value.as_matrix(), p2.value.as_matrix()))
            # compute accuracy of the average prediction
            a = np.append(a,compute_brier(
                    np.mean(np.array([p1.value,p2.value]),axis=0),
                    p1.answer_option== p1.outcome))
        
        # compute correlation coeff
        if len(overlap) <= 2:
            D = np.append(D, np.array(-999))
            A = np.append(A,np.array(-999)) # pearson's R
        else:
            D = np.append(D,np.mean(d))
            A = np.append(A,np.mean(a)) # pearson's R
    return D, A

#%%
conditions = ['individual','crowd info','pred markets','teams','SFs']
for g in [1]:
    #participants = forecasts_['user_id'].unique()
    participants = forecasts_['user_id'][np.isnan(forecasts_['team'])].unique()  # participants not in teams
#    participants = forecasts_['user_id'][
#            forecasts_['ctt'].astype(str).str[0].astype(int)==g].unique()  # participants in team X
    participants = participants[0:1000] # short version for debugging
    nparticipants = np.size(forecasts_['user_id'].unique())
    results = Parallel(n_jobs=num_cores-2)(delayed(compute_corr)(i,subj) 
                                           for i,subj in enumerate(
                                                   participants))

    X=pd.DataFrame(results,columns={'R', 'brier'})
    
    x = np.ma.masked_values(np.concatenate(X['R']), value =-999)
    y = np.ma.masked_values(np.concatenate(X['brier']), value =-999)
    
    idx = np.isnan(y)==False # remove nan values
    
    # plot
    plt.hist2d(x[idx], y[idx], bins=100,normed=True)
    #plt.title(['corr: %.2f pval: %.2f' %(pearsonr(x[idx],y[idx]))])
    print('condition: ', conditions[g-1])
    plt.colorbar()
    plt.xlabel('brier of the average')
    plt.ylabel('R')
    plt.show()

# %% LOAD and SAVE
with open('CorrBrier.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
        results = pickle.load(f)
results = Parallel(n_jobs=num_cores-2)(delayed(compute_corr)(i,subj) 
                                       for i,subj in enumerate(
                                               participants))
# save file
with open('CorrBrier.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(results, f)

