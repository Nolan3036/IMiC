#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


from sklearn.base import is_classifier,is_regressor
import shap
from scipy.stats import spearmanr, pointbiserialr
from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils import resample
from sklearn.preprocessing import label_binarize,OneHotEncoder
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,auc,roc_curve
from scipy.stats import mannwhitneyu
from statsmodels.multivariate.manova import MANOVA
from scikitplot.metrics import plot_roc_curve, plot_precision_recall


# In[ ]:


from sklearn.model_selection import GridSearchCV
def debiased_multiomis_framework(model_func,list_df,y,cov_df,params={},omics_name=[],bootstrap_round=10,k=10,k2=10,
               binary_feature_list=[],cov_X_list=[],confounder_list=[],cov_y_list=[]):
    """ 
    A debiased multiomics framework coupling double ML and late fusion with repeated CV strategy for covariate adustment and multiomics integration. 

    #Arguments:
    model_func: SKlearn model function.
    list_df: list of omics data. Each omics dataframe will have four other columns ("Unnamed: 0":sample ID,"ARM":ARM within study,"visit":visit within study,"pid":patient ID)
    y: outcome. (numpy.array)
    cov_df: covariate dataframe. 
    params: hyperparameters for Sklearn model function.
    omics_name: list of omics name for monitoring progress and saving results purposes.
    bootstrap_round: number of iterations for the framework.
    k: outer fold number.
    k2: inner fold number.
    binary_feature_list: list of binary features.
    cov_X_list: list of covariates for predictors.
    confounder_list: list of confounders.
    cov_y_list: list of covariates for outcomes.
    """
    ##generate result matrix
    #train result 
    train_residual_X=[np.full((bootstrap_round,k,list_df[i].shape[0], list_df[i].shape[1]-4), np.nan) for i in range(len(list_df))] #residuals for predictors (multiomics) in train set
    train_residual_y=np.full((list_df[0].shape[0],bootstrap_round,k), np.nan) #residuals for outcomes in train set
    train_mod_result=np.zeros((list_df[0].shape[0],bootstrap_round,k,len(omics_name))) #results for predictors (multiomics) in train set
    train_result=np.zeros((list_df[0].shape[0],bootstrap_round,k)) #results for outcomes in train set
    #test result
    residual_X = [np.zeros((bootstrap_round,list_df[i].shape[0],list_df[i].shape[1]-4)) for i in range(len(list_df))]#residuals for predictors (multiomics) in test set
    residual_y = np.zeros((list_df[0].shape[0],bootstrap_round))#residuals for outcomes in test set
    test_mod_result=np.zeros((list_df[0].shape[0],bootstrap_round,len(omics_name)))#results for predictors (multiomics) in test set
    test_result = np.zeros((list_df[0].shape[0],bootstrap_round))#results for outcomes in train set
    covariate_result = np.zeros((list_df[0].shape[0],bootstrap_round)) #results predicting outcome using covariates, saving if needed, but no further analysis has been done so far.
    #shap value
    test_shap=[np.zeros((bootstrap_round,df.shape[0],df.shape[1]-4)) for df in list_df] # shape value for each omics in test set
    shap_orig=np.zeros((bootstrap_round,list_df[0].shape[0],sum(df.shape[1]-4 for df in list_df))) #propagating shap value through late fusion leveraging Generalized DeepSHAP (G-DeepSHAP) https://doi.org/10.1038/s41467-022-31384-3
    
    ## Repeated CV coupling double ML with late fusion
    #initiate different seed for each iteration
    np.random.seed(88)
    KFold_rs=np.random.randint(low=0,high=(2**32 - 1),size=bootstrap_round)
    for rounds in tqdm(range(bootstrap_round),desc = 'Round'):
        #initiate outer fold for late fusion
        kf=KFold(n_splits=k, random_state=KFold_rs[rounds], shuffle=True)
        #spliting outer train test set, ensuring samples from same mother-infant dyads are kept in the same set.
        for (fold, (train_index, test_index)) in tqdm(enumerate(kf.split(list_df[0].pid.unique())),desc = 'fold'):
            train_index=list_df[0][list_df[0].pid.isin(list_df[0].pid.unique()[train_index])].index.to_numpy()
            test_index=list_df[0][list_df[0].pid.isin(list_df[0].pid.unique()[test_index])].index.to_numpy()
            train_X, test_X = split(list_df,train_index), split(list_df,test_index)
            train_cov_df, test_cov_df = cov_df.iloc[train_index].reset_index(drop=True), cov_df.iloc[test_index].reset_index(drop=True)
            train_y, test_y = y[train_index], y[test_index]
            ##covariate adjustment (double ML, inner fold)
            #generate variance expla
            train_explained_variance,train_explained_variance_o,ifold_index=covariate_adjustment(train_X,train_y,train_cov_df,
                                                                                          model_func,binary_feature_list,
                                                                                          omics_name,cov_X_list=cov_X_list,confounder_list=confounder_list,cov_y_list=cov_y_list,k2=k2)
            #create residual for test predictor (HM component)
            for i, omics in enumerate(omics_name):
                for col in train_X[i].drop(["Unnamed: 0","ARM","visit","pid"],axis=1).columns.tolist():
                    #whether the col is binary or continuous, here we're predicting the variance that could 
                    #be explained, so using Regressor
                    model=XGBRegressor(**params).fit(train_cov_df[confounder_list+cov_X_list],
                                                      train_explained_variance[i][:,train_X[i].drop(["Unnamed: 0","ARM","visit","pid"],axis=1)
                                                                                  .columns.tolist().index(col)])
                    test_prediction=model.predict(test_cov_df[confounder_list+cov_X_list])
                    test_X[i][col]=test_X[i][col].astype(float)-test_prediction
            #create residual for train, test prediction output
            #whether the col is binary or continuous, here we're predicting the variance that could 
            #be explained, so using Regressor
            model=XGBRegressor(**params).fit(train_cov_df[confounder_list+cov_y_list],train_explained_variance_o)
            test_prediction=model.predict(test_cov_df[confounder_list+cov_y_list])
            train_y=train_y-train_explained_variance_o
            test_y=test_y-test_prediction
            #record train_residual_y
            for j, sample in enumerate(train_index):
                train_residual_y[sample,rounds,fold]=train_y[j]
            #record residual_y
            for j, sample in enumerate(test_index):
                residual_y[sample,rounds]=test_y[j]
            #record covaraite_result
            for j, sample in enumerate(test_index):
                covariate_result[sample,rounds]=test_prediction[j]
            ##late fusion
            temp_model=[]
            #generate empty train,val result(use for meta learner in this fold)
            train_mod_prediction=None
            for i, omics in enumerate(omics_name):
                print(omics)
                train_X[i].drop(["Unnamed: 0","ARM","visit","pid"],axis=1,inplace=True)
                test_X[i].drop(["Unnamed: 0","ARM","visit","pid"],axis=1,inplace=True)
                train_X[i]=train_X[i].astype(float)
                #create residual for train HM component (creating residual after removing irrelevant columns)
                train_X[i]=train_X[i]-train_explained_variance[i]
                test_X[i]=test_X[i].astype(float)
                #record train_residual_X
                for j, sample in enumerate(train_index):
                    train_residual_X[i][rounds,fold,sample]=train_X[i].values[j]
                #record residual_X
                for j, sample in enumerate(test_index):
                    residual_X[i][rounds,sample]=test_X[i].values[j]
                #model,trainer set up | train, predict
                if is_classifier(model_func) and train_explained_variance_o.shape[0]!=1:
                    model=XGBRegressor(**params).fit(train_X[i],train_y)
                else:
                    model=model_func(**params).fit(train_X[i],train_y)
                #record train result(for meta learner training)
                if is_classifier(model_func) and train_explained_variance_o.shape[0]==1:
                    train_prediction=model.predict_proba(train_X[i])[:,1].reshape(-1,1)
                else:
                    train_prediction=model.predict(train_X[i]).reshape(-1,1)
                if i==0:
                    train_mod_prediction=train_prediction.copy()
                else:
                    train_mod_prediction=np.hstack((train_mod_prediction, train_prediction))
                #record train result
                for j, sample in enumerate(train_index):
                    train_mod_result[sample,rounds,fold,i]=train_prediction.reshape(-1)[j]
                #record trainer temporarily
                temp_model.append(model)
            #meta model,trainer set up | train, predict
            print("meta")
            if is_classifier(model_func) and train_explained_variance_o.shape[0]!=1:
                clf = GridSearchCV(XGBRegressor(**params), param_grid={"n_estimators":[10,30,50,70,100],
                                                                       "reg_alpha":[0,0.01,0.1,0.5],
                                                                       "reg_lambda":[0,0.01,0.1,0.5],
                                                                       "colsample_bytree":[0.5,0.7,0.9,1]},
                                  cv=ifold_index,
                                  verbose=1)
                clf.fit(train_mod_prediction,train_y)
                model=clf.best_estimator_
            else:
                clf = GridSearchCV(model_func(**params), param_grid={"n_estimators":[10,30,50,70,100],
                                                                     "reg_alpha":[0,0.01,0.1,0.5],
                                                       "reg_lambda":[0,0.01,0.1,0.5],
                                                       "colsample_bytree":[0.5,0.7,0.9,1]},
                  cv=ifold_index,
                  verbose=1)
                clf.fit(train_mod_prediction,train_y)
                model=clf.best_estimator_
            #record train result(for meta learner testing)
            if is_classifier(model_func) and train_explained_variance_o.shape[0]==1:
                train_prediction=model.predict_proba(train_mod_prediction)[:,1]
            else:
                train_prediction=model.predict(train_mod_prediction)
            #record train result(for final train result)
            for j, sample in enumerate(train_index):
                train_result[sample,rounds,fold]=train_prediction.reshape(-1)[j]
            #record meta trainer temporarily
            temp_model.append(model)
            
            ##generate test result
            np.random.seed(KFold_rs[rounds]+fold)
            base_inds=np.random.choice(np.arange(train_X[0].shape[0]),size=100,replace=False)
            test_mod_prediction=None
            #generate test result for individual model
            for i, omics in enumerate(omics_name):
                if is_classifier(model_func) and train_explained_variance_o.shape[0]==1:
                    test_prediction=temp_model[i].predict_proba(test_X[i])[:,1].reshape(-1,1)
                else:
                    test_prediction=temp_model[i].predict(test_X[i]).reshape(-1,1)
                if i==0:
                    test_mod_prediction=test_prediction.copy()
                else:
                    test_mod_prediction=np.hstack((test_mod_prediction, test_prediction))
                #record SHAP value
                if is_classifier(temp_model[i]):
                    model_output="probability"
                else:
                    model_output="raw"
                np.random.seed(KFold_rs[rounds]+fold)
                base_inds=np.random.choice(np.arange(train_X[0].shape[0]),size=100,replace=False)
                explainer = shap.TreeExplainer(temp_model[i],train_X[i].iloc[base_inds],feature_perturbation="interventional",model_output=model_output)
                shap_values = explainer.shap_values(test_X[i])
                for j, sample in enumerate(test_index):
                    test_shap[i][rounds,sample]=shap_values[j]
            for j, sample in enumerate(test_index):
                test_mod_result[sample,rounds]=test_mod_prediction[j]
            #generate test result for meta model
            if is_classifier(model_func) and train_explained_variance_o.shape[0]==1:
                test_prediction=temp_model[-1].predict_proba(test_mod_prediction)[:,1]
            else:
                test_prediction=temp_model[-1].predict(test_mod_prediction)
            for j, sample in enumerate(test_index):
                test_result[sample,rounds]=test_prediction[j]
            ##propagate SHAP value across individual omics or modality with Generalized DeepSHAP (G-DeepSHAP) https://doi.org/10.1038/s41467-022-31384-3
            np.random.seed(KFold_rs[rounds]+fold)
            base_inds=np.random.choice(np.arange(train_X[0].shape[0]),size=100,replace=False)
            shap_values=GDeepShap_SG(temp_model,split(train_X,base_inds)+[pd.DataFrame(train_mod_prediction).iloc[base_inds]],
                                     test_X+[pd.DataFrame(test_mod_prediction)])
            for j, sample in enumerate(test_index):
                shap_orig[rounds,sample]=shap_values[j]
    ###evaluate final performance and feature attribution analysis
    #average result
    train_mod_result_=np.sum(train_mod_result,axis=2)/(k-1) #average fold cv
    train_mod_result_=np.mean(train_mod_result_,axis=1)
    train_result_=np.sum(train_result,axis=2)/(k-1) #average fold cv
    train_result_=np.mean(train_result_,axis=1)
    test_mod_result_=np.mean(test_mod_result,axis=1)
    test_result_=np.mean(test_result,axis=1)
    residual_y_=np.mean(residual_y,axis=1)
    if set(np.unique(residual_y_)).issubset({0, 1}):
        corr_func=pointbiserialr
    else:
        corr_func=spearmanr
    #generate model performance
    test_corr,test_p=draw_correlation(residual_y_,residual_y_,train_mod_result_,train_result_,test_mod_result_,test_result_,corr_func,omics_name)
    ##selecting top feature in the model if the model performance is significant
    if len(list_df)==1:
        tcorr=test_corr[0]
        tp=test_p[0]
    else:
        tcorr=test_corr[-1]
        tp=test_p[-1]
    if (tcorr>0) and (tp<=0.05):
        for i in np.where((np.asarray(test_corr[:-1])>0) & (np.asarray(test_p[:-1])<=0.05))[0]:
            #generate corr, p for every round
            o_test_corr=[]
            o_test_p=[]
            for rounds in range(bootstrap_round):
                if set(np.unique(residual_y[:,rounds])).issubset({0, 1}):
                    r,pv=pointbiserialr(residual_y[:,rounds],test_mod_result[:,rounds,i])
                else:
                    r,pv=spearmanr(residual_y[:,rounds],test_mod_result[:,rounds,i])
                o_test_corr.append(r)
                o_test_p.append(pv)
                
            top_feature(list_df[i],train_residual_X[i],train_residual_y,residual_X[i],residual_y,
                        test_shap[i],o_test_corr,o_test_p,omics_name[i],params=params,
                        bootstrap_round=bootstrap_round,k=k)
    ###record model result for further analysis
    #average result
    train_mod_result=np.sum(train_mod_result,axis=2)/(k-1) #average fold cv
    train_mod_result=np.mean(train_mod_result,axis=1)
    train_result=np.sum(train_result,axis=2)/(k-1) #average fold cv
    train_result=np.mean(train_result,axis=1)
    test_mod_result=np.mean(test_mod_result,axis=1)
    test_result=np.mean(test_result,axis=1)
    covariate_result=np.mean(covariate_result,axis=1)
    residual_y=np.mean(residual_y,axis=1)
    for i in range(len(residual_X)):
        residual_X[i]=np.mean(residual_X[i],axis=0)
    for i in range(len(test_shap)):
        test_shap[i]=np.mean(test_shap[i],axis=0) 
    shap_orig=np.mean(shap_orig,axis=0) 

    list_df_=list_df.copy()
    for i, omics in enumerate(omics_name):
        pd.DataFrame(test_shap[i]).to_csv(f'{omics_name[i]}_shap.csv',index=False)
        list_df_[i]=list_df_[i].drop(["Unnamed: 0","ARM","visit","pid"],axis=1)
        list_df_[i]=list_df_[i].astype(float)
    #save test_result and original shap value
    final_result_temp_df=pd.DataFrame({'true_value': y,
                  "residual_y":residual_y,
                  'predict_value': test_result,
                                    "covariate_result": covariate_result
                  })
    for i, omics in enumerate(omics_name):
        final_result_temp_df[f"{omics}_predict_value"]=test_mod_result[:,i]
    final_result_temp_df.to_csv(f'late_fusion_result.csv',index=False)
    pd.DataFrame(shap_orig).to_csv(f'orig_shap.csv',index=False)
    pd.concat(list_df_,axis=1).to_csv(f'feature_value.csv',index=False)
    pd.DataFrame(np.concatenate(residual_X,axis=1)).to_csv(f'residual_feature_value.csv',index=False)
    return test_corr,test_p,test_result


# In[926]:


def covariate_adjustment(cov_X,cov_y,covariate_df,model_func,binary_feature_list,omics_name,cov_X_list=[],confounder_list=[],cov_y_list=[],k2=10):
    """
    Covariate adjustment leveraging double ML. Inner fold of the whole framework.
    
    Parameters:
    cov_X: outer fold training set predictors (omics).
    cov_y: outer fold training set outcome.
    covariate_df: outer fold training set covariate.
    model_func: SKlearn model function.
    binary_feature_list: list of binary features.
    omics_name:list of omics name for monitoring progress and saving results purposes.
    cov_X_list:list of covariates for predictors.
    confounder_list:list of confounders.
    cov_y_list:list of covariates for outcomes.
    k2: inner fold number.
    """
    ##generate result matrix
    explained_variance=[np.zeros((cov_X_.shape[0],cov_X_.shape[1]-4)) for cov_X_ in cov_X] #prediction results using covariates for predictors (omics)
    explained_variance_o=np.zeros(cov_y.shape[0]) #prediction results using covariates for outcome
    #generate inner fold splitting information for selecting top features in the main framework (for consistency)
    ifold_index=[]
    ##Double ML
    #initiate inner fold for double ML
    cov_kf=KFold(n_splits=k2, random_state=99, shuffle=True)
    for (cov_fold, (train_index_, test_index_)) in tqdm(enumerate(cov_kf.split(cov_X[0].pid.unique())),desc = 'fold'):
        #spliting inner train test set, ensuring samples from same mother-infant dyads are kept in the same set.
        train_index_=cov_X[0][cov_X[0].pid.isin(cov_X[0].pid.unique()[train_index_])].index.to_numpy()
        test_index_=cov_X[0][cov_X[0].pid.isin(cov_X[0].pid.unique()[test_index_])].index.to_numpy()
        train_X_, test_X_ = split(cov_X,train_index_), split(cov_X,test_index_)
        train_cov_df_, test_cov_df_ = covariate_df.iloc[train_index_], covariate_df.iloc[test_index_]
        train_y_, test_y_ = cov_y[train_index_], cov_y[test_index_]
        ifold_index.append((train_index_, test_index_))
        #Predicting HM components using covariates
        for i, omics in enumerate(omics_name):
            train_X_[i].drop(["Unnamed: 0","ARM","visit","pid"],axis=1,inplace=True)
            test_X_[i].drop(["Unnamed: 0","ARM","visit","pid"],axis=1,inplace=True)
            train_X_[i]=train_X_[i].astype(float)
            test_X_[i]=test_X_[i].astype(float)
            for f_i, col in enumerate(train_X_[i].columns):
                if col in binary_feature_list:
                    if len(np.unique(train_X_[i][col].values))==1:
                        model=XGBRegressor(**{"random_state":0}).fit(train_cov_df_[confounder_list+cov_X_list],train_X_[i][col].values)
                        test_prediction=model.predict(test_cov_df_[confounder_list+cov_X_list])
                        for j, sample in enumerate(test_index_):
                            explained_variance[i][sample,f_i]=test_prediction[j]
                    else:
                        model=XGBClassifier(**{"random_state":0}).fit(train_cov_df_[confounder_list+cov_X_list],train_X_[i][col].values)
                        test_prediction=model.predict_proba(test_cov_df_[confounder_list+cov_X_list])[:,1]
                        for j, sample in enumerate(test_index_):
                            explained_variance[i][sample,f_i]=test_prediction[j]
                else:
                    model=XGBRegressor(**{"random_state":0}).fit(train_cov_df_[confounder_list+cov_X_list],train_X_[i][col].values)
                    test_prediction=model.predict(test_cov_df_[confounder_list+cov_X_list])
                    for j, sample in enumerate(test_index_):
                        explained_variance[i][sample,f_i]=test_prediction[j]
        #Predicting outcomes using covariates
        model=model_func(**{"random_state":0}).fit(train_cov_df_[confounder_list+cov_y_list],train_y_)
        if is_classifier(model_func):
            test_prediction=model.predict_proba(test_cov_df_[confounder_list+cov_y_list])[:,1]
        else:
            test_prediction=model.predict(test_cov_df_[confounder_list+cov_y_list])
        for j, sample in enumerate(test_index_):
            explained_variance_o[sample]=test_prediction[j]
    return explained_variance,explained_variance_o,ifold_index


# In[ ]:


def GDeepShap_SG(model_list,baseline_list,explicand_list):
    ##Get attributions to original feature space
    #get shap value for explicand per baseline
    attr_list=[]
    for exp, explicand in enumerate(explicand_list):
        attr_list.append(np.zeros((baseline_list[exp].shape[0],explicand.shape[0],explicand.shape[1])))
        if is_classifier(model_list[exp]):
            model_output="probability"
        else:
            model_output="raw"
        for base in range(baseline_list[exp].shape[0]):
            explainer = shap.TreeExplainer(model_list[exp],data=baseline_list[exp].iloc[base].values.reshape(1,-1),
                                           feature_perturbation="interventional",model_output=model_output)
            attr_list[exp][base]=explainer.shap_values(explicand)
        attr_list[exp]=np.swapaxes(attr_list[exp],0,1) #shape(expl_bank.shape[0], refe_bank.shape[0], feature_number)
    # Rescale attributions
    attr_re_list=[]
    for att,attr in enumerate(attr_list[:-1]):
        expl_2 = np.repeat(model_list[att].predict(explicand_list[att], output_margin=True)[:,None],baseline_list[att].shape[0],axis=1) #shape[expl_num,refe_num]
        refe_2 = np.repeat(model_list[att].predict(baseline_list[att], output_margin=True)[None,:],explicand_list[att].shape[0],axis=0)
        denom = expl_2-refe_2
        numer = attr_list[-1][:,:,att]
        rescale = np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0) # Filter by denom. for instability
        attr_re_list.append(attr*rescale[:,:,None])
    # Get original space attributions
    attr_orig = np.zeros([explicand_list[0].shape[0],baseline_list[0].shape[0],
                          sum(arr.shape[1] for arr in explicand_list[:-1] # not including the meta learner
                             )])
    orig_f_i=0
    for attr_re in attr_re_list:
        for f_i in range(attr_re.shape[2]):
            attr_orig[:,:,orig_f_i] = attr_re[:,:,f_i]
            orig_f_i+=1
    return attr_orig.mean(1)


# In[15]:


from sklearn.base import is_classifier, is_regressor
import piecewise_regression
from sklearn.linear_model import LinearRegression
def top_feature(df,train_residual_X,train_residual_y,residual_X,residual_y,test_shap,test_corr,test_p,
                omic,params={},upperlimit=200,bootstrap_round=1,k=10):
    """
    Selecting top features based on shap value. Please see manuscript methods and materials for more details.
    
    Parameters:
    df: original omics data.
    train_residual_X: residuals train test for predictors
    train_residual_y: residuals train test for outcome
    residual_X:residuals test test for predictors
    residual_y:residuals test test for outcome
    test_shap: shap value
    test_corr: model result correlation coefficient
    test_p: model result p value
    omic: omic name for saving results.
    params: parameters for model
    upperlimit: the upper limit for top features to build model.
    bootstrap_round: iterations of the framework
    k: outer fold number of the framework.
                
    """
    #set up p value
    top_feature_dict={}
    #in case upperlimit is more than feature number
    upperlimit=min(upperlimit,df.shape[1]-4)
    # selecting top features
    for rounds in tqdm(range(bootstrap_round),desc = 'Round'):
        corr=np.random.random(upperlimit)
        p=np.random.random(upperlimit)
        test_result = np.random.random((upperlimit,df.shape[0]))
        feature_name=df.drop(["Unnamed: 0","ARM","visit","pid"],axis=1).columns[np.argsort(np.abs(test_shap[rounds]).mean(0))].tolist()
        for fold in tqdm(range(k),desc = 'fold'):
            train_X=train_residual_X[rounds,fold]
            test_index=np.isnan(train_X).all(axis=1)
            train_X=train_X[~test_index]
            train_y=train_residual_y[:,rounds,fold]
            train_y=train_y[~test_index]
            test_X=residual_X[rounds]
            test_X=test_X[test_index]
            test_y=residual_y[:,rounds]
            test_y=test_y[test_index]
            
            train_X=pd.DataFrame(train_X)
            test_X=pd.DataFrame(test_X)
            train_X.columns=df.drop(["Unnamed: 0","ARM","visit","pid"],axis=1).columns
            test_X.columns=df.drop(["Unnamed: 0","ARM","visit","pid"],axis=1).columns
            if (set(np.unique(train_y)).issubset({0, 1})) and (set(np.unique(test_y)).issubset({0, 1})):
                model_func=XGBClassifier
            else:
                model_func=XGBRegressor
            for i in range(upperlimit):
                #model,trainer set up | train, predict[feature_name[-1:-(1+i+1):-1]]
                model=model_func(**params).fit(train_X[feature_name[-1:-(1+i+1):-1]].values,train_y)
                #record train result(for meta learner training)
                if is_classifier(model_func):
                    test_prediction=model.predict_proba(test_X[feature_name[-1:-(1+i+1):-1]].values)[:,1]
                else:
                    test_prediction=model.predict(test_X[feature_name[-1:-(1+i+1):-1]].values)
                for j, sample in enumerate(np.where(test_index == True)[0]):
                    test_result[i,sample]=test_prediction[j]
        if (set(np.unique(residual_y[:,rounds])).issubset({0, 1})):
            cor_f=pointbiserialr
        else:
            cor_f=spearmanr
        for i in range(upperlimit):
            stat,pv=cor_f(residual_y[:,rounds].reshape(-1),test_result[i].reshape(-1))
            if (stat>=test_corr[rounds]) and (pv<=test_p[rounds]):
                top_feature_dict[rounds]=feature_name[-1:-(1+i+1):-1]
                break
            else:
                corr[i]=stat
                p[i]=pv
                if i == (upperlimit-1):
                    top_feature_dict[rounds]=np.hstack((corr.reshape(-1,1), p.reshape(-1,1)))
    with pd.ExcelWriter(f"FAA_{omic}.xlsx") as writer:
        for i in top_feature_dict.keys():
            pd.DataFrame(top_feature_dict[i]).to_excel(writer, sheet_name=f"round{i}", index=False)


# In[3]:


#select raws for every 
def split(list_df,index):
    result=[]
    for i,omic in enumerate(list_df):
        result.append(omic.iloc[index].reset_index(drop=True))
    return result


# In[ ]:


def draw_correlation(val_y,test_y,val_mod_result,val_result,test_mod_result,test_result,corr_func,omics_name):
    """
    evaluate model performance and draw plot.
    val_y:train outcome
    test_y:test outcome
    val_mod_result:train individual model result
    val_result:train meta model result
    test_mod_result:test individual model result
    test_result:test meta model result
    corr_func:function for evaluating model performance
    omics_name:list of omics name
    """
    val_p=[]
    test_p=[]
    test_corr=[]
    for i, omics in enumerate(omics_name):
        plt.figure(figsize=(8,4))
        plt.subplot(121)
        #calculate
        rho,p=corr_func(val_y.reshape(-1),val_mod_result[:,i].reshape(-1))
        val_p.append(p)
        MSE = mean_squared_error(val_y.reshape(-1), val_mod_result[:,i].reshape(-1))
        RMSE = math.sqrt(MSE)
        #plot
        plt.scatter(val_y.reshape(-1),val_mod_result[:,i].reshape(-1))
        plt.title("Train Result")
        plt.xlabel("True Value")
        plt.ylabel("Predict Value")
        ax = plt.gca()
        if round(p,3)<0.001:
            plt.text(0.5, 0.95, f"Spearman's \u03C1: {round(rho,3)},\nP value < 0.001,\nRMSE: {round(RMSE,3)}",horizontalalignment='center',verticalalignment='top',transform=ax.transAxes,weight='bold',bbox=dict(facecolor='tab:blue', alpha=0.5)) 
        else:
            plt.text(0.5, 0.95, f"Spearman's \u03C1: {round(rho,3)},\nP value: {round(p,3)},\nRMSE: {round(RMSE,3)}",horizontalalignment='center',verticalalignment='top',transform=ax.transAxes,weight='bold',bbox=dict(facecolor='tab:blue', alpha=0.5)) 
        
        plt.subplot(122)
        #calculate
        rho,p=corr_func(test_y.reshape(-1),test_mod_result[:,i].reshape(-1))
        test_p.append(p)
        test_corr.append(rho)
        MSE = mean_squared_error(test_y.reshape(-1), test_mod_result[:,i].reshape(-1))
        RMSE = math.sqrt(MSE)
        #plot
        plt.scatter(test_y.reshape(-1),test_mod_result[:,i].reshape(-1))
        plt.title("Test Result")
        plt.xlabel("True Value")
        plt.ylabel("Predict Value")
        ax = plt.gca()
        if round(p,3)<0.001:
            plt.text(0.5, 0.95, f"Spearman's \u03C1: {round(rho,3)},\nP value < 0.001,\nRMSE: {round(RMSE,3)}",horizontalalignment='center',verticalalignment='top',transform=ax.transAxes,weight='bold',bbox=dict(facecolor='tab:blue', alpha=0.5)) 
        else:
            plt.text(0.5, 0.95, f"Spearman's \u03C1: {round(rho,3)},\nP value: {round(p,3)},\nRMSE: {round(RMSE,3)}",horizontalalignment='center',verticalalignment='top',transform=ax.transAxes,weight='bold',bbox=dict(facecolor='tab:blue', alpha=0.5)) 
        plt.suptitle(omics)
        plt.show()
        plt.close()
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    #calculate
    rho,p=corr_func(val_y.reshape(-1),val_result.reshape(-1))
    val_p.append(p)
    MSE = mean_squared_error(val_y.reshape(-1), val_result.reshape(-1))
    RMSE = math.sqrt(MSE)
    #plot
    plt.scatter(val_y.reshape(-1),val_result.reshape(-1))
    plt.title("Train Result")
    plt.xlabel("True Value")
    plt.ylabel("Predict Value")
    ax = plt.gca()
    if round(p,3)<0.001:
        plt.text(0.5, 0.95, f"Spearman's \u03C1: {round(rho,3)},\nP value < 0.001,\nRMSE: {round(RMSE,3)}",horizontalalignment='center',verticalalignment='top',transform=ax.transAxes,weight='bold',bbox=dict(facecolor='tab:blue', alpha=0.5)) 
    else:
        plt.text(0.5, 0.95, f"Spearman's \u03C1: {round(rho,3)},\nP value: {round(p,3)},\nRMSE: {round(RMSE,3)}",horizontalalignment='center',verticalalignment='top',transform=ax.transAxes,weight='bold',bbox=dict(facecolor='tab:blue', alpha=0.5)) 
    plt.subplot(122)
    #calculate
    rho,p=corr_func(test_y.reshape(-1),test_result.reshape(-1))
    test_p.append(p)
    test_corr.append(rho)
    MSE = mean_squared_error(test_y.reshape(-1), test_result.reshape(-1))
    RMSE = math.sqrt(MSE)
    #plot
    plt.scatter(test_y.reshape(-1),test_result.reshape(-1))
    plt.title("Test Result")
    plt.xlabel("True Value")
    plt.ylabel("Predict Value")
    ax = plt.gca()
    if round(p,3)<0.001:
        plt.text(0.5, 0.95, f"Spearman's \u03C1: {round(rho,3)},\nP value < 0.001,\nRMSE: {round(RMSE,3)}",horizontalalignment='center',verticalalignment='top',transform=ax.transAxes,weight='bold',bbox=dict(facecolor='tab:blue', alpha=0.5)) 
    else:
        plt.text(0.5, 0.95, f"Spearman's \u03C1: {round(rho,3)},\nP value: {round(p,3)},\nRMSE: {round(RMSE,3)}",horizontalalignment='center',verticalalignment='top',transform=ax.transAxes,weight='bold',bbox=dict(facecolor='tab:blue', alpha=0.5)) 
    plt.suptitle("Stack Generalization")
    plt.show()
    plt.close()
    return test_corr,test_p

