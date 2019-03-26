
# coding: utf-8

# In[122]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[124]:


# import features on image scoring database
featuresOpenCvDF = pd.read_csv('D:\\google drive\\A PhD Project at Godlsmiths\\ArtistSupervisionProject\\code\\fractevoScores29apr2018.csv')
featuresOpenCvDF.columns

from ast import literal_eval
# import DATA from MUTATOR Capture
D=pd.read_table('D:\\google drive\\organic\\Machine Learning\\MLCapture.txt')
D.set_index('id')
#remove rr1 rr2 rr3 rr4 as they are nan
D.drop('rr1', axis=1, inplace=True)
D.drop('rr2', axis=1, inplace=True)
D.drop('rr3', axis=1, inplace=True)
D.drop('rr4', axis=1, inplace=True)
# apply literal to columns with list [-1 to 1 ] so they can be read as list not as string
D['startrot4'] = D['startrot4'].apply(literal_eval)
D['endrot4'] = D['endrot4'].apply(literal_eval)

# merge the two database on a single dataframe
featuresOpenCvDF.rename(columns = {'file':'id'}, inplace = True)
featuresOpenCvDF.columns
# merged dataframe
df = pd.merge(featuresOpenCvDF, D, on = 'id')


# drop the rows (frames) where judgement is '?'
df = df[df.judge != '?']
print (df.shape)

# drop the rows where parent id has value (delete judgement made twice)
numofframes = df.shape[0]
for f in range(numofframes):
    
    df = df[df.parentid != f]
print (df.shape)


# drop all the rows from entries id 38 to 97 cause error while judgements
# drop the rows where parent id has value (delete judgement made twice)

for f in range(38,98):
    
    df = df[df.id != f]
print (df.shape)
#df = df.drop(df.index[39:99])


# make the judge as type integer
df['judge'] = df['judge'].values.astype(int)


# In[125]:


print (df.columns.values)


# In[126]:


dfID = df[['id']]
dfLabel = df[['judge']]
# drop the lines of judges to simplify the dataset into 2 classes
#df = df[df.judge != 1 ]
#df = df[df.judge != 3 ]
#df = df[df.judge != 4 ]
# =============================================================================
# '''
# df = df[['Y_fractalMinkowski' ,'Y_ssimAsymmetry',
#  'Y_distanceRatioOfMassVsImageCentre' ,'Y_ratioSilhoutte', 'Y_orbVsArea',
#  'N_faceScore' , 'Y_percentageWarm' ,'Y_numberOfHolesWeighted',
#  'Y_PerimeterRatioConvexity' ,'Y_scoreBrightness', 'Y_Colorfulness',
#  'Y_contrastScore', 'Y_meanSaturation', 'Y_totalCorner', 'Y_ruleofthird',
#  'Y_chaosEdges', 'Y_n_blobs_Lorenzo', 'Y_totalAreaLorenzoWeigthed',
#  'Y_inflatedAreaSum', 'Y_numberOfHolesLorenzo', 'Y_numberOfFeaturesLor',
#  'Y_analyse_clear_silh_Lor', 'Y_perimeterCntVsHolesScore' ]]
# '''
# df = df[[  'Y_analyse_clear_silh_Lor', 'Y_inflatedAreaSum',
#         'Y_numberOfFeaturesLor', 'Y_contrastScore', 
#         'Y_scoreBrightness', 'Y_percentageWarm' , 'Y_ssimAsymmetry', 'Y_perimeterCntVsHolesScore']]
# =============================================================================
df = df[['areaLorenzo',
                      #'nFeatures',
                      'nBlobs',
                      'nHoles',
                      'asymmetryScore',
                      'clearSilhoutte',
                      'centreOfMass',
                      'fractalScore',
                      'longLenghtSilhoutteReward',
                      'largeAreaReward',
                      'jaggedlySilhoutteReward',
                      'colorfulnessReward',
                      'contrastReward'
                      ]]

# print the colums
print (df.columns.values)
print (dfLabel.columns.values)


# # Clustering for exploration of features

# In[127]:


from sklearn.cluster import KMeans


# In[128]:


kmeans = KMeans(n_clusters = 5)


# In[129]:


kmeans.fit(df)


# In[130]:


# to get the centers of the clusters
kmeans.cluster_centers_


# In[131]:


# the labels it believes are true for the clusters
kmeans.labels_


# In[132]:


sns.distplot(kmeans.labels_,kde = True , bins = 5 )


# In[133]:


for col in df.columns:
    
    sns.jointplot (x = col, y = kmeans.labels_, data = df, kind = 'reg', color = 'red')


# In[134]:


fig, (ax1, ax2) = plt.subplots(1,2, sharey = True, figsize=(10,6))

ax1.set_title('K_Means')
ax1.scatter(df['longLenghtSilhoutteReward'], df['jaggedlySilhoutteReward'], c=kmeans.labels_, cmap ='rainbow')

ax2.set_title('Original')
ax2.scatter(df['longLenghtSilhoutteReward'], df['jaggedlySilhoutteReward'], c=dfLabel['judge'], cmap ='rainbow')


# sns.lmplot(x = 'ratioSilhoutte', y ='ssimAsymmetry', data = df, hue = 'judge',
#           fit_reg = False, palette = 'coolwarm', size=6, aspect=1)

# g = sns.FacetGrid(df, hue='judge', palette='coolwarm', size = 6, aspect = 2 )
# g = g.map(plt.hist, 'ssimAsymmetry', bins = 20, alpha = 0.7)

# In[135]:


PredDf = pd.DataFrame ( data = kmeans.labels_, columns = ['pred'])


# create a new column to store the clusters
def converter (judge):
    if judge == 1:
        return 1
    elif judge == 2:
        return 2
    elif judge == 3:
        return 3
    elif judge == 4:
        return 4
    else:
        return 5

PredDf['pred'] = PredDf['pred'].apply(converter)


# In[139]:


# create dataframe of combined original plu new labels
dfID = pd.DataFrame.reset_index(dfID)
dfLabel = pd.DataFrame.reset_index(dfLabel)

dfID['pred'] = PredDf['pred']
dfID['judge'] = dfLabel.judge
dfID


# In[137]:


from sklearn.metrics import confusion_matrix, classification_report


# In[150]:


# put back the classes to match the original data labels

km = pd.DataFrame(data = kmeans.labels_ , columns = ['labels_'])
km['labels_'] = km['labels_'].apply(converter)

print(confusion_matrix(dfLabel['judge'], km['labels_']))
print ('\n')
print(classification_report(dfLabel['judge'], km['labels_']))



# In[151]:


# export the data csv for later use in rename for folder 
dfID.to_csv('Kmeans_judge_pred.csv', index = False)


# In[ ]:




