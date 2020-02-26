import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_excel('Data_Science_2020_v2.xlsx')
df.shape
df.head()
df.info()
df.describe(include = 'object')
df.isnull().sum()
pd.set_option('display.max_columns',None)
df.head(2)

df['Degree'].value_counts()
df['Other skills'].value_counts()
df['Stream'].value_counts()
df['Current Year Of Graduation'].value_counts()

df['Performance_PG'] = df['Performance_PG'].transform(lambda x: x.fillna('No Degree'))
df1 = df['Performance_UG'].str.split('/',expand=True)
df2 = df['Performance_12'].str.split('/',expand=True)
df3 = df['Performance_10'].str.split('/',expand=True)

df1.drop(1,axis = 1, inplace = True)
df2.drop(1,axis = 1, inplace = True)
df3.drop(1,axis = 1, inplace = True)

df1 = df1.rename(columns={0:'Performance_UG'})
df2 = df2.rename(columns= {0:'Performance_12'})
df3 = df3.rename(columns= {0:'Performance_10'})

scores = pd.concat([df1,df2,df3],axis = 1)
scores.head()

df = df.drop(['Performance_PG','Performance_UG','Performance_12','Performance_10'],axis = 1)

data = pd.concat([df,scores],axis = 1)
data.head()

data['Performance_UG'] = data['Performance_UG'].astype(float)
data['Performance_12'] = data['Performance_12'].astype(float)
data['Performance_10'] = data['Performance_10'].astype(float)

data['Performance_UG'] = data['Performance_UG'].transform(lambda x: x.fillna(data['Performance_UG'].mean()))
data['Performance_12'] = data['Performance_12'].transform(lambda x:x.fillna(data['Performance_12'].mean()))
data['Performance_10'] = data['Performance_10'].transform(lambda x: x.fillna(data['Performance_10'].mean()))

plt.subplots(figsize = (11,11))
sns.heatmap(data.corr(),square = True,annot = True)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

label = LabelEncoder()
std = StandardScaler()

data['Other skills'] = data['Other skills'].transform(lambda x:x.fillna('unknown'))
data['Stream'] = data['Stream'].transform(lambda x:x.fillna('unknown'))
data['Degree'] = data['Degree'].transform(lambda x:x.fillna('unknown'))

data_labeled = data.apply(label.fit_transform)
data_scaled = std.fit_transform(data_labeled)

data_scaled = pd.DataFrame(data_scaled,columns = data.columns)
data_scaled = data_scaled.drop('Application_ID',axis = 1)

sns.pairplot(data = data_scaled)

from sklearn.decomposition import PCA
pca = PCA()

pc = pca.fit_transform(data_scaled)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_,color = 'red')
plt.xlabel('PCA Features')
plt.ylabel(('Variance'))
plt.xticks(features)

PCA_elements = pd.DataFrame(pc,columns = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11','pc12'])
PCA_elements.head()

plt.subplots(figsize = (10,10))
sns.heatmap(PCA_elements.corr(),square = True,annot = True)

#Cluster analysis to determine k 

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, random_state = 42)
    kmeans.fit(PCA_elements)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 15), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

clusters_df = pd.DataFrame( { "num_clusters":range(1,15), "wcss": wcss } )
clusters_df[0:15]

kmeans = KMeans(n_clusters = 2,init = 'k-means++',n_init = 10,max_iter = 300,random_state = 0)
kmeans.fit(PCA_elements)
centroids = kmeans.cluster_centers_
centroid_fd = pd.DataFrame(centroids,columns = list(PCA_elements))

df_labels = pd.DataFrame(kmeans.labels_,columns = list(['labels']))
df_labels['labels'] = df_labels['labels'].astype('category')

select_df_label = PCA_elements.join(df_labels)

selection_clusters = select_df_label.groupby(['labels'])
df0 = selection_clusters.get_group(0)
df1 = selection_clusters.get_group(1)
finaldf = pd.concat([df0,df1])
finaldf.head()


final_data = pd.concat([data_labeled,finaldf['labels']],axis = 1)

#Predicting whether a student will be selected or not

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

x = final_data.drop(['Application_ID','labels'],axis = 1)
y = final_data['labels']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state =0)

array = [LogisticRegressionCV(),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=50,n_jobs=5),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        SVC(),
        GaussianNB()]

for i in range(0,len(array)):
    array[i].fit(x_train,y_train)
    
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,classification_report    
l =[]
for i in range(0,len(array)):
    y_pred = array[i].predict(x_test)
    l.append(accuracy_score(y_pred,y_test))
    
algorithms = pd.DataFrame(l,index=('Logistic regreesion','Decision tree','Randomforest','Ada boost','Gradient boost', 'Kernel SVM','Naive_Bayes'),columns=['Accuracy score'])
algorithms    

nb = GaussianNB()
nb.fit(x_train,y_train)
y_pred_nb = nb.predict(x_test)


#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred_nb)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_nb))
