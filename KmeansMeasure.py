import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, LabelBinarizer
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns 
from sklearn import metrics

#matplotlib.use('tkagg')

data = pd.read_excel('CleanedDataNetworkTraffic_v1.xlsx', nrows = 5000)
data.columns = [ 'No.', 'Time', 'Source', 'Destination', 'Protocol', 'Length', 'Info', 'Random']
test_data = pd.read_excel('SYNFlood50.xlsx', nrows = 50)
test_data.columns =[ 'No.', 'Time', 'Source', 'Destination', 'Protocol', 'Length', 'Info'] 
#View Data#
#print(data.describe())

#CleanData
#data['Source'].

#barchart
#srcipcount = data['Source'].value_counts()
#sns.set(style='darkgrid')
#sns.barplot(srcipcount.index, srcipcount.values, alpha=0.9)
#plt.title('Freq of Source IPs')
#plt.ylabel('Number of Occurences', fontsize=12)
#plt.xlabel('Source IP', fontsize=2)
#plt.show()

#boxplot
#data.boxplot('Length', rot = 30, figsize=(5,6))
#plt.savefig('mygraph.png')

#One hot encode source IP
ohe = OneHotEncoder()
transformed = ohe.fit_transform(data[['Source']]).toarray()
SrcIP_df = pd.DataFrame(transformed)
#print(transformed)
#print(SrcIP_df.head())


#one hot encode destinationIP
ohe2 = OneHotEncoder()
transformed2 = ohe2.fit_transform(data[['Destination']]).toarray()
DstIP_df = pd.DataFrame(transformed2)
#print(transformed2.toarray())
#print(ohe2.categories_)



###DropAddressesandAssembleTrainingData###
#print(data.head(n=10))
#data = data.drop(['No.'], axis=1)
data = data.drop(['Time'], axis=1)
data = data.drop(['Source'], axis=1)
data = data.drop(['Destination'], axis=1)
data = data.drop(['Info'], axis=1)
data = data.drop(['Protocol'],axis=1)
data = data.drop(['Random'], axis=1)
#print(data.head(n=10))i

TrainingData = pd.concat([data, SrcIP_df,DstIP_df], axis=1)
#print(TrainingData.head())

###BuildingModel###
km = KMeans(n_clusters = 5)
y_km=km.fit_predict(TrainingData)
#print('Distortion: %.2f' % km.inertia_)
distortion = ((TrainingData - km.cluster_centers_[y_km])**2.0).sum(axis = 1)
print('km intertia =', km.inertia_)
print('km distortion =', distortion.sum())

##Assign cluster variables to 7 cluster laels determined by kmeans###
cluster = km.labels_

#Creates the column with the cluster labels
TrainingData['cluster'] = km.labels_

#Creates the column with the distortion per row
TrainingData['distortion'] = distortion

#print('Modified Data Set\n', TrainingData.head(n=20))

# ###########Kmeans Silhouette Coefficient
print('The KMeans Silhouette Coefficient is ' , metrics.silhouette_score(TrainingData, y_km, metric='euclidean'))
print('*'*50)
distortions = []  # my computer can't handle this plot
for i in range(1,8):
    km = KMeans(n_clusters=i, init='k-means++', n_init=20, max_iter=1000, tol=1e-04, random_state=0)
    km.fit(TrainingData)
    distortions.append(km.inertia_)
    print('Cluster:', i)
    print('Distortion: %.2f' % km.inertia_)
plt.plot(range(1,8), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.show()
plt.savefig('Clusters')


###SCATTERPLOT###
#print(type(TrainingData)) #type = dataframe
cluster_dist = TrainingData[['cluster','distortion']]
print(cluster_dist)
#print(cluster_dist.iloc[2]) indexing by row
plt.clf()
plt.scatter(cluster_dist.index,cluster_dist['distortion'])
plt.xlabel('Index')
plt.ylabel('Distortion')
plt.show()
plt.savefig('Scatterplot')

###TestData(SYN50)###
##Preprocessing##
ohe3 = OneHotEncoder()
transformed3 = ohe3.fit_transform(test_data[['Source']]).toarray()
SrcIP2_df = pd.DataFrame(transformed3)

ohe4 = OneHotEncoder()
transformed4 = ohe4.fit_transform(test_data[['Destination']]).toarray()
DestIP2_df = pd.DataFrame(transformed4)

#Drop Uneccesary Data#
test_data = test_data.drop(['Time'], axis=1)
test_data = test_data.drop(['Source'], axis=1)
test_data = test_data.drop(['Destination'], axis=1)
test_data = test_data.drop(['Info'], axis=1)
test_data = test_data.drop(['Protocol'],axis=1)

TestData = pd.concat([test_data, SrcIP2_df,DestIP2_df], axis=1)
#print(TestData.head())
test_results = km.fit_predict(TestData)
#print(test_results)
TestData['cluster'] = km.labels_
TestData['Distortion'] = distortion 
#print(TestData)
test_cluster_dist = TestData[['cluster' , 'Distortion']]
print(test_cluster_dist)
#Train_Test = pd.concat([TrainingData, TestData], ignore_index = True, axis = 0)
#print(TrainingData.join(TestData, how="outer", lsuffix = '_left'))

###ScatterPlot(Training and Test Data)###
plt.clf()
plt.scatter(test_cluster_dist.index,test_cluster_dist['Distortion'])
plt.xlabel('Index')
plt.ylabel('Distortion')
plt.show()
plt.savefig('Scatterplot_train_test')
