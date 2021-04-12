import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv(r'Mall_Customers.csv')

algorithm = (KMeans(n_clusters = 4 ,
                    init='k-means++', 
                    random_state= 111  , 
                    algorithm='auto'))

X1 = df[['Age' , 'Spending Score (1-100)']].iloc[: , :].values

algorithm.fit(X1)
labels1 = algorithm.labels_  # assign the cluster number to customer 
centroids1 = algorithm.cluster_centers_ # tells the x,y value of all four clusters

df_age_spending = df.copy()
df_age_spending["cluster_num"] = labels1

X2 = df[['Annual Income (k$)' , 'Spending Score (1-100)']].iloc[: , :].values
algorithm = (KMeans(n_clusters = 5 ,
                    init='k-means++', 
                    n_init = 10 ,
                    max_iter=300, 
                    tol=0.0001,  
                    random_state= 111 , 
                    algorithm='auto'))

algorithm.fit(X2)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_

df_income_spending = df.copy()
df_income_spending["cluster_num"] = labels2

X3 = df[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)']].iloc[: , :].values
algorithm = (KMeans(n_clusters = 6 ,
                    init='k-means++', 
                    n_init = 10 ,
                    max_iter=300, 
                    tol=0.0001,  
                    random_state= 111 , 
                    algorithm='auto'))

algorithm.fit(X3)
labels3 = algorithm.labels_
centroids3 = algorithm.cluster_centers_

df_age_income_spending = df.copy()
df_age_income_spending['cluster_num'] = labels3


df_all_clusters = df.copy()
df_all_clusters['age_spending_cluster'] = labels1
df_all_clusters['income_spending_cluster'] = labels2
df_all_clusters['age_income_spending_cluster'] = labels3

df_all_clusters.to_csv('mall_customers_clustered1.csv')