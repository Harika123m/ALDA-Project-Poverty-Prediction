import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pprint import pprint

# We directly use the normalized data for Un supervised learning
df = pd.read_csv('TrainNormalized.csv')

df_y = df['Target']
df_x = df.drop(['Target'], axis=1)

k_means_clusters = [2, 4]
# when k=2, we assume that target values {1,2} are 1 and target variable {3,4} are 2
for cluster in k_means_clusters:
    kmeans = KMeans(n_clusters=cluster, random_state=42, algorithm='full', precompute_distances='auto', n_init=1, max_iter=1000).fit(df_x)
    df_x['kmeans_{}'.format(cluster)] = kmeans.labels_

# Let's merge the target back for comparison
df_x['Target'] = df_y
groups = df_x.groupby(['Target'])

# Find the mis classification in each group
# We find this by taking the major label in each group and
target_map = {
   'kmeans_{}'.format(c): {} for c in k_means_clusters
}
for cluster in k_means_clusters:
    print('\n\n')
    for target, group in groups:
        print('Target {} clusters {}'.format(target, cluster))
        counts = group['kmeans_{}'.format(cluster)].value_counts().to_dict()
        print(counts)
        target_map['kmeans_{}'.format(cluster)][target] = counts
    df_plot = pd.DataFrame(target_map['kmeans_{}'.format(cluster)]).T
    df_plot.columns.name = 'Cluster'
    ax = df_plot.plot( kind="bar", title='Kmeans - Distribution of {} Clusters against Actual Targets'.format(cluster))
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    plt.savefig('fig/k_means_cluster_{}.png'.format(cluster))
    plt.close()
pprint(target_map)
