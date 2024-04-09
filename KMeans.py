# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:02:42 2024

@author: B100631
"""

import numpy as np
import pandas as pd
import sys
pd.set_option('display.max_columns', None)
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(self, k=None, max_iter=50, weights=None, n_restarts=1, initialization='k-means++'):
        
        if not (initialization == 'k-means++' or initialization == 'random'):
            print('You can only choose between random and k-means++ initialization')
            sys.exit()
                          
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.assignments = None
        self.weights = weights
        self.data_mean = None
        self.data_std = None
        
        if self.k==None: #U slučaju da nije zadato k, dok ispitujem po silhuette score-u ne želim automatska restartovanja
            self.n_restarts_backup=n_restarts 
            self.n_restarts = 1
        else:
            self.n_restarts = n_restarts
        
        self.initialization=initialization
        self.cluster_size=None
        self.global_center=None
    
    
    def _find_best_k(self, data):
        
        best_k=None
        best_score=-1
    
        for k in range(3,8):
            self.k=k
            self.learn(data)
            silhouette_score=self._calculate_silhouette_score(data)
            if silhouette_score > best_score:
                best_score=silhouette_score
                best_k=k
        
        print('___________________________________')
        print('___________________________________')
        print('Best silhuette score is: ', best_score)
        print('Best k based on silhuette score : ', best_k)
        print('___________________________________')
        print('___________________________________')
        
        self.n_restarts = self.n_restarts_backup
        return best_k
    
    def _calculate_silhouette_score(self, data):

        n=len(data)
        
        labels=self.transform(data)
        
        data = (data - self.data_mean) / self.data_std
        
        data_ = data.to_numpy()
        n,m = data.shape
        D = np.zeros((n, n))
        s = np.zeros(n)

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((data_[i] - data_[j])**2))
                D[i, j] = dist
                D[j, i] = dist
                
        for i in range(n):
            
            a = np.mean([D[i, j] for j in range(n) if labels[j] == labels[i]])

            b = np.min([np.mean([D[i, j] for j in range(n) if labels[j] == c])
                       for c in range(self.k) if c != labels[i]])

            s[i] = (b - a) / max(a, b)
        
        return np.mean(s)
     
        
    
    def _initialize_centroids(self, data):
        
        centroids = [data.sample().iloc[0]]
        min_distances_sq = np.zeros(len(data))

        for _ in range(1, self.k):

            for i, point in enumerate(data.values):
                min_distances_sq[i] = np.min(np.sum((np.array(centroid) - point) ** 2) for centroid in centroids)

            probabilities = min_distances_sq / np.sum(min_distances_sq)
            new_centroid = data.sample(weights=probabilities).iloc[0]
            centroids.append(new_centroid)

        return pd.DataFrame(np.vstack(centroids))
    
    def _evaluate_clustering(self, data):
       
        # Calculate global center
        global_center = data.mean().to_numpy() 
        
        # Calculate BSS/TSS ratio
        bss = sum((sum(((global_center - np.array(self.centroids))**2).T)*self.cluster_size))
        tss = sum(sum(((global_center - data.to_numpy())**2)))
        
        ratio = bss / tss
        
        # Warn about poorly represented clusters
        if ratio < 0.6:
            print(f'There are clusters that aren\'t represented well by their centroid (BSS/TSS={ratio:.3f})')
            
        # Calculate distances between centroids
        centroid_distances = cdist(self.centroids, self.centroids)
          
        # Warn about clusters that are too similar
        threshold = 0.2
        for i in range(self.k):
            for j in range(i + 1, self.k):
                if centroid_distances[i, j] < threshold:
                    print(f"Centroids {i} and {j} are similar (distance={centroid_distances[i, j]:.3f})")
                            
                    
    def learn(self, data):
        best_quality = float('inf')
        best_centroids = None
        best_assignments = None
        
        if len(self.weights) == 0:
            self.weights = np.ones((data.shape[1],))
            
        if self.k==None:
            self.k=self._find_best_k(data)    
         
        for n_iteration in range(self.n_restarts):
            n, m = data.shape
            
            print(f'k = {self.k}, try {n_iteration+1}')
            
            # Normalizacija
            self.data_mean = data.mean()
            self.data_std = data.std()
            normalized_data = (data - self.data_mean) / self.data_std
            self.global_center=normalized_data.mean().to_numpy()
            
            # Inicijalizacija
            if self.initialization == 'kmeans++':
                centroids = self._initialize_centroids(normalized_data)
            else:
                centroids = normalized_data.sample(self.k).reset_index(drop=True)
            
            assignments = np.zeros((n, 1))
            old_quality = float('inf')
            
            for iteration in range(self.max_iter):
                quality = np.zeros(self.k)
                
                # 1. Dodela tačaka klasterima
                for i in range(n):
                    instance = normalized_data.iloc[i]
                    if self.weights is not None:
                        distances = (((instance - centroids) ** 2) * self.weights).sum(axis=1)
                    else:
                        distances = ((instance - centroids) ** 2).sum(axis=1)
                    assignments[i] = np.argmin(distances)
                
                # 2. Preračunavanje centroida
                for c in range(self.k):
                    subset = normalized_data[assignments.flatten() == c]
                    centroids.loc[c] = subset.mean()
                    quality[c] = subset.var().sum() * len(subset)
                
                total_quality = quality.sum()
                
                if old_quality == total_quality:
                    print(f'Converged at iteration {iteration+1} due to cluster quality not improving')
                    print(f'Best quailty for try {old_quality}')
                    break
                old_quality = total_quality
                
                if (iteration == self.max_iter - 1):
                    print(f'Converged at iteration {iteration+1} due to max number of iterations reached')
                    print(f'Best quailty for try {old_quality}')
                
            # Čuvanje najboljeg modela
            if total_quality < best_quality:
                best_quality = total_quality
                best_centroids = centroids
                best_assignments = assignments
                cluster_size = pd.DataFrame(best_assignments).value_counts().sort_index()
        
        # Postavljanje najboljeg modela kao konačnog modela
        self.centroids = best_centroids
        self.assignments = best_assignments
        self.cluster_size= cluster_size
        
        print('**********')
        print('Best clustering quality overall: ', best_quality)
        
        self._evaluate_clustering(normalized_data)
    
    def transform(self, data):

        normalized_data = (data - self.data_mean) / self.data_std

        n = len(data)
        assignments = np.zeros(n)
        for i in range(n):
            case = normalized_data.iloc[i]
            distances = ((case - self.centroids) ** 2).sum(axis=1)
            assignments[i] = np.argmin(distances)
        
        return assignments.astype(int)
    
#%% EXAMPLE OF USE

weights = np.array([2, 1, 1, 1, 2, 5, 3, 2, 4, 4, 3, 4, 1, 4])
kmeans = KMeans(k=3, max_iter=50, weights=weights, n_restarts=4, initialization='k-means++')
#kmeans = KMeans(max_iter=50, weights=weights, n_restarts=4, initialization='k-means++')

data = pd.read_csv('data/boston.csv')

kmeans.learn(data)

cluster_assignments = kmeans.transform(data)

cluster_assignments_df = pd.DataFrame(cluster_assignments, columns=['Cluster'])

combined_df = pd.concat([data, cluster_assignments_df], axis=1)

#%% DATA DICTIONARY - BOSTON SET

"""
CRIM: Stopa kriminala po glavi stanovnika po kvartu.
ZN: Procenat rezidencijalne zemlje izgrađene na parcelama većim od 25.000 kvadratnih stopa.
INDUS: Procenat neretail biznis hektara po kvartu.
CHAS: Dummy varijabla za reku Charles (1 ako parcela graniči reku; 0 inače).
NOX: Koncentracija azotnog oksida (delovi na milion).
RM: Prosečan broj soba po stanu.
AGE: Procenat zgrada koje su izgrađene pre 1940. godine.
DIS: Težišna rastojanje do pet bostonskih centara zapošljavanja.
RAD: Indeks pristupa autoputevima.
TAX: Poreska stopa na imovinu u vrednosti od 10.000 dolara.
PTRATIO: Učiteljski odnos u kvartu (broj učenika po učitelju).
B: 1000(Bk - 0.63)^2 gde je Bk procenat Afroamerikanaca po kvartu.
LSTAT: Procenat stanovništva sa nižim statusom.
MEDV: Medijana vrednost kuće u hiljadama dolara.
"""




