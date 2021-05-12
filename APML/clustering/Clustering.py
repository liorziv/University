import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
from sklearn import datasets
import random
import pandas as pd

def circles_example():
    """
    an example function for generating and plotting synthesised data.
    """

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.matrix([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.matrix([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.matrix([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.matrix([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1).T
    circles = np.squeeze(np.asarray(circles))

    #plt.plot(circles[0,:], circles[1,:], '.k')
    #plt.show()
    return circles

def apml_pic_example(path='APML_pic.pickle'):
    """
    an example function for loading and plotting the APML_pic data.

    :param path: the path to the APML_pic.pickle file
    """

    with open(path, 'rb') as f:
        apml = pickle.load(f)

    # plt.plot(apml[:, 0], apml[:, 1], '.')
    # plt.show()
    return apml

def microarray_exploration(data_path='microarray_data.pickle',
                            genes_path='microarray_genes.pickle',
                            conds_path='microarray_conds.pickle'):
    """
    an example function for loading and plotting the microarray data.
    Specific explanations of the data set are written in comments inside the
    function.

    :param data_path: path to the data file.
    :param genes_path: path to the genes file.
    :param conds_path: path to the conds file.
    """

    # loading the data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(genes_path, 'rb') as f:
        genes = pickle.load(f)
    with open(conds_path, 'rb') as f:
        conds = pickle.load(f)

    # look at a single value of the data matrix:
    i = 128
    j = 63
    print("gene: " + str(genes[i]) + ", cond: " + str(
        conds[0, j]) + ", value: " + str(data[i, j]))

    # make a single bar plot:
    plt.figure()
    plt.bar(np.arange(len(data[i, :])) + 1, data[i, :])
    #plt.show()

    # look at two conditions and their correlation:
    plt.figure()
    plt.scatter(data[:, 27], data[:, 29])
    plt.plot([-5,5],[-5,5],'r')
    #plt.show()

    # see correlations between conds:
    correlation = np.corrcoef(np.transpose(data))
    plt.figure()
    plt.imshow(correlation, extent=[0, 1, 0, 1])
    plt.colorbar()
    #plt.show()

    # look at the entire data set:
    plt.figure()
    plt.imshow(data, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    plt.colorbar()
    # plt.show()

    return data

def euclid(X, Y):
    """
    return the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """
    N,D = X.shape
    M = Y.shape[0]
    res = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            res[i,j] = np.linalg.norm(X[i,:]- Y[j,:])
    return res


def maxi(X, Y):
    """
    return the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """
    N,D = X.shape
    M = Y.shape[0]
    res = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            res[i,j] = np.linalg.norm(X[i,:]- Y[j,:])
    return res

def euclidean_centroid(X):
    """
    return the center of mass of data points of X.
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster.
    """

    N,D = X.shape

    return np.sum(X,axis=0)/N

def kmeans_pp_init(X, k, metric):
    """
    The initialization function of kmeans++, returning k centroids.
    :param X: The data matrix.
    :param k: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """
    N,D = X.shape
    #initialize the probabilities - uniformly
    probabilities = np.full((N),1/N)
    point_num = np.array(range(N))
    init_centers = []

    #choose centroieds by k-means++ algorithm
    for i in range(k):
        #chose a point
        chosen_point = np.random.choice(point_num,size=1,p=probabilities)
        init_centers.append((X[chosen_point,:]).reshape((D)))
        #calc the distance of centeroides to points
        dist_mat = euclid(X,np.array(init_centers))
        #calculate the new probabilities
        minimal_in_row = np.min(dist_mat,axis=1)
        minimal_in_row_2 = np.power(minimal_in_row,2)
        probabilities = minimal_in_row_2/np.sum(minimal_in_row_2)

    return np.array(init_centers)

def silhouette(X, clusterings, centroids):
    """
    Given results from clustering with K-means, return the silhouette measure of
    the clustering.
    :param X: The NxD data matrix.
    :param clustering: A list of N-dimensional vectors, each representing the
                clustering of one of the iterations of K-means.
    :param centroids: A list of kxD centroid matrices, one for each iteration.
    :return: The Silhouette statistic, for k selection.
    """
    N,D = X.shape
    k = centroids.shape[0]
    ai = np.zeros((N,1))
    bi = np.full((N,1),np.inf)


    for i in range(N):
        for C_i in range(k):
            idx =  clusterings == C_i
            dist = np.sum(euclid(X[idx,:],X[i,:].reshape(1,D)))/idx.sum()
            if(clusterings[i] == C_i):
                ai[i] = dist
            else:
                if(bi[i] > dist):
                    bi[i] = dist


    return np.sum((bi-ai)/np.amax(np.row_stack((ai,bi)),axis = 0))

def kmeans(X, k, iterations=1, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init, stat=silhouette):
    """
    The K-Means function, clustering the data X into k clusters.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :param stat: A function for calculating the statistics we want to extract about
                the result (for K selection, for example).
    :return: a tuple of (clustering, centroids, statistics)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    statistics - whatever data you choose to use for your statistics (silhouette by default).
    """
    optimal_sum = np.inf
    optimal_labels = []
    for i in range(iterations): #different initializations
        centroids = init(X=X,k=k,metric=metric)
        prev_labels = np.ones((X.shape[0],1))
        curr_labels = np.zeros((X.shape[0],1))

        #untill we converge
        while(prev_labels != curr_labels).any():
            dist_mat = metric(X, centroids)  # centroids in column
            prev_labels = curr_labels
            curr_labels = np.argmin(dist_mat, axis=1)

            #calculate new centroids
            for C_i in range(k):
                C_i_group = X[curr_labels==C_i,:]
                centroids[C_i,:] = center(C_i_group)

        #check if the current labels give the optimal clustering
        tmp_score = score_kmeans(X, centroids, k, curr_labels, metric)
        if(tmp_score < optimal_sum):
            optimal_sum = tmp_score
            optimal_labels = curr_labels


    return optimal_labels,centroids,silhouette(X,optimal_labels,centroids)

def score_kmeans(X,centroids,k,labels,metric):
    '''
    calculates the total distances of the points from their centeroids
    :param X: point set NXD
    :param centroids: centroids of the data 1XK
    :param k: number of centroids
    :param labels: vector containing the labels of each point
    :param metric: for calculating distance
    :return: summed squared distance between points and their centroids
    '''
    dist_mat = metric(X, centroids)  # centroids in column
    D = X.shape[1]
    labels = np.argmin(dist_mat, axis=1)
    total_dist = np.zeros((k,1))
    for C_i in range(k):
        C_i_group = X[labels == C_i, :]
        total_dist[C_i] = np.sum((metric(C_i_group,centroids[C_i,:].reshape(1,D)))**2)

    return np.sum(total_dist)

def heat(X, sigma):
    """
    calculate the heat kernel similarity of the given data matrix.
    :param X: A NxD data matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """

    dist_mat = euclid(X,X)
    return np.exp(-(dist_mat**2)/(2*sigma**2))

def mnn(X, m):
    """
    calculate the m nearest neighbors similarity of the given data matrix.
    :param X: A NxD data matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """
    N, D = X.shape
    dist_mat = euclid(X, X)
    sim_mat = np.zeros((N, N))
    for i in range(N):
        idxs = np.argsort(dist_mat[i, :])
        sim_mat[idxs[1:m+1], i] = 1
        sim_mat[i,idxs[1:m+1]] = 1

    return sim_mat

def spectral(X, k, similarity_param, similarity=heat):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the hear kernel.
    :param similarity: The similarity transformation of the data.
    :return: clustering, as in the kmeans implementation.
    """

    #calculate the laplasian matrix
    W = similarity(X,similarity_param)
    D_sqrt = np.sqrt(np.diag(1/np.sum(W,axis=1)))
    L = np.identity(X.shape[0]) - D_sqrt@W@D_sqrt

    #eigen values of the L matrix
    eigen_values,U = np.linalg.eig(L)
    idx = np.argsort(eigen_values)
    U = U[:,idx]
    k_eigen_vals = U[:,0:k]

    #In order to see the eigan gap
    plt.figure()
    plt.plot(range(len(eigen_values)),eigen_values)
    plt.show
    #run k means on L

    return kmeans(k_eigen_vals,k,100)

def choose_k():
    '''
    This function checks with two different data set the
    elbow method and the silhuette which are methods to choose k
    '''

    #create synthtic data - 3 Gaussians
    x1 = np.random.normal(0, 1, 40)
    x1.resize(20, 2)
    x2 = np.random.normal(5, 1, 40)
    x2.resize(20, 2)
    x3 = np.random.normal(10, 1, 40)
    x3.resize(20, 2)
    X = np.row_stack((x1, x2, x3))


    fig1 = plt.figure(figsize=(10,10))
    ax1 = fig1.add_subplot(1, 3, 1)
    ax1.scatter(x1[:,0], x1[:, 1],color='red')
    ax1.scatter(x2[:, 0], x2[:, 1],color= 'blue')
    ax1.scatter(x3[:, 0], x3[:, 1], color='green')
    ax1.set_title("Original data")

    #using apml 2017 fig
    ampl_pic = apml_pic_example()
    fig = plt.figure(figsize=(10,10))
    ax12 = fig.add_subplot(1, 3, 1)
    ax12.scatter(ampl_pic[:,0],ampl_pic[:,1])
    ax12.set_title('Original Data')

    score_km_apml = []
    sill_list_apml = []
    score_km_Gau = []
    sill_list_Gau = []
    for i in range(2,10):
        labels_a, centers_a, sill_a = kmeans(ampl_pic, i, 30)
        score_km_apml.append(score_kmeans(ampl_pic,centers_a,i,labels_a,euclid))
        sill_list_apml.append(sill_a)

        labels_g, centers_g, sill_g = kmeans(X, i, 30)
        score_km_Gau.append(score_kmeans(X, centers_g, i, labels_g, euclid))
        sill_list_Gau.append(sill_g)


    ax22 = fig.add_subplot(1,3,2)
    ax22.plot(range(2,10),score_km_apml)
    ax22.set_title('Variance')
    ax32 = fig.add_subplot(1,3,3)
    ax32.plot(range(2,10),sill_list_apml)
    ax32.set_title('Silhouette')
    fig.savefig('AMPL_DATA');
    plt.show()

    ax2 = fig1.add_subplot(1, 3, 2)
    ax2.plot(range(2, 10), score_km_Gau)
    ax2.set_title('Variance')
    ax3 = fig1.add_subplot(1, 3, 3)
    ax3.plot(range(2, 10), sill_list_Gau)
    ax3.set_title('Silhouette')
    fig1.savefig('GAUSSIANS_DATA');
    plt.show()

def run_spectral_clustering_mnn():
    '''
    This function demostrates the results of spectral clustering on
    two different data sets, using mnn
    '''

    #using apml 2017 fig
    ampl_pic = apml_pic_example()
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(ampl_pic[:, 0], ampl_pic[:, 1])
    ax1.set_title('Original Data')
    ax2 = fig.add_subplot(1,2,2)
    labels_a, centers_a, sill_a = spectral(ampl_pic, 8, 8, mnn)
    ax2.scatter(ampl_pic[:,0],ampl_pic[:,1],c=labels_a)
    ax2.set_title('k = 8 , N = 8')
    fig.savefig('AMPL_DATA_spec_mnn');
    plt.show()

    # using circels example
    circles = circles_example()
    fig2 = plt.figure(figsize=(10, 10))
    ax1 = fig2.add_subplot(1,2, 1)
    ax1.scatter(circles[:,0], circles[:,1])
    ax1.set_title('Original Data')
    ax2 = fig2.add_subplot(1, 2, 2)
    labels_c, centers_c, sill_c = spectral(circles, 4,7, mnn)
    ax2.scatter(circles[:,0], circles[:,1], c=labels_c)
    ax2.set_title('k = 4 N = 7')
    fig2.savefig('CIRC_DATA_spec_mnn');
    plt.show()

def run_spectral_clustering_heat():
    '''
    This function demostrates the results of spectral clustering on
    two different data sets, using heat kernel
    '''
    #using apml 2017 fig
    ampl_pic = apml_pic_example()
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(ampl_pic[:, 0], ampl_pic[:, 1])
    ax1.set_title('Original Data')
    ax2 = fig.add_subplot(1,2,2)
    labels_a, centers_a, sill_a = spectral(ampl_pic, 8, 10, heat)
    ax2.scatter(ampl_pic[:,0],ampl_pic[:,1],c=labels_a)
    ax2.set_title('k = 8 sigma = 10')
    fig.savefig('AMPL_DATA_spec');
    plt.show()

    # using circels example
    circles = circles_example()
    fig2 = plt.figure(figsize=(10, 10))
    ax1 = fig2.add_subplot(1,2, 1)
    ax1.scatter(circles[:,0], circles[:,1])
    ax1.set_title('Original Data')
    ax2 = fig2.add_subplot(1, 2, 2)
    labels_c, centers_c, sill_c = spectral(circles, 4,0.1, heat)
    ax2.scatter(circles[:,0], circles[:,1], c=labels_c)
    ax2.set_title('k = 4 , N = 0.1')
    fig2.savefig('CIRC_DATA_spec');
    plt.show()

def run_on_microarray():

    total_data = microarray_exploration()
    idx = np.random.choice(range(total_data.shape[0]),size= 500)
    data = total_data[idx,:]
    labels,centroids,sil = spectral(data, 30, 7, heat)
    idx = np.argsort(labels)
    data = data[idx,:]
    plt.figure()
    plt.imshow(data, aspect='auto')
    plt.title('Spectral Clustering')
    plt.show()

    labels_2,centroids_2,sil_2 = kmeans(data, 30, 1)
    idx_2 = np.argsort(labels_2)
    data = data[idx_2,:]
    plt.figure()
    plt.imshow(data, aspect='auto')
    plt.title('K - means')
    plt.show()

def run_kmeans():
    '''
       This function demostrates the results of spectral clustering on
       two different data sets, using heat kernel
       '''
    # using apml 2017 fig
    ampl_pic = apml_pic_example()
    plt.figure(figsize=(10, 10))
    labels_a, centers_a, sill_a = kmeans(ampl_pic, 8, 8)
    plt.scatter(ampl_pic[:, 0], ampl_pic[:, 1], c=labels_a)
    plt.title('Kmeans k =8 N = 8')
    plt.savefig('AMPL_DATA_kmeans');
    plt.show()

    # using circels example
    circles = circles_example()
    plt.figure(figsize=(10, 10))
    labels_c, centers_c, sill_c = kmeans(circles, 4, 7)
    plt.scatter(circles[:, 0], circles[:, 1], c=labels_c)
    plt.title('K-means K = 4, N = 7')
    plt.savefig('CIRC_DATA_K-means');
    plt.show()


def run_TSNE():

    #take the data and create labels using spctral clustering
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target
    digits_r = TSNE(n_components=2).fit_transform(data)
    #show the TSNE
    plt.figure(figsize=(10, 10))
    plt.scatter(digits_r[:,0],digits_r[:,1],c = labels)
    plt.title('digits using TSNE')
    plt.savefig('DIGITS_TSNE')
    plt.show()


    # take the data and create labels using spctral clustering
    s_curve,labels_2 = datasets.samples_generator.make_s_curve(1000,random_state=0)

    s_curve_r = TSNE(n_components=2,init='pca').fit_transform(s_curve)
    # show the TSNE
    plt.figure(figsize=(10, 10))
    plt.scatter(s_curve_r[:, 0], s_curve_r[:, 1], c=labels_2)
    plt.title('S curve using TSNE')
    plt.savefig('S_CURVE_TSNE')

    plt.show()

def print_n(n):
    if n>=1:
        print(1)
    else:
        print_n(n-1)
        print(n)

def exp_n_x(n, x):  # this function return the exp sum
    if n == 1:
        return x
    return exp_n_x(n - 1, x) * (x / n - 1) + exp_n_x(n - 1, x)
def print_binary_sequences(n):#this function print all the sequences of 0,1 in lenth n
    if n==0:
        return
    seq=['']*n
    print_binary_sequences_with_prefix(n,0,seq)



def print_binary_sequences_with_prefix(prefix,n,seq):# this function helps the print binary functtion
    if prefix>=n:
        print(''.join(seq))
        return
    seq[prefix]='0'
    print_binary_sequences_with_prefix(prefix+1,n,seq)
    seq[prefix]='1'
    print_binary_sequences_with_prefix(prefix+1,n,seq)






def print_sequences(char_list,n): # this function print the sequences that combined from the chars in the list in lenth n
    start = 0
    if n == 0:
        return ''
    seq_list = [''] * n
    print_sequences_helper(start, n, seq_list, char_list)



def print_sequences_helper(start,n,seq_list,char_list):#this function helps the print sequences function
    if start>=n:
        print(''.join(seq_list))
        return
    for char in char_list:
        seq_list[start]=char
        print_sequences_helper(start+1,n,seq_list,char_list)




def print_no_repition(char_list,n):
    start=0

    for i in range(len(char_list)):
        start=i
        #print_no_repition_helper(start,n,char_list)















if __name__ == '__main__':

    #Q1
    choose_k()

    #O2
    run_spectral_clustering_mnn()
    run_spectral_clustering_heat()

    run_kmeans()

    #Q3

   run_on_microarray()

    #Q5
    run_TSNE()
    print_n(5)
    AvgMatrix = pd.read_csv("C:/Users/Lior/Documents/MATLAB/Zscore.csv", sep=',', na_values=".")
    print("hhh")
