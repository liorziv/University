import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import math


def calcDistanceMat(X,n):
    '''
    calculets X distance matrix
    '''
    delta = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1,n):
            delta[i, j] = np.linalg.norm(X[i, :] - X[j, :])
            delta[j, i] = delta[i, j]
    return delta


def digits_example():
    '''
    Example code to show you how to load the MNIST data and plot it.
    '''

    # load the MNIST data:
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target

    # plot examples:

    plt.gray()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        plt.imshow(np.reshape(data[i, :], (8, 8)))
        plt.title("Digit " + str(labels[i]))
    plt.show()

def digits_display():
    '''
    Perform a dimension reduction on the MNIST
    data using MDS, LLE and Diffusion map algorithms.
    '''

    # load the MNIST data:
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target

    #create the plot
    fig = plt.figure()

    # MDS
    X_MDS = MDS(data, 2)
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.scatter(X_MDS[:, 0], X_MDS[:, 1], c=labels, cmap=plt.cm.Spectral)
    ax1.set_title("MDS data")

    # LLE
    X_LLE = LLE(data, 2, 7)
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.scatter(X_LLE[:, 0], X_LLE[:, 1], c=labels, cmap=plt.cm.Spectral)
    ax2.set_title("After LLE")

    # Diffusion Map
    X_diffMap = DiffusionMap(data, 2, 0.25, 10, 40)
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.scatter(X_diffMap[:, 0], X_diffMap[:, 1], c=labels, cmap=plt.cm.Spectral)
    ax3.set_title("After Diffusion Map")

    fig.tight_layout()
    fig.savefig('digits.png');
    plt.show()


def swiss_roll_example():
    '''
        Example code to show you how to load the swiss roll data and plot it.
        '''

    # load the dataset:
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)

    # plot the data:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()


def swiss_roll_display():
    '''
        Perform a dimension reduction on the swiss roll
        data using MDS, LLE and Diffusion map algorithms.
    '''

    # load the dataset:
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=1000)

    # plot the data:
    fig = plt.figure()
    fig.suptitle('Swiss Roll', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.set_title("Original data")

    X_mds = MDS(X, 2)
    ax1 = fig.add_subplot(2, 2, 2)
    ax1.scatter(X_mds[:, 0], X_mds[:, 1], c=color, cmap=plt.cm.Spectral)
    ax1.set_title("After MDS")

    X_LLE = LLE(X, 2, 10)
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.scatter(X_LLE[:, 0], X_LLE[:, 1], c=color, cmap=plt.cm.Spectral)
    ax2.set_title("After LLE")

    X_diffMap = DiffusionMap(X, 2, 0.6, 1 )
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.scatter(X_diffMap[:, 0], X_diffMap[:, 1], c=color, cmap=plt.cm.Spectral)
    ax3.set_title("After Diffusion Map")

    fig.tight_layout()
    fig.savefig('swiss_roll.png');
    plt.show()


def faces_example(path):
    '''
    Example code to show you how to load the faces data.
    '''

    with open(path, 'rb') as f:
        X = pickle.load(f)

    num_images, num_pixels = np.shape(X)
    d = int(num_pixels**0.5)
    print("The number of images in the data set is " + str(num_images))
    print("The image size is " + str(d) + " by " + str(d))

    # plot some examples of faces:
    plt.gray()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(np.reshape(X[i, :], (d, d)))
    plt.show()




def faces_display(path):
    '''
        Perform a dimension reduction on the faces images
        using MDS, LLE and Diffusion map algorithms.
    '''

    with open(path, 'rb') as f:
        X = pickle.load(f)

    X_MDS = MDS(X, 2)
    fig1 = plot_with_images(X_MDS, X, 'Faces using MDS', 20)
    fig1.savefig('faces_MDS.png');
    fig1.show()

    X_LLE = LLE(X, 2, 7)
    fig2 = plot_with_images(X_LLE, X, 'Faces using LLE', 20)
    fig2.savefig('faces_LLE.png');
    fig2.show()


    X_DiffMap = DiffusionMap(X, 2, 1000, 10, 10)
    fig3 = plot_with_images(X_DiffMap, X, 'Faces using Diffusion Maps',20)
    fig3.savefig('faces_DiffMap.png');
    fig3.show()


def plot_with_images(X, images, title, image_num=25):
    '''
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param num_to_plot: Number of images to plot along with the scatter plot.
    :return: the figure object.
    '''

    n, pixels = np.shape(images)
    img_size = int(pixels**0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # get the size of the embedded images for plotting:
    x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
    y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08

    # draw random images and plot them in their relevant place:
    for i in range(image_num):
        img_num = np.random.choice(n)
        x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
        x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
        img = images[img_num, :].reshape(img_size, img_size)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000,
                  extent=(x0, x1, y0, y1))

    # draw the scatter plot of the embedded data points:
    ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)

    return fig


def MDS(X, d):
    '''
    Given a NxN pairwise distance matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.

    :param X: NxN distance matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix.
    '''

    # calculate the distance matrix
    n = X.shape[0]
    delta = calcDistanceMat(X, n)
    for i in range(n):
        for j in range(i + 1, n):
            delta[i, j] = np.linalg.norm(X[i, :] - X[j, :])
            delta[j, i] = delta[i, j]

    # calculate H
    H = np.full((n, n), -(1 / n))
    np.fill_diagonal(H, 1 - (1 / n))

    # calculate S
    S = -0.5 * (H.dot(delta.dot(H)))

    # diagonalize s
    lamb, U = np.linalg.eig(S)
    idx = np.argsort(-lamb)
    lamb = lamb[idx]
    U = U[:, idx]

    # create return matrix
    retMat = np.zeros((n, d))
    for i in range(d):
        retMat[:, i] = U[:, i] * np.sqrt(lamb[i])

    return retMat;

def MDS_scree(X, d):
    '''
    Copy used by scree_plot
    Given a NxN pairwise distance matrix and the number of desired dimensions,
    return the eigen values compured as part of reducing data points dimension using MDS.

    :param X: NxN distance matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix.
    '''

    # calculate the distance matrix
    n = X.shape[0]
    delta = calcDistanceMat(X, n)
    for i in range(n):
        for j in range(i + 1, n):
            delta[i, j] = np.linalg.norm(X[i, :] - X[j, :])
            delta[j, i] = delta[i, j]

    # calculate H
    H = np.full((n, n), -(1 / n))
    np.fill_diagonal(H, 1 - (1 / n))

    # calculate S
    S = -0.5 * (H.dot(delta.dot(H)))

    # diagonalize s
    lamb, U = np.linalg.eig(S)
    idx = np.argsort(-lamb)
    lamb = lamb[idx]


    return lamb[:10]



def LLE(X, d, k):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the LLE algorithm.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param k: the number of neighbors for the weight extraction.
    :return: Nxd reduced data matrix.
    '''

    # Calculate distance matrix
    n = X.shape[0]
    delta = calcDistanceMat(X, n)

    # find k nearst neighboors
    W = np.zeros((n, n))
    for i in range(n):
        idx = np.argsort(delta[i, :])[1:k + 1]
        zi = X[idx, :] - X[i, :]
        G = zi.dot(zi.T)
        unitVec = np.ones((1, k))
        invG = np.linalg.pinv(G)
        lamb = 2 / (unitVec.dot(invG).dot(unitVec.T))[0]
        wi = (lamb / 2) * invG.dot(unitVec.T)
        W[i, idx] = wi.T

    # calculting M and its eigen values
    M = (np.identity(n) - W).T.dot((np.identity(n) - W))
    lamb, U = np.linalg.eig(M)
    idx = np.argsort(lamb)
    U = U[:, idx]

    return U[:, 1:d + 1]



def DiffusionMap(X, d, sigma, t, k = -1):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the Diffusion Map algorithm. The k parameter allows restricting the
    gram matrix to only the k nearest neighbor of each data point.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param sigma: the sigma of the gaussian for the gram matrix transformation.
    :param t: the scale of the diffusion (amount of time steps).
    :param k: the amount of neighbors to take into account when calculating the gram matrix.
    :return: Nxd reduced data matrix.
    '''

    # creates heat kernel
    n = X.shape[0]
    K = np.zeros((n,n))


    #case were k was given
    if(k != -1):
        delta = calcDistanceMat(X, n)
        for i in range(n):
            idx = np.argsort(delta[i, :])[1:k + 1]
            only_neigbhoors = delta[i,idx]
            K[i,idx] = np.exp(np.divide(-only_neigbhoors,sigma) )
    #case were we take all the points(k is not given)
    else:
        K = np.exp(np.divide(-calcDistanceMat(X, n), sigma))

    # normelize K = A
    Di = np.sum(K, axis=1)
    D = np.zeros((n, n))
    np.fill_diagonal(D, Di)
    A = np.linalg.inv(D).dot(K)

    # return the eigen vectors* lambda(i)^t
    lamb, U = np.linalg.eig(A)
    idx = np.argsort(-lamb)
    idx = idx[1:d + 1]
    lamb = lamb[idx]
    retMat = U[:, idx]
    for i in range(d):
        retMat[:, i] = (math.pow(lamb[i], t)) * retMat[:, i]

    return retMat

def Scree_plot():
    '''
    Plot the eigen values of 2D matrix embedded in 3D
    dimension and added noise in different rates
    '''

    #creating 2D data embedded in 3D
    number_of_samples = 1000
    dim = 3
    parb_1 = np.zeros((number_of_samples, dim))
    parb_2 = np.zeros((number_of_samples, dim))

    for i in range(number_of_samples):
        parb_1[i,0] = np.random.uniform()*10
        parb_1[i,1] =  parb_1[i,0]**2

    #creatung gaussian matrix
    rand_gauss_mat = np.random.normal(0,1,dim*dim)
    rand_gauss_mat.resize((dim, dim))

    #creating the rotation matrix using QR on the gaussian matrix
    [Q,R] = np.linalg.qr(rand_gauss_mat)
    res1 = parb_1.dot(Q)


    #create the noise matrix and plot the result
    eigen_mat = np.zeros((10,10))
    cnt = 0;
    for noise_rate in range(1,160,50):
        noise_mat = np.random.normal(0, 1, number_of_samples * dim)*noise_rate
        noise_mat.resize((number_of_samples,dim))
        tmp = res1 + noise_mat
        eigen_mat[cnt,:] = MDS_scree(tmp,2)
        cnt +=1

    mark_eigen = [i for i in range (10)]
    fig = plt.figure()


    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(eigen_mat[0, :])
    ax1.scatter(mark_eigen,eigen_mat[0, :])
    ax1.set_title('Noise Factor 1')

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(mark_eigen,eigen_mat[1,:])
    ax2.scatter(mark_eigen, eigen_mat[1, :])
    ax2.set_title('Noise Factor 51')

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.plot(mark_eigen,eigen_mat[2, :])
    ax3.scatter(mark_eigen, eigen_mat[2, :])
    ax3.set_title('Noise Factor 101')

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.plot(mark_eigen,eigen_mat[3, :])
    ax4.scatter(mark_eigen, eigen_mat[3, :])
    ax4.set_title('Noise Factor 151')

    fig.tight_layout()
    fig.savefig('Scree_Plot.png')
    fig.show()


def lossy_compression_distance():
    '''
    Takes original 100 dimension data and reduce it to 70,50,25 and 10
    dimension, for each dimension reduction calculate the distance matrix
    and compare it with the original distance matrix.
    '''
    reduce_dim_mat = np.random.normal(0, 1, 10000)
    reduce_dim_mat.resize((100,100))

    fig = plt.figure()

    delta_orig = calcDistanceMat(reduce_dim_mat,100)
    ax1 = fig.add_subplot(1, 4, 1)
    delta_orig.resize((1, 10000))
    ax1.scatter(delta_orig,delta_orig)
    ax1.set_title('Original Dimension')

    x_50 = MDS(reduce_dim_mat, 50)
    delta_50 = calcDistanceMat(x_50, 100)
    ax3 = fig.add_subplot(1, 4, 2)
    delta_50.resize((1, 100 * 100))
    ax3.scatter(delta_orig,delta_50)
    ax3.set_title('Reduced to 50')

    x_25 = MDS(reduce_dim_mat, 25)
    delta_25 = calcDistanceMat(x_25, 100)
    ax4 = fig.add_subplot(1, 4, 3)
    delta_25.resize((1, 100 * 100))
    ax4.scatter(delta_orig,delta_25)
    ax4.set_title('Reduced to 25')

    x_3 = MDS(reduce_dim_mat, 3)
    delta_3 = calcDistanceMat(x_3, 100)
    ax5 = fig.add_subplot(1, 4, 4)
    delta_3.resize((1, 100 * 100))
    ax5.scatter(delta_orig, delta_3)
    ax5.set_title('Reduced to 3')

    fig.tight_layout()
    fig.savefig('lossy_compression_distance.png')
    plt.show()


def LLE_neighbors():
    '''
        Perform a dimension reduction on the swiss roll
        data using LLE with different k values
    '''

    # load the dataset:
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=1000)

    # plot the data:
    fig = plt.figure()

    ax = fig.add_subplot(3, 1, 1, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.set_title("Original data")

    X_LLE_5 = LLE(X, 2, 5)
    ax1 = fig.add_subplot(3, 1, 2)
    ax1.scatter(X_LLE_5[:, 0], X_LLE_5[:, 1], c=color, cmap=plt.cm.Spectral)
    ax1.set_title(" k = 5")

    X_LLE_10 = LLE(X, 2, 10)
    ax2 = fig.add_subplot(3, 1, 3)
    ax2.scatter(X_LLE_10[:, 0], X_LLE_10[:, 1], c=color, cmap=plt.cm.Spectral)
    ax2.set_title(" k = 10")

   # fig.tight_layout()
   # fig.savefig('LLE_neighbors1.png');
    plt.show()

    # plot the data:
    fig2 = plt.figure()
    X_LLE_25 = LLE(X, 2, 25)
    ax3 = fig2.add_subplot(3, 1, 1)
    ax3.scatter(X_LLE_25[:, 0], X_LLE_25[:, 1], c=color, cmap=plt.cm.Spectral)
    ax3.set_title(" k = 25")

    X_LLE_35 = LLE(X, 2, 50)
    ax3 = fig2.add_subplot(3, 1, 2)
    ax3.scatter(X_LLE_35[:, 0], X_LLE_35[:, 1], c=color, cmap=plt.cm.Spectral)
    ax3.set_title(" k = 50")

    X_LLE_50 = LLE(X, 2, 100)
    ax3 = fig2.add_subplot(3, 1, 3)
    ax3.scatter(X_LLE_50[:, 0], X_LLE_50[:, 1], c=color, cmap=plt.cm.Spectral)
    ax3.set_title(" k = 100")

    fig2.tight_layout()
    fig2.savefig('LLE_neighbors2.png');
    plt.show()


if __name__ == '__main__':

    digits_display()
    swiss_roll_display()
    faces_display('faces.pickle')
    Scree_plot()
    lossy_compression_distance()
    LLE_neighbors()
