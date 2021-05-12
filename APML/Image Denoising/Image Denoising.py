import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.stats
from scipy.stats import multivariate_normal
#from skimage.util import view_as_windows as viewW


def images_example(path='train_images.pickle'):
    """
    A function demonstrating how to access to image data supplied in this exercise.
    :param path: The path to the pickle file.
    """
    patch_size = (8, 8)

    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=patch_size, n=20000)

    plt.figure()
    plt.imshow(train_pictures[0])
    plt.title("Picture Example")

    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(patches[:, i].reshape(patch_size), cmap='gray')
        plt.title("Patch Example")
    plt.show()


def im2col(A, window, stepsize=1):
    """
    an im2col function, transferring an image to patches of size window (length
    2 list). the step size is the stride of the sliding window.
    :param A: The original image (NxM size matrix of pixel values).
    :param window: Length 2 list of 2D window size.
    :param stepsize: The step size for choosing patches (default is 1).
    :return: A (heightXwidth)x(NxM) matrix of image patches.
    """
    return viewW(np.ascontiguousarray(A), (window[0], window[1])).reshape(-1,
                                                                          window[0] * window[1]).T[:, ::stepsize]


def greyscale_and_standardize(images, remove_mean=True):
    """
    The function receives a list of RGB images and returns the images after
    grayscale, centering (mean 0) and scaling (between -0.5 and 0.5).
    :param images: A list of images before standardisation.
    :param remove_mean: Whether or not to remove the mean (default is True).
    :return: A list of images after standardisation.
    """
    standard_images = []

    for image in images:
        standard_images.append((0.299 * image[:, :, 0] +
                                0.587 * image[:, :, 1] +
                                0.114 * image[:, :, 2]) / 255)

    sum = 0
    pixels = 0
    for image in standard_images:
        sum += np.sum(image)
        pixels += image.shape[0] * image.shape[1]
    dataset_mean_pixel = float(sum) / pixels

    if remove_mean:
        for image in standard_images:
            image -= np.matlib.repmat([dataset_mean_pixel], image.shape[0],
                                      image.shape[1])

    return standard_images


def sample_patches(images, psize=(8, 8), n=10000, remove_mean=True):
    """
    sample N p-sized patches from images after standardising them.

    :param images: a list of pictures (not standardised).
    :param psize: a tuple containing the size of the patches (default is 8x8).
    :param n: number of patches (default is 10000).
    :param remove_mean: whether the mean should be removed (default is True).
    :return: A matrix of n patches from the given images.
    """
    d = psize[0] * psize[1]
    patches = np.zeros((d, n))
    standardized = greyscale_and_standardize(images, remove_mean)

    shapes = []
    for pic in standardized:
        shapes.append(pic.shape)

    rand_pic_num = np.random.randint(0, len(standardized), n)
    rand_x = np.random.rand(n)
    rand_y = np.random.rand(n)

    for i in range(n):
        pic_id = rand_pic_num[i]
        pic_shape = shapes[pic_id]
        x = int(np.ceil(rand_x[i] * (pic_shape[0] - psize[1])))
        y = int(np.ceil(rand_y[i] * (pic_shape[1] - psize[0])))
        patches[:, i] = np.reshape(np.ascontiguousarray(
            standardized[pic_id][x:x + psize[0], y:y + psize[1]]), d)

    return patches


def denoise_image(Y, model, denoise_function, noise_std, patch_size=(8, 8)):
    """
    A function for denoising an image. The function accepts a noisy gray scale
    image, denoises the different patches of it and then reconstructs the image.

    :param Y: the noisy image.
    :param model: a Model object (MVN/ICA/GSM).
    :param denoise_function: a pointer to one of the denoising functions (that corresponds to the model).
    :param noise_std: the noise standard deviation parameter.
    :param patch_size: the size of the patch that the model was trained on (default is 8x8).
    :return: the denoised image, after each patch was denoised. Note, the denoised image is a bit
    smaller than the original one, since we lose the edges when we look at all of the patches
    (this happens during the im2col function).
    """
    (h, w) = np.shape(Y)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))

    # split the image into columns and denoise the columns:
    noisy_patches = im2col(Y, patch_size)
    denoised_patches = denoise_function(noisy_patches, model, noise_std)

    print([cropped_h, cropped_w])

    # reshape the denoised columns into a picture:
    x_hat = np.reshape(denoised_patches[middle_linear_index, :],
                       [cropped_h, cropped_w])

    return x_hat


def crop_image(X, patch_size=(8, 8)):
    """
    crop the original image to fit the size of the denoised image.
    :param X: The original picture.
    :param patch_size: The patch size used in the model, to know how much we need to crop.
    :return: The cropped image.
    """
    (h, w) = np.shape(X)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))
    columns = im2col(X, patch_size)
    return np.reshape(columns[middle_linear_index, :], [cropped_h, cropped_w])


def normalize_log_likelihoods(X):
    """
    Given a matrix in log space, return the matrix with normalized columns in
    log space.
    :param X: Matrix in log space to be normalised.
    :return: The matrix after normalization.
    """
    h, w = np.shape(X)
    return X - np.matlib.repmat(logsumexp(X, axis=0), h, 1)


def test_denoising(image, model, denoise_function,
                   noise_range=(0.01, 0.05, 0.1, 0.2), patch_size=(8, 8)):
    """
    A simple function for testing your denoising code. You can and should
    implement additional tests for your code.
    :param image: An image matrix.
    :param model: A trained model (MVN/ICA/GSM).
    :param denoise_function: The denoise function that corresponds to your model.
    :param noise_range: A tuple containing different noise parameters you wish
            to test your code on. default is (0.01, 0.05, 0.1, 0.2).
    :param patch_size: The size of the patches you've used in your model.
            Default is (8, 8).
    """
    h, w = np.shape(image)
    noisy_images = np.zeros((h, w, len(noise_range)))
    denoised_images = []
    cropped_original = crop_image(image, patch_size)

    # make the image noisy:
    for i in range(len(noise_range)):
        noisy_images[:, :, i] = image + (
            noise_range[i] * np.random.randn(h, w))

    # denoise the image:
    for i in range(len(noise_range)):
        denoised_images.append(
            denoise_image(noisy_images[:, :, i], model, denoise_function,
                          noise_range[i], patch_size))

    # calculate the MSE for each noise range:
    for i in range(len(noise_range)):
        print("noisy MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((crop_image(noisy_images[:, :, i],
                                  patch_size) - cropped_original) ** 2))
        print("denoised MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((cropped_original - denoised_images[i]) ** 2))

    plt.figure()
    for i in range(len(noise_range)):
        plt.subplot(2, len(noise_range), i + 1)
        plt.imshow(noisy_images[:, :, i], cmap='gray')
        plt.subplot(2, len(noise_range), i + 1 + len(noise_range))
        plt.imshow(denoised_images[i], cmap='gray')
    plt.show()


class GMM_Model:
    """
    A class that represents a Gaussian Mixture Model, with all the parameters
    needed to specify the model.

    mixture - a length k vector with the multinomial parameters for the gaussians.
    means - a k-by-D matrix with the k different mean vectors of the gaussians.
    cov - a k-by-D-by-D tensor with the k different covariance matrices.
    """

    def __init__(self, mix, means, cov):
        self.mix = mix
        self.means = means
        self.cov = cov


class MVN_Model:
    """
    A class that represents a Multivariate Gaussian Model, with all the parameters
    needed to specify the model.

    mean - a D sized vector with the mean of the gaussian.
    cov - a D-by-D matrix with the covariance matrix.
    gmm - a GMM_Model object.
    """

    def __init__(self, means, cov, gmm):
        self.means = means
        self.cov = cov
        self.gmm = gmm


class GSM_Model:
    """
    A class that represents a GSM Model, with all the parameters needed to specify
    the model.

    cov - a k-by-D-by-D tensor with the k different covariance matrices. the
        covariance matrices should be scaled versions of each other.
    mix - k-length probability vector for the mixture of the gaussians.
    gmm - a GMM_Model object.
    """

    def __init__(self, cov, mix, gmm):
        self.cov = cov
        self.mix = mix
        self.gmm = gmm


class ICA_Model:
    """
    A class that represents an ICA Model, with all the parameters needed to specify
    the model.

    P - linear transformation of the sources. (X = P*S)
    vars - DxK matrix whose (d,k) element corresponds to the variance of the k'th
        gaussian in the d'th source.
    mix - DxK matrix whose (d,k) element corresponds to the weight of the k'th
        gaussian in d'th source.
    gmms - A list of D GMM_Models, one for each source.
    """

    def __init__(self, P, vars, mix, means, gmms):
        self.P = P
        self.vars = vars
        self.means = means
        self.mix = mix
        self.gmms = gmms


def MVN_log_likelihood(X, model):
    """
    Given image patches and a MVN model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A MVN_Model object.
    :return: The log likelihood of all the patches combined.
    """

    LL = np.sum(multivariate_normal.logpdf(X.T, model.means.T, model.cov, allow_singular=True))

    return LL


def learn_GMM(X, k, initial_model, learn_mixture=True, learn_means=True,
              learn_covariances=True, learn_r=False, iterations=10):
    """
    A general function for learning a GMM_Model using the EM algorithm.

    :param X: a DxN data matrix, where D is the dimension, and N is the number of samples.
    :param k: number of components in the mixture.
    :param initial_model: an initial GMM_Model object to initialize EM with.
    :param learn_mixture: a boolean for whether or not to learn the mixtures.
    :param learn_means: a boolean for whether or not to learn the means.
    :param learn_covariances: a boolean for whether or not to learn the covariances.
    :param iterations: Number of EM iterations (default is 10).
    :return: (GMM_Model, log_likelihood_history)
            GMM_Model - The learned GMM Model.
            log_likelihood_history - The log-likelihood history for debugging.
    """
    GMM_initial = init_GMM(X, k)
    phi_i, mu_y, cov, r_, LL = Gen_EM(X, k, initial_model, learn_mixture=True, learn_means=True,
                                      learn_covariances=True, learn_r=False, iterations=10)
    return GMM_Model(phi_i, mu_y, cov), LL


def Gen_EM(X, k, initial_model, learn_mixture=True, learn_means=True,
           learn_covariances=True, learn_r=False, iterations=10):
    """
        A general function for learning using the EM algorithm.

        :param X: a DxN data matrix, where D is the dimension, and N is the number of samples.
        :param k: number of components in the mixture.
        :param initial_model: an initial GMM/ICA/GSM object to initialize EM with.
        :param model_type : MVN,GSM and ICA
        :param learn_mixture: a boolean for whether or not to learn the mixtures.
        :param learn_means: a boolean for whether or not to learn the means.
        :param learn_covariances: a boolean for whether or not to learn the covariances.
        :param iterations: Number of EM iterations (default is 10).
        :return: (Model, log_likelihood_history)
                Model - The learned Model.
                log_likelihood_history - The log-likelihood history for debugging.
        """

    d, N = X.shape
    C_i_y = np.zeros((N, k))
    phi_i = np.array(initial_model.mix.T[0])
    mu_y = initial_model.means

    sigma_y = initial_model.cov
    X_i_centered = np.zeros((d, N))

    LL_tmp = np.zeros((iterations, 1))
    LL = []
    R_i = np.ones((k, 1))
    if (learn_r):
        R_i = [i / k for i in range(k)]
    # Try to converge with iterations
    for iter in range(iterations):

        # calc C_Y

        for i in range(k):
            C_i_y[:, i] = np.log(phi_i[i]) + multivariate_normal.logpdf(X.T, initial_model.means[i, None],
                                                                        initial_model.cov[i], allow_singular=True)

        C_i_y = np.exp(normalize_log_likelihoods(C_i_y))
        C_i_yT = C_i_y.T

        # Calc Phi
        if (learn_mixture):
            phi_i = (1 / N) * np.sum(C_i_y, axis=0)

        # calc Mu
        if (learn_means):
            for i in range(k):
                mu_y[i] = C_i_yT[i].dot(X.T) / np.sum(C_i_y[:, i])  # multypling 1XN with NXD

        if (learn_r):

            for i in range(k):
                R_i[i] = np.sum(C_i_y[:, i] * np.diag(X.T.dot(np.linalg.pinv(initial_model.cov[i])).dot(X))) / (
                d * np.sum(C_i_y[:, i]))
                sigma_y[i] = initial_model.cov[i] * R_i[i]

        # calc Cov
        for i in range(k):
            if (learn_covariances):
                X_centered = np.subtract(X, mu_y[i].T)  # DX1
                sigma_y[i] = (C_i_y[:, i] * X_i_centered).dot(X_i_centered.T) / np.sum(C_i_y[:, i]) * R_i[i]

        # calc LL
        for i in range(k):
            tmp = multivariate_normal.pdf(X.T, mu_y[i], sigma_y[i], allow_singular=True)
            LL_tmp[i] = np.sum(C_i_y[:, i] * np.log(phi_i[i] * tmp))

        LL.append(1 + np.sum(np.sum(LL_tmp, axis=1), axis=0))
    return phi_i, mu_y, sigma_y, R_i, LL


def initialize_means(X, K):
    '''
    initialization of k gaussians , used by GMM
    :param X: the data
    :param K: the number of gaussians
    :return: k gaussians
    '''
    C = [X[0]]
    for k in range(1, K):
        D2 = scipy.array([min([scipy.inner(c - x, c - x) for c in C]) for x in X])
        probs = D2 / D2.sum()
        cumprobs = probs.cumsum()
        r = scipy.rand()
        for j, p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        C.append(X[i])
    return C


def init_GMM(X, k):
    """
    Initialze a new GMM object
    :return: GMM object
    """

    d, N = X.shape
    means = np.random.rand(k, d)
    cov = np.ones((k, d, d))
    phi = np.asarray(np.matlib.repmat(1 / k, k, 1))

    return GMM_Model(phi, means, cov)


def init_MVN(X, d, N):
    '''
    This function initialze the MNV object
    using MLE
    :param X: The imgae set
    :return: MVN object
    '''

    mean = X.sum(axis=1) / d

    X_centred = np.zeros((d, N))
    for i in range(N):
        X_centred[:, i] = np.subtract(X[:, i], mean.T) / d

    cov = X_centred.dot(X_centred.T)
    return MVN_Model(mean, cov, GMM_Model(0, 0, 0)), X_centred


def learn_MVN(X):
    """
    Learn a multivariate normal model, given a matrix of image patches.
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :return: A trained MVN_Model object.
    """
    [d, N] = X.shape
    MVN_object, x_centered = init_MVN(X, d, N)

    # C and Phi in this case are 1 so we dont really need them
    mu = X.sum(axis=1) / N
    cov = (x_centered.dot(x_centered.T)) / N

    print('MVN log likelihood', MVN_log_likelihood(X, MVN_Model(mu, cov, GMM_Model(1, mu, cov))))

    return MVN_Model(mu, cov, GMM_Model(1, mu, cov))


def init_GSM(X, k):
    """
        A class that represents a GSM Model, with all the parameters needed to specify
        the model.

        cov - a k-by-D-by-D tensor with the k different covariance matrices. the
            covariance matrices should be scaled versions of each other.
        mix - k-length probability vector for the mixture of the gaussians.
        gmm - a GMM_Model object.
        """
    # random R^2
    cov = [X.dot(X.T) for i in range(k)]
    mix = np.matlib.repmat(1 / k, 1, k)

    return GSM_Model(cov, mix, GMM_Model(mix, np.zeros((cov[0].shape)), cov))


def learn_GSM(X, k):
    """
      Learn parameters for a Gaussian Scaling Mixture model for X using EM.

      GSM components share the variance, up to a scaling factor, so we only
      need to learn scaling factors and mixture proportions.

      :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
      :param k: The number of components of the GSM model.
      :return: A trained GSM_Model object.
      """
    GSM_initial = init_GSM(X, k)
    # send the parametes to gen _em to calculte the params
    phi_i, mu_y, cov, R_2, LL = Gen_EM(X, k, GSM_initial.gmm, learn_mixture=True, learn_means=False,
                                       learn_covariances=False, learn_r=True, iterations=10)

    return GSM_Model(cov, phi_i, GMM_Model(phi_i, mu_y, cov)), LL


def init_ICA(X, k):
    d, N = X.shape
    vars = []
    mixs = []
    gmms = []
    means = []
    P = np.linalg.eig(X.dot(X.T))[1]
    for i in range(d):
        GMM_Model = init_GMM(X[i, None], k)
        gmms.append(GMM_Model)
        vars.append(gmms[i].cov)
        mixs.append(gmms[i].mix)
        means.append(gmms[i].means)

    # s = ICA_Model.P.T.dot(X)  # each column is S_i
    return ICA_Model(np.array(P), vars, mixs, means, gmms)


def learn_ICA(X, k):
    """
    Learn parameters for a complete invertible ICA model.

    We learn a matrix P such that X = P*S, where S are D independent sources
    And for each of the D coordinates we learn a mixture of K univariate
    0-mean gaussians using EM.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components in the source gaussian mixtures.
    :return: A trained ICA_Model object.
    """
    d, N = X.shape
    ICA_model = init_ICA(X, k)
    trained_gmms = []
    trained_ICAS = []
    vars = []
    mixs = []
    means = []
    LL = np.zeros((d, 10))
    P_trans_y = ICA_model.P.T.dot(X)
    for i in range(d):
        phi_i, mu_y, cov, R_2, LL[i, :] = Gen_EM(P_trans_y[i, None], k, ICA_model.gmms[i], learn_mixture=True,
                                                 learn_means=True,
                                                 learn_covariances=True, learn_r=False, iterations=10)

        # _, P_add = np.linalg.eig(cov)
        trained_gmms.append(GMM_Model(phi_i, mu_y, cov))
        mixs.append(phi_i)
        vars.append(cov)
        means.append(mu_y)
    LL = np.sum(LL, axis=0)

    return ICA_Model(np.array(ICA_model.P), vars, mixs, means, trained_gmms), LL


def MVN_Denoise(Y, mvn_model, noise_std):
    """
    Denoise every column in Y, assuming an MVN model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a single
    0-mean multi-variate normal distribution.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param mvn_model: The MVN_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """

    return calc_weiner(Y, mvn_model.cov, mvn_model.means, noise_std)


def calc_weiner(Y, cov, means, noise_std):
    '''
    Calculates the weiner formula
    :param Y: the noised image
    :param cov: covarience matrix
    :param means: means matrix
    :param noise_std: the noise the image was noised with
    :return: x star - the clean image
    '''
    d = Y.shape[0]
    cov_inv = np.linalg.pinv(cov)
    sigma_noise = 1 / np.power(noise_std, 2)
    a = cov_inv + (sigma_noise) * np.identity(d)
    b = np.dot(cov_inv, means)[:, np.newaxis] + ((sigma_noise) * Y)
    final = np.dot(np.linalg.pinv(a), b)
    print(final.shape)
    return final


def GSM_Denoise(Y, gsm_model, noise_std):
    """
    Denoise every column in Y, assuming a GSM model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a mixture of
    0-mean gaussian components sharing the same covariance up to a scaling factor.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param gsm_model: The GSM_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.

    """
    d, N = Y.shape
    print(d, N)
    k = len(gsm_model.mix)
    opt_x = []
    C_i_y = np.zeros((N, k))
    var = np.identity(d) * (noise_std ** 2)

    for i in range(k):
        C_i_y[:, i] = np.log(gsm_model.mix[i]) + multivariate_normal.logpdf(Y.T, gsm_model.gmm.means[i][:],
                                                                            gsm_model.cov[i] + var, allow_singular=True)

    C_i_y = np.exp(normalize_log_likelihoods(C_i_y))
    for i in range(k):
        print((C_i_y[:, i] * calc_weiner(Y, gsm_model.cov[i], gsm_model.gmm.means[i][:], noise_std)).shape)
        opt_x.append(C_i_y[:, i] * calc_weiner(Y, gsm_model.cov[i], gsm_model.gmm.means[i][:], noise_std))

    opt_x = np.sum(opt_x, axis=0)

    print(np.array(opt_x).shape)
    return np.array(opt_x)


def ICA_Denoise(Y, ica_model, noise_std):
    """
    Denoise every column in Y, assuming an ICA model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by an ICA 0-mean
    mixture model.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param ica_model: The ICA_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    d, N = Y.shape
    P_T_Y = ica_model.P.T.dot(Y)
    opt_x = []
    tmp = []
    for i in range(d):
        for j in range(k):
            tmp.append(calc_weiner(P_T_Y[i, None], ica_model.gmms[i].cov[j], ica_model.gmms[i].means[j], noise_std))

        opt_x.append(np.sum(tmp, axis=0))
        tmp = []
    opt_x = np.array(np.sum(opt_x, axis=1))

    return ica_model.P.dot(opt_x)


if __name__ == '__main__':
    number_of_images = 200;
    k = 3;
    iterations = [i for i in range(1,11)]

    # load pictures snd normalize them
    with open('train_images.pickle', 'rb') as f1:
        train_pictures = pickle.load(f1)
    with open('test_images.pickle', 'rb') as f2:
        test_pictures = pickle.load(f2)

    # X = greyscale_and_standardize(train_pictures)
    X_train = sample_patches(train_pictures, (8, 8))
    X_test = greyscale_and_standardize(test_pictures, (8, 8))

    # learn with MVN,GSM and ICA models

    #MNV
    MVN_obj = learn_MVN(X_train)
    test_denoising(X_test[:][0], MVN_obj, MVN_Denoise)

    #GSM
    GSM_obj, LL = learn_GSM(X_train, k)
    plt.plot(iterations, LL, 'ro')
    plt.show()
    test_denoising(X_test[:][9], GSM_obj, GSM_Denoise)

    # ICA
    ICA_obj, LL = learn_ICA(X_train, k)
    plt.plot(iterations, LL, 'ro')
    plt.show()
    test_denoising(X_test[:][9], ICA_obj, ICA_Denoise)


