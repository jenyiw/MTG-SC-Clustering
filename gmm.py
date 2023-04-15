import numpy as np
from matplotlib import pyplot as plt
import os

rng = np.random.default_rng(133221333123111)

def CalculateMultiVarDensity(x, mean, cov, reg_covar: float = 1e-6):
    """
    Calculates the multivariate normal for one sample.
    Input:
        x - An (d,) numpy array
        mean - An (d,) numpy array; the mean vector
        cov - a (d,d) numpy arry; the covariance matrix
    Output:
        prob - a scaler
    """
    centered_x = x - mean
    cov = cov + np.eye(cov.shape[0]) * reg_covar
    p = np.power(np.linalg.det(2 * np.pi * cov), -1 / 2) * np.exp(
        -1 / 2 * centered_x.T @ np.linalg.pinv(cov) @ centered_x
    )
    return p


def MultiVarNormal(x, mean, cov, reg_covar: float = 1e-6):
    """
    MultiVarNormal implements the PDF for a mulitvariate gaussian distribution
    (You can do one sample at a time of all at once)
    Input:
        x - An (d) numpy array
            - Alternatively (n,d)
        mean - An (d,) numpy array; the mean vector
        cov - a (d,d) numpy arry; the covariance matrix
        reg_covar - regularization for covariance to ensure invertibility.
    Output:
        prob - a scaler
            - Alternatively (n,)

    Hints:
        - Use np.linalg.pinv to invert a matrix
        - if you have a (1,1) you can extrect the scalar with .item(0) on the array
            - this will likely only apply if you compute for one example at a time
    """
    probabilities = np.apply_along_axis(CalculateMultiVarDensity, 1, x, mean, cov, reg_covar)
    return probabilities


def UpdateMixProps(hidden_matrix):
    """
    Returns the new mixing proportions given a hidden matrix
    Input:
        hidden_matrix - A (n, k) numpy array
    Output:
        mix_props - A (k,) numpy array
    Hint:
        - See equation in Lecture 10 pg 42
    """
    updated_mix_props = np.sum(hidden_matrix, axis=0) / hidden_matrix.shape[0]
    return updated_mix_props


def UpdateMeans(X, hidden_matrix):
    """
    Returns the new means for the gaussian distributions given the data and the hidden matrix
    Input:
        X - A (n, d) numpy arrak
        hidden_matrix - A (n, k) numpy array
    Output:
        new_means - A (k,d) numpy array
    Hint:
        - See equation in Lecture 10 pg 43
    """
    updated_mean = hidden_matrix.T @ X / np.sum(hidden_matrix, axis=0)[:, np.newaxis]
    return updated_mean


def UpdateCovar(X, hidden_matrix_col, mean):
    """
    Returns new covariance for a single gaussian distribution given the data, hidden matrix, and distribution mean
    Input:
        X - A (n, d) numpy arrak
        hidden_matrix_col - A (n,) numpy array
        mean - A (d,) numpy array; the mean for this distribution
    Output:
        new_cov - A (d,d) numpy array
    Hint:
        - See equation in Lecture 10 pg 43
    """
    centered_X = X - mean
    cov = centered_X.T @ np.diag(hidden_matrix_col) @ centered_X / np.sum(hidden_matrix_col)
    return cov


def UpdateCovars(X, hidden_matrix, means):
    """
    Returns a new covariance matrix for all distributions using the function UpdateCovar()
    Input:
        X - A (n, d) numpy arrak
        hidden_matrix - A (n, k) numpy array
        means - A (k,d) numpy array; All means for the distributions
    Output:
        new_covs - A (k,d,d) numpy array
    Hint:
        - Use UpdateCovar() function
    """
    k = hidden_matrix.shape[1]
    d = X.shape[1]
    cov = np.zeros((k, d, d))
    for i in range(k):
        cov[i, :, :] = UpdateCovar(X, hidden_matrix[:, i], means[i, :])

    return cov


def HiddenMatrix(X, means, covs, mix_props, reg_covar: float = 1e-6):
    """
    Computes the hidden matrix for the data. This function should also compute the log likelihood
    Input:
        X - An (n,d) numpy array
        means - An (k,d) numpy array; the mean vector
        covs - a (k,d,d) numpy arry; the covariance matrix
        mix_props - a (k,) array; the mixing proportions
    Output:
        hidden_matrix - a (n,k) numpy array
        ll - a scalar; the log likelihood
    Hints:
        - Construct an intermediate matrix of size (n,k). This matrix can be used to calculate the loglikelihood and the hidden matrix
            - Element t_{i,j}, where i in {1,...,n} and j in {1,k}, should equal
            P(X_i | c = j)P(c = j)
        - Each rows of the hidden matrix should sum to 1
            - Element h_{i,j}, where i in {1,...,n} and j in {1,k}, should equal
                P(X_i | c = j)P(c = j) / (Sum_{l=1}^{k}(P(X_i | c = l)P(c = l)))
    """
    n = X.shape[0]
    k = means.shape[0]
    hm = np.zeros((n, k))
    for i in range(k):
        hm[:, i] = mix_props[i] * MultiVarNormal(X, means[i, :], covs[i, :, :], reg_covar)

    ll = np.sum(np.log(np.sum(hm, axis=1)))
    hm = hm / np.sum(hm, axis=1)[:, np.newaxis]
    return hm, ll


def GMM(X, init_means, init_covs, init_mix_props, thresh=0.001, reg_covar: float = 1e-6, max_iterations: int = 1000):
    """
    Runs the GMM algorithm
    Input:
        X - An (n,d) numpy array
        init_means - a (k,d) numpy array; the initial means
        init_covs - a (k,d,d) numpy arry; the initial covariance matrices
        init_mix_props - a (k,) array; the initial mixing proportions
    Output:
        - clusters: a (n,) numpy array; the cluster assignment for each sample
        - ll: th elog likelihood at the stopping condition
        - hm: The hidden matrix (probability that each sample is in each cluster)
    Hints:
        - Use all the above functions
        - Stoping condition should be when the difference between your ll from
            the current iteration and the last iteration is below your threshold
    """
    means = init_means
    covs = init_covs
    mix_props = init_mix_props

    i = 0
    loss = np.zeros(max_iterations)
    while i < max_iterations:
        hidden_matrix, loss[i] = HiddenMatrix(X, means, covs, mix_props, reg_covar)
        if i > 0 and loss[i] - loss[i - 1] < thresh:
            break
        mix_props = UpdateMixProps(hidden_matrix)
        covs = UpdateCovars(X, hidden_matrix, means)
        means = UpdateMeans(X, hidden_matrix)
        print(i, loss[i])
        i += 1
    return np.argmax(hidden_matrix, axis=1), loss[:i], hidden_matrix


def CustomPlot(y, x=None, title=None, xlabel=None, ylabel=None, save_path=None):
    if title is None:
        title = ""
    if xlabel is None:
        xlabel = ""
    if ylabel is None:
        ylabel = ""
    if save_path is None:
        save_path = "loss_plot.png"

    if x is None:
        x = np.asarray([i for i in range(len(y))])

    fig = plt.figure(0)
    ax = fig.subplots()
    ax.plot(x, y, marker=".", linestyle="-")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(save_path)
    fig.clear()


def RandomParams(gmm_data, k, n_features, epsilon=0.005, eye_covar=False):
    means = gmm_data[rng.choice(range(gmm_data.shape[0]), k, replace=False), :]
    if not eye_covar:
        covars = []
        for _ in range(k):
            covar = (np.eye(n_features, n_features) * rng.random()) + rng.normal(size=(n_features, n_features))
            covars += [covar.T.dot(covar)]
    else:
        covars = np.stack([np.diag(([1] * 10))] * k)

    mix_props = rng.random(size=(k))
    mix_props = mix_props / np.sum(mix_props)

    return means, np.stack([x + np.eye(n_features, n_features) * epsilon for x in covars]), mix_props


def Question6A(data, test_means):
    k = 3
    d = data.shape[1]
    init_cov = np.zeros((k, d, d))
    for i in range(k):
        init_cov[i, :, :] = np.diag(np.ones(data.shape[1]))
    init_mix_props = np.asarray([0.3, 0.3, 0.4])

    _, loss, hm = GMM(data, test_means, init_cov, init_mix_props, reg_covar=0)
    # np.savetxt("6a.txt", hm[0, :])

    print(loss[-1])
    # CustomPlot(
    #     loss,
    #     xlabel="Iteration Number",
    #     ylabel="Log Likelihood",
    #     title="Log Likelihood vs Iteration Number for 3 clusters",
    #     save_path="6a.png",
    # )


def Question6C(data):
    min_k = 2
    max_k = 15
    best_losses = -np.inf * np.ones(max_k-min_k+1)
    d = data.shape[1]
    for k in range(min_k, max_k+1):
        for i in range(1):
            print(f"\rNumber of Clusters: {k}\tIteration: {i}.", end = "")
            init_means, init_covs, init_mix_props = RandomParams(data, k, d)
            _, loss, _ = GMM(data, init_means, init_covs, init_mix_props)
            if loss[-1] > best_losses[k - 3]:
                best_losses[k - 3] = loss[-1]
    print("\nDone!")
    CustomPlot(
        best_losses,
        x=np.asarray(range(min_k, max_k+1)),
        xlabel="Number of Clusters",
        ylabel="Log Likelihood",
        title="Log Likelihood vs Number of clusters",
        save_path="TEST.png",
    )


if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    data_path = os.path.join(dir, "data", "mouse-data", "hip1000.txt")
    test_path = os.path.join(dir, "data", "test_mean.txt")

    data = np.loadtxt(data_path, dtype=np.float32, delimiter=",").T
    test_means = np.loadtxt(test_path).T
    data = data[:, :10]
    test_means = test_means[:, :10]
    print("Data shape:", data.shape)
    print("test_means shape: ", test_means.shape)

    Question6A(data, test_means)
    # Question6C(data)
