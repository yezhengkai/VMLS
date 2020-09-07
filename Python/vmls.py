import numpy as np

# ========== chapter 01 ==========
def linear_combination(coef, vectors):
    """Linear combination of vectors"""
    n = len(vectors[0])
    comb = np.zeros(n)
    for i in range(len(vectors)):
        comb = comb + coef[i] * vectors[i]
    return comb


def compact_linear_combination(coef, vectors):
    return sum(coef[i]*vectors[i] for i in range(len(vectors)))


# ========== chapter 03 ==========
def standardize(x):
    x_tilde = x - np.mean(x)
    return x_tilde/np.std(x_tilde)


def corr_coef(a, b):
    a_tilde = a - np.mean(a)
    b_tilde = b - np.mean(b)
    denom = (np.linalg.norm(a_tilde) * np.linalg.norm(b_tilde))
    return (a_tilde @ b_tilde) /denom


# ========== chapter 04 ==========
def kmeans1(X, num_clusters, max_iters=100, tolerance=1e-4, random_seed=None):
    num_samples = len(X)  # number of samples
    n = len(X[0])  # dimension of each vector
    distances = np.zeros(num_samples)  # stores distances of each point to nearest rep
    # point to the nearest representative.
    reps = [np.zeros(n)
            for j in range(num_clusters)]  # stores representatives
    progress = []  # used to visualize J updates
    
    # placeholder for ending condition: 
    # if previous J and current J haven't changed much, finish
    J_prev = np.Infinity
    
    # intializes a random assignment of num_samples integers between 0 and num_clusters
    random_state = np.random.RandomState(random_seed)
    assignment = [random_state.randint(num_clusters) for i in range(num_samples)]
    
    # start iteration
    for itr in range(max_iters):    
        # Cluster j representative is average of points in cluster j.
        for j in range(num_clusters):
            group = [i for i in range(num_samples) if assignment[i] == j]
            reps[j] = sum(X[group] / len(group))
        
        # For each X[i], find distance to the nearest representative 
        # and its group index.
        for i in range(num_samples):
            #(distance and index):
            (distances[i], assignment[i]) = (
                np.amin(
                    [np.linalg.norm(X[i] - reps[j]) for j in range(num_clusters)]
                ), 
                [np.linalg.norm(X[i] - reps[j]) for j in range(n_clusters)].index(
                    np.amin([np.linalg.norm(X[i] - reps[j]) for j in range(num_clusters)])
                )
            )
        J = (np.linalg.norm(distances)**2) / num_samples  # calculate objective
        
        progress.append([J, itr])
        print("Iteration " + str(itr) + ": Jclust = " + str(J) + ".")
        
        # check whether the stop condition is met
        if (itr > 1) and (abs(J - J_prev) < (tolerance * J)):
            return assignment, reps, progress
        J_prev = J
        
    return assignment, reps, progress


def kmeans2(X, num_clusters, max_iters=100, tolerance=1e-4, random_seed=None):
    
    num_samples = len(X)
    distances = np.zeros(num_samples)  # stores distances of each point to nearest rep
    assignment = np.zeros(num_samples)
    
    progress = []  # used to visualize J updates
    
    # placeholder for ending condition: 
    # if previous J and current J haven't changed much, finish
    J_prev = np.Infinity
    
    # initialize random centroids (i.e. representatives)
    random_state = np.random.RandomState(random_seed)
    seeds = random_state.permutation(num_samples)[:num_clusters]
    reps = X[seeds]
    
    # start iteration
    for itr in range(max_iters):
        
        # group assignment
        for i in range(num_samples):
            #(distance and index):
            (distances[i], assignment[i]) = (
                np.amin(
                    [np.linalg.norm(X[i] - reps[j]) for j in range(num_clusters)]
                ), 
                [np.linalg.norm(X[i] - reps[j]) for j in range(n_clusters)].index(
                    np.amin([np.linalg.norm(X[i] - reps[j]) for j in range(num_clusters)])
                )
            )
        J = (np.linalg.norm(distances)**2) / num_samples  # calculate objective
        
        # show info
        progress.append([J, itr])
        print("Iteration " + str(itr) + ": Jclust = " + str(J) + ".")
        
        # check whether the stop condition is met
        if (itr > 1) and (abs(J - J_prev) < (tolerance * J)):
            return assignment, reps, progress
        J_prev = J
        
        
        # Update centroid
        for j in range(num_clusters):
            group = [i for i in range(num_samples) if assignment[i] == j]
            reps[j] = sum(X[group] / len(group))
        
    return assignment, reps, progress


# ========== chapter 05 ==========
def gram_schmidt(a, tol=1e-10):
    q = []
    for i in range(len(a)):
        #orthogonalization
        q_tilde = a[i]
        for j in range(len(q)):
            q_tilde = q_tilde - (q[j] @ a[i])*q[j]
        #Test for dependennce
        if np.sqrt(sum(q_tilde**2)) <= tol:
            print('Vectors are linearly dependent.')
            print('GS algorithm terminates at iteration ', i+1)
            return q
        #Normalization
        else:
            q_tilde = q_tilde / np.sqrt(sum(q_tilde**2))
            q.append(q_tilde)
    print('Vectors are linearly independent.')
    return q


# ========== chapter 06 ==========
def running_sum(n):
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            S[i, j] = 1
    return S


def compact_running_sum(n):
    S = np.tril(np.ones((n, n)))
    return S


def vandermonde(t, n):
    m = len(t)
    V = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            V[i, j] = t[i]**(j)
    return V


# ========== chapter 07 ==========
def rotate2d_at(arr, theta, center=[0, 0]):
    center = np.array(center)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    new_arr = R @ (arr-center) + center
    return new_arr


# From: https://scipy-cookbook.readthedocs.io/items/RankNullspace.html
def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def toeplitz(b, n):
    m = len(b)
    T = np.zeros((n+m-1, n))
    for j in range(n):
        T[j:j+m, j] = b
    return T


# ========== chapter 10 ==========
def QR_factorization(A):
    Q_transpose = np.array(gram_schmidt(A.T))
    R = Q_transpose @ A
    Q = Q_transpose.T
    return Q, R


# ========== chapter 11 ==========
def back_subst(R, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - R[i, i+1:n] @ x[i+1:n]) / R[i, i]
    return x


def solve_via_backsub(A, b):
    Q,R = QR_factorization(A)
    b_tilde = Q.T @ b
    x = back_subst(R,b_tilde)
    return x


# ========== chapter 14 ==========
def confusion_matrix(y, yhat, K):
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            C[i,j] = sum((y == i+1) & (yhat == j+1))
    return C


def one_hot(ycl, K):
    N = len(ycl)
    Y = np.zeros((N, K))
    for j in range(K):
        Y[np.where(ycl == j), j] = 1
    return Y


def ls_multiclass(X, ycl, K):
    n, N = X.shape
    Theta = np.linalg.lstsq(X.T, 2*one_hot(ycl - 1, K) - 1, rcond=None)[0]
    yhat = 1 + row_argmax(X.T @ Theta)
    return Theta, yhat


# ========== chapter 15 ==========
def mols_solve(As, bs, lambdas):
    """Multi-objective least squares"""
    k = len(lambdas)
    Atil = np.vstack([np.sqrt(lambdas[i])*As[i] for i in range(k)])
    btil = np.hstack([np.sqrt(lambdas[i])*bs[i] for i in range(k)])
    return np.linalg.lstsq(Atil, btil, rcond=None)[0]