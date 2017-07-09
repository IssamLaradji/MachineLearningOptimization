import numpy as np
import scipy 
import pickle
import sys 

def mat2pkl(pkl_path, mat_path):
    x = scipy.io.loadmat(mat_path, squeeze_me=True)
    del x['__header__']
    del x['__version__']
    del x['__globals__']

    with open(pkl_path, 'wb') as f:
        pickle.dump(x, f, protocol=2)

    f.close()

    print("%s saved..." % pkl_path)

def save_pkl(fname, data):
    """Reads a pkl file.

    Parameters
    ----------
    fname : the name of the .pkl file

    Returns
    -------
    data :
        Returns the .pkl file as a 'dict'
    """
    fname = fname.replace(".pickle", "")
    fname += ".pickle"

    with open(fname, 'wb') as fl:
        pickle.dump(data, fl, protocol=2)

    print "%s saved..." % fname

    return data

def load_pkl(fname):
    """Reads a pkl file.

    Parameters
    ----------
    fname : the name of the .pkl file

    Returns
    -------
    data :
        Returns the .pkl file as a 'dict'
    """
    if sys.version_info[0] < 3:
        # Python 2
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    else:
        # Python 3
        with open(fname, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

    return data


def to_categorical(y):
    n_values = np.max(y) + 1
    
    return np.eye(n_values)[y]



def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma, mu, sigma
    
def check_gradient(f_func, g_func, nWeights, nCells=5):
    # This checks that the gradient implementation is correct
    w = np.random.rand(nWeights)

    # Check the gradient
    estimated_gradient = approx_fprime(w,
                                       f_func, 
                                       epsilon=1e-8,
                                       nCells=nCells)[:nCells]

    implemented_gradient = g_func(w)[:nCells]
    
    if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
        raise Exception('User and numerical derivatives differ:\n%s\n%s' % 
             (estimated_gradient[:5], implemented_gradient[:5]))
    else:
        print('User and numerical derivatives agree.')

def approx_fprime(x, f_func, epsilon=1e-7, nCells=5):
  # Approximate the gradient using the complex step method
  n_params = x.size
  e = np.zeros(n_params)
  gA = np.zeros(n_params)
  for n in range(nCells):

    e[n] = epsilon

    gA[n] = (f_func(x + e) - f_func(x - e)) / (2*epsilon)

    e[n] = 0

  return gA


def approx_fprime_complex(x, f_func, epsilon=1e-7, nCells=5):
  # Approximate the gradient using the complex step method
  n_params = x.size
  e = np.zeros(n_params)
  gA = np.zeros(n_params)
  for n in range(nCells):

    e[n] = 1.

    val = f_func(x + e * np.complex(0, epsilon))

    gA[n] = np.imag(val) / epsilon
    e[n] = 0

  return gA