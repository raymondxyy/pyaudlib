# Time Analysis of Audio signals
# Author: Raymond Xia (yangyanx@andrew.cmu.edu)

# Change log:
#   09/05/17:
#       * Create this file
from scipy.linalg import toeplitz, solve_toeplitz, inv
from scipy.signal import fftconvolve
import numpy as np
import pdb


def lpc(x, order, method='autocorr', levinson=False, out='full', force_stable=True):
    """
    Implement linear predictive coding (LPC).
    Args:
        x        - time sequence (usually a windowed short-time sequence)
        order    - LPC order
        method   - one of autocorr,cov,parcor
        levinson - Use Levinson-Durbin recursion?
        out      - one of full,alpha
    """
    assert order < len(x)  # LPC order cannot exceed autocorrelation dimension
    eps = 1e-9
    if method == 'autocorr':  # implement autocorrelation method
        phi = xcorr(x)
        #phi[np.abs(phi)<eps] = 0 # avoid very small nonzero values
        if levinson:  # use levinson-durbin recursion to solve for coefficients
            try:
                alpha = solve_toeplitz(phi[:order], phi[1:order+1])
            except np.linalg.linalg.LinAlgError:
                print(
                    "WARNING: singular matrix - adding small value to phi[0]")
                print(phi[:order])
                phi[0] += 1e-9
                alpha = solve_toeplitz(phi[:order], phi[1:order+1])
        else:  # solve by inversion. slower but more precise.
            alpha = inv(toeplitz(phi[:order])).dot(phi[1:order+1])
        if force_stable and (not lpc_is_stable(alpha)):
            print("Unstable LPC detected. Reflecting back to unit circle.")
            #pdb.set_trace()
            alpha = lpc2stable(alpha)

    elif method == 'cov':  # TODO: implement cov and parcor
        pass
    elif method == 'parcor':
        pass
    else:
        raise ValueError("Method must be one of [autocorr,cov,parcor].")
    if out == 'full':
        return np.insert(-alpha, 0, 1)
    else:
        return alpha


def xcorr(x, y=None, one_side=True):
    """
    Calculate the cross-correlation between x and y defined as:
        \phi_xy[k] = \sum_m x[m]*y[m+k]
    Args:
        x - time sequence
        y - time sequence
    Returns:
        cross-correlation sequence
    """
    if y is None:  # auto-correlation mode
        if one_side:  # return only one side of the symmetric sequence
            return fftconvolve(x[::-1], x)[len(x)-1:]
        else:  # return the entire symmetric sequence
            return fftconvolve(x[::-1], x)
    else:  # cross-correlation mode
        return fftconvolve(x[::-1], y)


def lpc2ref(alpha):
    """
    Convert a set of LPC coefficients to reflection coefficients.
    Args:
        alpha - LPC coefficients (excluding 1)
    Returns:
        k     - reflection coefficients of the same order
    """
    order = len(alpha)
    a = np.zeros((order, order))
    a[-1] = alpha
    for i in range(order-2, -1, -1):
        a[i, :i+1] = (a[i+1, :i+1]+a[i+1, i+1] *
                      np.flipud(a[i+1, :i+1]))/(1-a[i+1, i+1]**2)
    return np.diag(a)


def ref2lpc(k):
    """
    Convert a set of reflection coefficients `k` to LPC coefficients `alpha`.
    Args:
        k     - reflection coefficients
    Returns:
        alpha - LPC coefficients (excluding 1)
    """
    alphas = np.diag(k)
    for i in range(1, alphas.shape[0]):
        alphas[i, :i] = alphas[i-1, :i] - k[i]*np.flipud(alphas[i-1, :i])
    return alphas[-1]


def lpc2stable(alpha):
    """
    Receive a set of LPC coefficients, and make them stable by reflecting
    any pole location outside the unit circle inside.
    Args:
        alpha   - LPC coefficients
        alpha_s - stable LPC coefficients
    """
    #pdb.set_trace()
    poles = np.roots(np.insert(-alpha, 0, 1))
    for i in range(len(poles)):
        if np.abs(poles[i]) > 1:
            poles[i] /= (np.abs(poles[i])**2)  # reflect back to unit circle
        if np.abs(poles[i]) > (1/1.01):
            # this is a temporary fix for pole location very close to 1
            # it might cause instability after ld recursion
            poles[i] /= np.abs(poles[i])
            poles[i] /= 1.01
    alpha_s = -np.poly(poles)[1:]
    if not lpc_is_stable(alpha_s):
        pdb.set_trace()
    return alpha_s


def ref2stable(k):
    """
    Make reflection coefficients stable.
    """
    return lpc2ref(lpc2stable(ref2lpc(k)))


def ref_is_stable(k):
    """
    Is the set of reflection coefficients stable?
    """
    return np.all(np.abs(k) < 1)


def lpc_is_stable(alpha):
    """
    Is the set of LPC coefficients stable?
    """
    return ref_is_stable(lpc2ref(alpha))


def ref2lar(k):
    """
    Convert a set of reflection coefficients to log area ratio.
    Args:
        k   - reflection coefficients
    Returns:
        g   - log area ratio (lar)
    """
    if np.greater_equal(k, 1).any():
        raise ValueError(
            "Reflection coefficient magnitude must be smaller than 1.")
    try:
        lar = np.log((1-k))-np.log((1+k))
    except RuntimeWarning:
        print("Invalid log argument")
        print(k)
        lar = 0
    return lar


def lpc2lar(alpha):
    """
    Convert a set of LPC coefficients to log area ratio.
    Args:
        alpha - reflection coefficients
    Returns:
        g     - log area ratio (lar)
    """
    return ref2lar(lpc2ref(alpha))


############## Test functions ###############
if __name__ == '__main__':

    # xcorr block
    a = np.array([1, 2], dtype='float_')
    b = np.array([1, 2, 3], dtype='float_')
    print("Testing [xcorr]...")
    r = xcorr(a, b)
    print("\txcorr({},{}) == {}".format(a, b, r))
    assert np.allclose(r, np.array([2, 5, 8, 3], dtype='float_'))
    r = xcorr(a, a)
    print("\txcorr({},{}) == {}".format(a, a, r))
    assert np.allclose(r, np.array([2, 5, 2], dtype='float_'))
    r = xcorr(b)
    print("\txcorr({}) == {}".format(b, r))
    assert np.allclose(r, np.array([14, 8, 3], dtype='float_'))
    print("Tests all passed for [xcorr].")

    # LPC block
    order = 5
    print("\n\nTesting [lpc] with order={}...".format(order))
    s = np.arange(10, dtype='float_')+1
    print("\tUsing levinson-durbin: lpc({})={}".format(s,
                                                       lpc(s, order, levinson=True)))
    print("\tUsing matrix inversion: lpc({})={}".format(s, lpc(s, order)))

    # LPC to Reflection coefficients block
    #alpha = np.array([1,2,3,4,5],dtype='float_')
    #print("\n\nTesting [lpc2ref]...")
    #k = lpc2ref(alpha)
    #k_answer = np.array([967004466,28046,202,9,5],dtype='float_')
    #print("Test passed? [{}]".format(np.allclose(k,k_answer)))

    # Reflection coefficients to LPC block
    k = np.array([1, 2, 3, 4, 5], dtype='float_')
    print("\n\nTesting [ref2lpc]...")
    alpha = ref2lpc(k)
    alpha_answer = np.array([-39, -170, 106, 99, 5], dtype='float_')
    print("Test passed? [{}]".format(np.allclose(alpha, alpha_answer)))
