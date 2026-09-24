import numpy as np 
import scipy as sc 
import statsmodels.api as sm 
from scipy.signal import savgol_filter
import skfda
from skfda.misc.hat_matrix import (
    KNeighborsHatMatrix,
    LocalLinearRegressionHatMatrix,
    NadarayaWatsonHatMatrix,
)
from skfda.preprocessing.smoothing import KernelSmoother
from skfda.preprocessing.smoothing.validation import SmoothingParameterSearch


def lowess(x, y, window = 100, n_points : int = 1000):
    '''
    '''
    smooth = sm.nonparametric.lowess(endog = y, exog = x, frac = window/len(y))

    x_smooth = np.linspace(smooth[:, 0].min(), smooth[:, 0].max(), n_points)
    y_smooth = np.interp(x_smooth, smooth[:, 0], smooth[:, 1])

    return x_smooth, y_smooth


def savgol(x, y, window = 101, polyorder = 3, n_points : int = 1000):
    '''
    '''
    y_smooth = savgol_filter(y, window_length = window, polyorder = polyorder)

    x_smooth = np.linspace(x.min(), x.max(), n_points)
    y_smooth = np.interp(x_smooth, x, y_smooth)

    return x_smooth, y_smooth


def convolve(x, y, window = 101, n_points : int = 1000):
    '''
    '''
    kernel = np.ones(window) / window

    y_forward = np.convolve(y, kernel, mode = "same")

    y_backward = np.convolve(y[::-1], kernel, mode = "same")[::-1]

    y_avg = np.mean([y_forward, y_backward], axis = 0)

    x_smooth = np.linspace(x.min(), x.max(), n_points)
    y_smooth = np.interp(x_smooth, x, y_avg)

    return x_smooth, y_smooth


def kernel_nadaraya_watson(
    x,
    y,
    bandwidth = None,
    bandwidths = None,
    n_points : int = 1000,
):
    '''
    '''
    fd = skfda.FDataGrid(
        data_matrix = np.asarray(y)[None, :],
        grid_points = np.asarray(x),
    )

    if bandwidths is not None:
        smoother = SmoothingParameterSearch(
            KernelSmoother(
                kernel_estimator = NadarayaWatsonHatMatrix(),
            ),
            bandwidths,
            param_name = "kernel_estimator__bandwidth",
        )

        smoother.fit(fd)
        fd_smooth = smoother.transform(fd)

    else:
        smoother = KernelSmoother(
            kernel_estimator = NadarayaWatsonHatMatrix(
                bandwidth = bandwidth,
            ),
        )

        fd_smooth = smoother.fit_transform(fd)

    x_smooth = np.linspace(x.min(), x.max(), n_points)
    y_smooth = np.interp(
        x_smooth,
        x,
        fd_smooth.data_matrix[0, :, 0],
    )

    return x_smooth, y_smooth


def kernel_local_linear(
    x,
    y,
    bandwidth = None,
    bandwidths = None,
    n_points : int = 1000,
):
    '''
    '''
    fd = skfda.FDataGrid(
        data_matrix = np.asarray(y)[None, :],
        grid_points = np.asarray(x),
    )

    if bandwidths is not None:
        smoother = SmoothingParameterSearch(
            KernelSmoother(
                kernel_estimator = LocalLinearRegressionHatMatrix(),
            ),
            bandwidths,
            param_name = "kernel_estimator__bandwidth",
        )

        smoother.fit(fd)
        fd_smooth = smoother.transform(fd)

    else:
        smoother = KernelSmoother(
            kernel_estimator = LocalLinearRegressionHatMatrix(
                bandwidth = bandwidth,
            ),
        )

        fd_smooth = smoother.fit_transform(fd)

    x_smooth = np.linspace(x.min(), x.max(), n_points)
    y_smooth = np.interp(
        x_smooth,
        x,
        fd_smooth.data_matrix[0, :, 0],
    )

    return x_smooth, y_smooth


def kernel_knn(
    x,
    y,
    n_neighbors = None,
    n_neighbors_grid = None,
    n_points : int = 1000,
):
    '''
    '''
    fd = skfda.FDataGrid(
        data_matrix = np.asarray(y)[None, :],
        grid_points = np.asarray(x),
    )

    if n_neighbors_grid is not None:
        smoother = SmoothingParameterSearch(
            KernelSmoother(
                kernel_estimator = KNeighborsHatMatrix(),
            ),
            n_neighbors_grid,
            param_name = "kernel_estimator__n_neighbors",
        )

        smoother.fit(fd)
        fd_smooth = smoother.transform(fd)

    else:
        smoother = KernelSmoother(
            kernel_estimator = KNeighborsHatMatrix(
                n_neighbors = n_neighbors,
            ),
        )

        fd_smooth = smoother.fit_transform(fd)

    x_smooth = np.linspace(x.min(), x.max(), n_points)
    y_smooth = np.interp(
        x_smooth,
        x,
        fd_smooth.data_matrix[0, :, 0],
    )

    return x_smooth, y_smooth