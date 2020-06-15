from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
import numpy as np


class LinearKernel(Kernel):
    """Linear kernel.
    The Linear kernel is non-stationary and can be obtained from Bayesiann
    linear regression by putting N(0, \sigma_v^2) priors on the coefficients of
    x_d (d = 1, . . . , D) and a prior of N(0, \sigma_b^2) on the bias. It is
    parameterized by parameters sigma_v and sigma_0. The parameters of the
    Linear kernel are about specifying the origin. The kernel is given by:
    k(x_i, x_j) = sigma_b ^ 2 + sigma_v ^ 2 * (x_i \cdot x_j)
    The Linear kernel can be combined with other kernels, more commonly with
    periodic kernels.
    Parameters
    ----------
    sigma_b : float >= 0, default: 1.0
        Parameter adding an uncertainty offset to the kernel.
    sigma_v : float >= 0, default: 1.0
        Parameter adding an uncertainty offset to the kernel.
    sigma_b_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on sigma_b
    sigma_v_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on sigma_v
    """

    def __init__(self, sigma_b=1.0, sigma_v=1.0, sigma_b_bounds=(1e-2, 1e2),
                 sigma_v_bounds=(1e-2, 1e2)):
        self.sigma_b = sigma_b
        self.sigma_v = sigma_v
        self.sigma_b_bounds = sigma_b_bounds
        self.sigma_v_bounds = sigma_v_bounds

    @property
    def hyperparameter_sigma_b(self):
        return Hyperparameter("sigma_b", "numeric", self.sigma_b_bounds)

    @property
    def hyperparameter_sigma_v(self):
        return Hyperparameter("sigma_v", "numeric", self.sigma_v_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        if Y is None:
            K = (self.sigma_v ** 2) * np.inner(X, X) + (self.sigma_b ** 2)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            K = (self.sigma_v ** 2) * np.inner(X, Y) + (self.sigma_b ** 2)

        if eval_gradient:
            # gradient with respect to sigma_b
            if not self.hyperparameter_sigma_b.fixed:
                sigma_b_gradient = np.empty((K.shape[0], K.shape[1], 1))
                sigma_b_gradient[..., 0] = 2 * self.sigma_b ** 2
            else:
                sigma_b_gradient = np.empty((X.shape[0], X.shape[0], 0))
            # gradient with respect to sigma_v
            if not self.hyperparameter_sigma_v.fixed:
                sigma_v_gradient = 2 * self.sigma_v ** 2 * np.inner(X, X)
                sigma_v_gradient = sigma_v_gradient[:, :, np.newaxis]
            else:
                sigma_v_gradient = np.empty((X.shape[0], X.shape[0], 0))
            return K, np.dstack((sigma_b_gradient, sigma_v_gradient))
        else:
            return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).
        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return (self.sigma_v ** 2) * np.einsum('ij,ij->i', X, X) + \
            self.sigma_b ** 2

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return False

    def __repr__(self):
        return "{0}(sigma_b={1:.3g}, sigma_v={2:.3g})".format(
            self.__class__.__name__, self.sigma_b, self.sigma_v)

