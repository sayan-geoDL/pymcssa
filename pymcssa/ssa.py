import numpy as np
import math as ma
from .utils import Tmat

class SSA:
    """
    Basic Single-Channel Singular Spectrum Analysis (SSA).

    Performs Singular Spectrum Analysis (SSA) decomposition and reconstruction
    of a 1D time series using Singular Value Decomposition (SVD) of the
    trajectory (Hankel) matrix.

    This method decomposes the input signal into interpretable components such
    as trends, oscillations, and noise, based on temporal correlations.

    Attributes
    ----------
    data : ndarray
        Input 1D time series.
    m : int
        Embedding window length.
    k_vals : int
        Number of leading components to reconstruct.

    Notes
    -----
    SSA is a non-parametric spectral decomposition technique that extracts
    dominant temporal patterns without assuming any underlying model.
    """
    def __init__(self,data,m,k_vals):
        """
        Initialize the SSA object.

        Parameters
        ----------
        data : array-like
            Input 1D time series.
        m : int
            Embedding (window) length.
        k_vals : int
            Number of reconstructed components to retain.
        """
        self.data=np.asarray(data)
        self.m=m
        self.k_vals=k_vals
    def ssa(self):
        """
        Perform SSA decomposition and reconstruction.

        Constructs the trajectory matrix using `Tmat`, performs SVD to obtain
        eigenvectors (EOFs) and singular values, computes principal components,
        and reconstructs the leading `k_vals` components.

        Returns
        -------
        results : dict
            Dictionary containing:
            
            - **eofs** : ndarray, shape (m, m)
                EOFs (Empirical Orthogonal Functions) or spatial patterns.
            - **eigen_values** : ndarray, shape (m,)
                Eigenvalues (squared singular values) representing variance.
            - **percent_explained** : ndarray, shape (m,)
                Percentage of variance explained by each component.
            - **pcs** : ndarray, shape (n - m + 1, m)
                Principal components (temporal coefficients).
            - **rcs** : ndarray, shape (n, k_vals)
                Reconstructed time series components using first `k_vals` modes.

        Raises
        ------
        Exception
            If the input data is multi-dimensional (use MSSA or PCA instead).
        """

        n=np.shape(self.data)[0]
        if len(np.shape(self.data))>1:
            raise Exception("multi dimensional data use mssa or pca. can't make an ssa here")
        elif len(np.shape(self.data))==1:
            nn=n-self.m+1
            T=Tmat(self.data,self.m)
            T=T*(1/ma.sqrt(nn))
            T=np.array(T)
            eofs,s,rhot=np.linalg.svd(T)
            pcs=np.zeros((nn,self.m))
            for t in range(nn):
                for k in range(self.m):
                    for j in range(self.m):
                        pcs[t,k]+=(self.data[t+j]*eofs[j,k])
            p_vars=s*s
            exps=(p_vars/np.sum(p_vars))*100
            params=[0.0,0.0,0.0]
            rcs=np.zeros((n,self.k_vals))
            for k in range(self.k_vals):
                for t in range(n):
                    if t>=0 and t<=(self.m-2):
                        params[0]=(t+1)
                        params[1]=0
                        params[2]=t+1
                    if t>=self.m-1 and t<=nn-1:
                        params[0]=self.m
                        params[1]=0
                        params[2]=self.m
                    if t>=nn and t<=n-1:
                        params[0]=(n-t)
                        params[1]=t-n+self.m
                        params[2]=self.m
                    for j in range(params[1],params[2]):
                        rcs[t,k]=rcs[t,k]+((1/params[0])*(pcs[t-j,k]*eofs[j,k]))
            results={'eofs':eofs,
                     'eigen_values':p_vars,
                     'percent_explained':exps,
                     'pcs':pcs,
                     'rcs':rcs}
            return results
