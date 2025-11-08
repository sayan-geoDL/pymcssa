import numpy as np
import warnings

def Tmat(ts, m):
        """Construct a trajectory (Hankel) matrix from a 1-D time series.
        Each row of the resulting matrix represents a lagged segment of the input
        series of length `m`. The matrix has shape (nd, m), where nd = n - m + 1.
        
        Args:
        ts (array-like): 1-D time series of length n.
        m (int): Window length (embedding dimension). Must satisfy 1 < m <= n.
        
        Returns:
        ndarray: Trajectory matrix of shape (nd, m).
        
        Raises:
        ValueError: If m > n.
        
        Warns:
        UserWarning: If m >= n/2, which may lead to reduced separability of 
                     components in SSA or MCSSA analysis.
        """
        ts = np.asarray(ts)
        n=len(ts)
        if m>n:
            raise ValueError("Window length m cannot exceed time series length .")
        if m >= n/2:
             warnings.warn("Window length m is large (m ≥ n/2); "
                           "this may reduce separability of components.",UserWarning)

        nd=n-m+1
        T=np.zeros((nd,m))
        for i in range(np.shape(T)[0]):
            for j in range(np.shape(T)[1]):
                T[i,j]=ts[i+j]
        return T