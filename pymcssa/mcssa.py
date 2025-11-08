import numpy as np
import math as ma
from .core import AR1estimator
from .utils import Tmat
"""
mcssa module
------------

Provides a high-level Monte Carlo Singular Spectrum Analysis (MCSSA) class
built on a small AR(1) parameter estimator (AR1estimator). The MCSSA class
implements common MCSSA workflows: data-basis testing, ensemble-basis testing,
and Procrustes-aligned testing using AR(1) surrogates.

Public classes:
- MCSSA: main user-facing class for running MCSSA analyses.
"""

class MCSSA(AR1estimator):
    """
    Monte Carlo Singular Spectrum Analysis (MCSSA).

    Extends AR1estimator to provide MCSSA workflows that generate AR(1)
    surrogate ensembles, compute trajectory/covariance matrices, and produce
    percentile-based confidence bounds for SSA eigenvalues.

    Usage:
        mc = MCSSA(data, m)
        results = mc.mcssa_basic(up_perc=97.5, down_perc=2.5, ns=1000)

    Attributes:
        data (ndarray): 1-D input time series as numpy array.
        m (int): Window length for SSA embedding.
        n (int): Length of the input time series.
        nd (int): Number of rows in trajectory matrix (n - m + 1).
        gamma (float): Estimated AR(1) coefficient (set by inherited methods).
        alpha (float): Estimated AR(1) innovation std (set by inherited methods).
        surrs (ndarray): Last generated surrogate ensemble (ns x n).
    """
    def __init__(self,data,m):
        """
        Initialize MCSSA instance.

        Args:
            data (array-like): 1-D input time series.
            m (int): Window length (embedding dimension). Must satisfy 1 <= m <= len(data).

        Raises:
            ValueError: If m is larger than the length of the data.
        """
        super().__init__(data)
        self.data=np.asarray(data)
        self.m=m
        self.n=len(data)
        self.nd=self.n-self.m+1
        if self.nd<=0:
            raise ValueError("m must be<=length of data")
    def mcssa_basic(self,up_perc,down_perc,ns=100000,max_iter=1000,return_surrogates=False):
        """
        Monte-Carlo SSA using the data EOF basis.

        Generates `ns` AR(1) surrogates using the AR(1) parameters estimated
        from the data, projects each surrogate covariance matrix into the data's
        EOF basis and computes upper/lower percentile confidence bounds for the
        projected eigenvalues.

        Args:
            up_perc (float): Upper percentile (e.g. 97.5).
            down_perc (float): Lower percentile (e.g. 2.5).
            ns (int, optional): Number of surrogate realizations. Default 100000.
            max_iter (int, optional): Max iterations for AR(1) estimation routines.
            return_surrogates (bool, optional): If True, returned dict includes
                the surrogate ensemble under "surrogates".

        Returns:
            dict: {
                "data_eigenvalues": ndarray (sorted descending),
                "upper_confidence": list of upper percentile values,
                "lower_confidence": list of lower percentile values,
                "data_eigenvectors": ndarray (columns are EOFs),
                "gamma": float,
                "alpha": float,
                "spreads": ndarray (ns x m) of projected surrogate variances,
                ("surrogates": ndarray) optional
            }

        Raises:
            ValueError: If ns < 1, percentiles out of [0,100], or down_perc >= up_perc.
        """
        if ns < 1:
            raise ValueError("ns must be >= 1")
        if not (0 <= down_perc <= 100 and 0 <= up_perc <= 100):
            raise ValueError("Percentiles must be between 0 and 100")
        if down_perc >= up_perc:
            raise ValueError("down_perc must be less than up_perc")
        T_dat=Tmat(self.data,self.m)
        c=(1/self.nd)*np.transpose(T_dat)@T_dat
        evs,eofs=np.linalg.eigh(c)
        idx=np.argsort(evs)[::-1]
        evs=evs[idx]
        eofs=eofs[:, idx]
        self.gamma=round(self.gambar(max_iter=max_iter),8)
        self.alpha=round(self.alph(max_iter=max_iter),8)
        surrs=[]
        for i in range(ns):
            surrs.append(self.ar1_model(self.alpha,self.gamma))
        surrs=np.vstack(surrs)
        surrs=surrs-np.mean(surrs,axis=1,keepdims=True)
        self.surrs=surrs
        spreads=[]
        for i in range(np.shape(surrs)[0]):
            Tm=Tmat(surrs[i,:],self.m)
            c_s=(1/self.nd)*np.transpose(Tm)@Tm
            c_s_eval=np.transpose(eofs)@c_s@eofs
            c_s_eval=np.diag(c_s_eval)
            spreads.append(c_s_eval)
        spreads=np.vstack(spreads)
        ups=[]
        downs=[]
        for i in range(np.shape(spreads)[1]):
            ups.append(np.percentile(spreads[:,i],up_perc))
            downs.append(np.percentile(spreads[:,i],down_perc))
        result= {"data_eigenvalues": evs,
                  "upper_confidence": ups,
                  "lower_confidence": downs,
                  "data_eigenvectors": eofs,
                  "gamma": self.gamma,
                  "alpha": self.alpha,
                  "spreads": spreads,
                  "surrogates": self.surrs,
                  }
        if not return_surrogates:
            result.pop("surrogates")
        return result
    def mcssa_ensemble(self,up_perc,down_perc,ns=100000,max_iter=1000,return_surrogates=False):
        """
        Ensemble MCSSA using the mean surrogate covariance eigenbasis.

        Steps:
        - generate ns AR(1) surrogates,
        - compute each surrogate trajectory covariance and form their mean,
        - compute the eigenbasis of the mean covariance,
        - project surrogate covariances and the data covariance onto that basis,
        - compute percentile confidence bounds from projected surrogate spreads.

        Args:
            up_perc (float): Upper percentile for confidence.
            down_perc (float): Lower percentile for confidence.
            ns (int, optional): Number of surrogate realizations.
            max_iter (int, optional): Max iterations for AR(1) estimation.
            return_surrogates (bool, optional): If True include surrogates in result.

        Returns:
            dict: {
                "data_eigenvalues": ndarray (projected onto mean surrogate basis),
                "upper_confidence": list,
                "lower_confidence": list,
                "mean_surrogate_eigenvalues": ndarray,
                "mean_surrogate_eigenvectors": ndarray,
                "spreads": ndarray (ns x m),
                "alpha": float,
                "gamma": float,
                ("surrogates": ndarray) optional
            }

        Raises:
            ValueError: If ns < 1 or invalid percentile arguments.
        """
        if ns < 1:
            raise ValueError("ns must be >= 1")
        if not (0 <= down_perc <= 100 and 0 <= up_perc <= 100):
            raise ValueError("Percentiles must be between 0 and 100")
        if down_perc >= up_perc:
            raise ValueError("down_perc must be less than up_perc")
        self.gamma=round(self.gambar(max_iter=max_iter),8)
        self.alpha=round(self.alph(max_iter=max_iter),8)
        surrs=[]
        for i in range(ns):
            surrs.append(self.ar1_model(self.alpha,self.gamma))
        surrs=np.vstack(surrs)
        surrs=surrs-np.mean(surrs,axis=1,keepdims=True)
        self.surrs=surrs
        cs=[]
        for i in range(np.shape(surrs)[0]):
            Ts=Tmat(surrs[i,:],self.m)
            cs.append((1/self.nd)*np.transpose(Ts)@Ts)
        c_m=np.mean(np.stack(cs, axis=0), axis=0)
        c_m_eval, c_m_evs=np.linalg.eigh(c_m)
        self.c_mean_eval=c_m_eval
        self.c_mean_evs=c_m_evs
        idx=np.argsort(c_m_eval)[::-1]
        c_m_eval=c_m_eval[idx]
        c_m_evs=c_m_evs[:, idx]
        spreads=[]
        for i in range(np.shape(surrs)[0]):
            Traj=Tmat(surrs[i],self.m)
            c_s=(1/self.nd)*np.transpose(Traj)@Traj
            c_s_eval=c_m_evs.T@c_s@c_m_evs
            c_s_eval=np.diag(c_s_eval)
            spreads.append(c_s_eval)
        spreads=np.vstack(spreads)
        dats=self.data-np.mean(self.data)
        T_dat=Tmat(dats,self.m)
        c=(1/self.nd)*np.transpose(T_dat)@T_dat
        c_eval=np.transpose(c_m_evs)@c@c_m_evs
        c_eval=np.diag(c_eval)
        ups=[]
        downs=[]
        for i in range(np.shape(spreads)[1]):
            ups.append(np.percentile(spreads[:,i],up_perc))
            downs.append(np.percentile(spreads[:,i],down_perc))
        results={"data_eigenvalues": c_eval,
                   "upper_confidence": ups,
                   "lower_confidence": downs,
                   "mean_surrogate_eigenvalues": c_m_eval,
                   "mean_surrogate_eigenvectors": c_m_evs,
                   "surrogates": surrs,
                   "spreads": spreads,
                   "alpha": self.alpha,
                   "gamma": self.gamma}
        if not return_surrogates:
            results.pop("surrogates")
        return results
    def mcssa_procrustes(self,up_perc,down_perc,ns=100000,max_iter=1000,return_surrogates=False):
        """
        Procrustes-aligned MCSSA.

        For each surrogate:
        - compute surrogate covariance eigenvectors and eigenvalues,
        - scale surrogate EOFs by sqrt(eigenvalues),
        - compute the orthogonal Procrustes transform that best aligns the
          surrogate-scaled EOFs with the data-scaled EOFs,
        - apply the transform to surrogate eigenvalues and collect projected
          diagonal spreads to form percentile confidence bounds.

        Args:
            up_perc (float): Upper percentile for confidence.
            down_perc (float): Lower percentile for confidence.
            ns (int, optional): Number of surrogate realizations.
            max_iter (int, optional): Max iterations for AR(1) estimation.
            return_surrogates (bool, optional): If True include surrogates in the result.

        Returns:
            dict: {
                "data_eigenvalues": ndarray,
                "upper_confidence": ndarray,
                "lower_confidence": ndarray,
                "surrogate_eigenvalues": list of ndarrays,
                "surrogate_eigenvectors": list of ndarrays,
                "procrustes_transformations": list of ndarrays,
                "surrogates": ndarray,
                "alpha": float,
                "gamma": float
            }

        Raises:
            ValueError: If ns < 1 or invalid percentile arguments.
        """
        if ns < 1:
            raise ValueError("ns must be >= 1")
        if not (0 <= down_perc <= 100 and 0 <= up_perc <= 100):
            raise ValueError("Percentiles must be between 0 and 100")
        if down_perc >= up_perc:
            raise ValueError("down_perc must be less than up_perc")
        self.gamma=round(self.gambar(max_iter=max_iter),8)
        self.alpha=round(self.alph(max_iter=max_iter),8)
        surrs=[]
        for i in range(ns):
            surrs.append(self.ar1_model(self.alpha,self.gamma))
        surrs=np.vstack(surrs)
        surrs=surrs-np.mean(surrs,axis=1,keepdims=True)
        self.surrs=surrs
        l_rs, e_rs, e_rscaled=[],[],[]
        for i in range(np.shape(surrs)[0]):
            T_r=Tmat(surrs[i,:],self.m)
            c_r=(1/self.nd)*np.transpose(T_r)@T_r
            l_r, e_r=np.linalg.eigh(c_r)
            idx=np.argsort(l_r)[::-1]
            l_r=l_r[idx]
            e_r=e_r[:,idx]
            sigma_r=np.sqrt(l_r)
            e_scaled=e_r*sigma_r[np.newaxis,:]
            l_rs.append(l_r)
            e_rs.append(e_r)
            e_rscaled.append(e_scaled)
        T_dat=Tmat(self.data-np.mean(self.data),self.m)
        c=(1/self.nd)*np.transpose(T_dat)@T_dat
        l,e=np.linalg.eigh(c)
        idx=np.argsort(l)[::-1]
        l=l[idx]
        e=e[:,idx]
        sigma=np.sqrt(l)
        e_scaled=e*sigma[np.newaxis,:]
        T_es=[]
        for e_r_sig in e_rscaled:
            M=e_r_sig.T@e_scaled
            U,S,Vt=np.linalg.svd(M)
            T_e=U@Vt
            T_es.append(T_e)
        l_proj_list=[]
        for l_r,T_e in zip(l_rs,T_es):
            l_rmat=np.diag(l_r)
            l_proj=T_e.T@l_rmat@T_e
            l_proj_diag=np.diag(l_proj)
            l_proj_list.append(l_proj_diag)
        l_projs=np.vstack(l_proj_list)
        ups=np.percentile(l_projs,up_perc,axis=0)
        downs=np.percentile(l_projs,down_perc,axis=0)
        results={"data_eigenvalues": l,
                   "upper_confidence": ups,
                   "lower_confidence": downs,
                   "surrogate_eigenvalues": l_rs,
                   "surrogate_eigenvectors": e_rs,
                   "procrustes_transformations": T_es,
                   "surrogates": surrs,
                   "alpha": self.alpha,
                   "gamma": self.gamma}
        if not return_surrogates:
            results.pop("surrogates")
        return results