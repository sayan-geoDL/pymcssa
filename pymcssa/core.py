import numpy as np
import math as ma

class AR1estimator:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.n = len(data)
    def c_hats(self,l):
        """Estimate lag-l sample covariance (biased by n-l).

        Computes c_hat(l) = sum_{i=0}^{n-l-1} (x_i - mean)*(x_{i+l} - mean) / (n-l)

        Args:
            l (int): Lag value (non-negative).

        Returns:
            float: Estimated covariance at lag l.
        """
        ch=0
        for i in range(len(self.data)-l):
            ch+=((self.data[i]-np.mean(self.data))*(self.data[i+l]-np.mean(self.data)))
        ch=ch/(self.n-l)
        return ch
    def mu2(self,gam):
        """Compute mu2 function used in AR(1) parameter estimation.

        Args:
            gam (float): AR(1) coefficient.

        Returns:
            float: Value of mu2(gam) used in Newton-Raphson iterations.
        """
        mu2=0        
        for i in range(self.n-1):
            mu2+=2*(self.n-(i+1))*(gam**(i+1))
        mu2=(1/self.n)+((1/(self.n*self.n))*mu2)
        return mu2
    def tmumudash(self,gamma):
        """Compute derivative-like term tmumudash used in nr_dash.

        Args:
            gamma (float): AR(1) coefficient.

        Returns:
            float: Value used to construct derivative in nr_dash.
        """
        mud=0
        for i in range(self.n-1):
            mud+=(i+1)*(self.n-(i+1))*(gamma**i)
        mud=2*mud/(self.n*self.n)
        return mud
    def nr_func(self,gamma):
        """Newton-Raphson objective function for estimating gamma.

        The equation solved is (gamma - mu2(gamma)) / (1 - mu2(gamma)) = c1/c0.

        Args:
            gamma (float): Current gamma estimate.

        Returns:
            float: Function value.

        Raises:
            ZeroDivisionError: If estimated c0 (lag 0) is zero.
        """
        c1=self.c_hats(1)
        c0=self.c_hats(0)
        if c0 == 0:
            raise ZeroDivisionError("c0 is zero in nr_func")
        f=((gamma-self.mu2(gamma))/(1-self.mu2(gamma)))-(c1/c0)
        return f
    def nr_dash(self,gamma):
        """Approximate derivative of the NR objective for gamma update.

        Args:
            gamma (float): Current gamma estimate.

        Returns:
            float: Derivative approximation used in NR step.

        Raises:
            ZeroDivisionError: If estimated c0 (lag 0) is zero.
        """
        c1=self.c_hats(1)
        c0=self.c_hats(0)
        if c0 == 0:
            raise ZeroDivisionError("c0 is zero in nr_dash")
        denom=(1-self.mu2(gamma))
        if denom == 0:
            denom=1e-10
        fd=1+(self.tmumudash(gamma)*(self.nr_func(gamma)+(c1/c0)-1))
        fd=fd/denom
        return fd
    def gambar(self,max_iter=1000):
        """Estimate AR(1) coefficient gamma via Newton-Raphson.

        Starts from c1/c0 and iterates until relative error < 1e-6 or max_iter reached.

        Args:
            max_iter (int): Maximum NR iterations.

        Returns:
            float: Estimated gamma value (also stored as self.gam).
        """
        c1=self.c_hats(1)
        c0=self.c_hats(0)
        ini=c1/c0
        err=100
        iter_count=0
        while err>1e-6 and iter_count<max_iter:
            ini_prev=ini
            nr_f=self.nr_func(ini)
            nr_d=self.nr_dash(ini)
            if nr_d == 0:
                break
            ini=ini-(nr_f/nr_d)
            err=abs((ini-ini_prev)/(ini_prev if ini_prev!= 0 else 1))
            iter_count += 1
        self.gam=ini
        return ini
    def alph(self,max_iter=1000):
        """Estimate AR(1) innovation standard deviation alpha.

        Uses the estimated gamma and mu2 to compute alpha = sqrt(c0bar * (1 - gamma)),
        where c0bar = c0 / (1 - mu2(gamma)). Small positive floor applied to avoid
        negative or zero divisors.

        Args:
            max_iter (int): Maximum NR iterations.

        Returns:
            float: Estimated alpha (also stored as self.alpha).
        """
        gam=self.gambar(max_iter=max_iter)
        c0=self.c_hats(0)
        mu2_val=self.mu2(gam)
        denom=(1-mu2_val)
        if denom <= 0:
            denom=1e-10
        c0bar=c0/denom
        sqrt_term=c0bar*(1-gam)
        sqrt_term=sqrt_term if sqrt_term > 0 else 1e-10  # prevent sqrt of negative
        alpha=ma.sqrt(sqrt_term)
        self.alpha=alpha
        return alpha

    def ar1_model(self,alpha,gamma):
        """Generate a single AR(1) surrogate time series of length n.

        Model: r[t] = gamma * r[t-1] + alpha * w[t] with w ~ N(0,1) and r[0]=0.

        Args:
            alpha (float): Innovation std.
            gamma (float): AR(1) coefficient.

        Returns:
            ndarray: 1-D surrogate time series of length n.
        """
        rn=np.zeros(self.n)
        wn=np.random.normal(size=self.n)
        for i in range(1,self.n):
            rn[i]=gamma*(rn[i-1])+alpha*wn[i]
        return rn