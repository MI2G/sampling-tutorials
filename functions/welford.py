# Welford's algorithm for calculating mean and variance
# Collection of summary statistics of the Markov chain iterates
# 
# call update() when generating a new sample
# call get_mean() or get_var() to obtain the current mean or variance
#
# https://doi.org/10.2307/1266577
class welford: 
    def __init__(self, x):
        self.k = 1
        self.M = x.clone()
        self.S = 0

    def update(self, x):
        self.k += 1
        Mnext = self.M + (x - self.M) / self.k
        self.S += (x - self.M)*(x - Mnext)
        self.M = Mnext
    
    def get_mean(self):
        return self.M
    
    def get_var(self):
        #when the number of samples is 1, we divide by self.k 
        if self.k < 2:
            return self.S/(self.k)
        else:
            return self.S/(self.k-1)