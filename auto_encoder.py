# Sparse auto-encoder
# by Sainaa

import numpy as np
import matplotlib.pyplot as plt

class AutoEncoder:
    sparsity = 0.035
    beta = 5
    lrate = 0.001
    batch_sz = 100
    lam = 0.003

    def __init__(self, in_dim, out_dim):
        r = np.sqrt(6) / np.sqrt(in_dim + out_dim + 1)
        self.W1 = np.random.rand(in_dim, out_dim)*2*r - r
        self.W2 = np.random.rand(out_dim, in_dim)*2*r - r
        self.b1 = np.zeros(out_dim)
        self.b2 = np.zeros(in_dim)

    def train(self, data, max_epoch):
        W1mom = np.zeros(self.W1.shape)
        W2mom = np.zeros(self.W2.shape)
        b1mom = np.zeros(self.b1.shape)
        b2mom = np.zeros(self.b2.shape)

        mom = 0
        cost = np.zeros(max_epoch)
        data_sz = data.shape[0]
        for epoch in range(max_epoch):
            avg_act = 0
            if epoch == 5:
                mom = 0.9
            for batch_ind in range(data_sz / self.batch_sz):
                batch = data[batch_ind * self.batch_sz: \
                                 (batch_ind+1) * self.batch_sz,:]
                self.calc_act(batch)
                cost[epoch] += self.calc_cost(batch)
                (W1grad, W2grad, b1grad, b2grad) = self.calc_grad(batch)
                W1mom = W1mom * mom - W1grad
                W2mom = W2mom * mom - W2grad
                b1mom = b1mom * mom - b1grad
                b2mom = b2mom * mom - b2grad
                self.W1 += self.lrate * W1mom
                self.W2 += self.lrate * W2mom
                self.b1 += self.lrate * b1mom
                self.b2 += self.lrate * b2mom

                avg_act += self.h1.mean()

            cost[epoch] /= (data_sz / self.batch_sz)
            avg_act /= (data_sz / self.batch_sz)
            print "%d\t%f\t%f" % (epoch, cost[epoch], avg_act)
            plt.plot(cost)

    def calc_act(self, data):
        # calculate activations
        a1 = data.dot(self.W1) + self.b1
        self.h1 = 1. / (1. + np.exp(-a1))
        a2 = self.h1.dot(self.W2) + self.b2
        self.h2 = a2

    def calc_cost(self, data):
        cost = 0.5*((data - self.h2)**2).sum(1).mean(0)        
        # sparsity cost
        cost_sparse = self.sparsity * np.log(self.sparsity / self.h1.mean(0)) \
            + (1.-self.sparsity) * np.log((1.-self.sparsity)/(1.-self.h1.mean(0)))
        cost += self.beta * cost_sparse.sum()
        # weight decay
        cost += self.lam * 0.5 * (self.W1**2).sum()
        cost += self.lam * 0.5 * (self.W2**2).sum()
        return cost

    def calc_grad(self, data):
        N = data.shape[0]
        t2 = (self.h2 - data) / N
        b2grad = t2.sum(0)
        W2grad = np.dot(self.h1.transpose(), t2)

        t1 = t2.dot(self.W2.transpose())
        t3 = self.beta * (-self.sparsity / self.h1.mean(0) \
                               + (1.-self.sparsity) / (1.-self.h1.mean(0))) / N
        t1 = t1 + t3
        t1 = t1 * self.h1 * (1 - self.h1)
        b1grad = t1.sum(0)
        W1grad = data.transpose().dot(t1)        
        #weight decay
        W1grad += self.lam * self.W1
        W2grad += self.lam * self.W2

        return (W1grad, W2grad, b1grad, b2grad)

