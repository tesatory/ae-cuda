from numpy import *
from auto_encoder import *
from grad_check import *

def ae_cost(params, data):
    ae = AutoEncoder(10,15)
    i = 0
    ae.W1 = params[i:i+ae.W1.size].reshape(ae.W1.shape)
    i += ae.W1.size
    ae.W2 = params[i:i+ae.W2.size].reshape(ae.W2.shape)
    i += ae.W2.size
    ae.b1 = params[i:i+ae.b1.size].reshape(ae.b1.shape)
    i += ae.b1.size
    ae.b2 = params[i:i+ae.b2.size].reshape(ae.b2.shape)

    ae.calc_act(data)
    return ae.calc_cost(data)


data = random.randn(20,10)
ae = AutoEncoder(10,15)
params = concatenate((ae.W1.flatten(), 
                      ae.W2.flatten(), 
                      ae.b1.flatten(), 
                      ae.b2.flatten()))
ae.calc_act(data)
(a,b,c,d) = ae.calc_grad(data)
grad = concatenate((a.flatten(), 
                    b.flatten(), 
                    c.flatten(), 
                    d.flatten()))
(grad1) = calc_grad_numeric(ae_cost, params, data)

print np.mean(np.abs(grad - grad1))
