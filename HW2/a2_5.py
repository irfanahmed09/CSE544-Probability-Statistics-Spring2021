import numpy as np
def steady_state_power(transition_matrix):
    power = 1
    P = transition_matrix
    while(1):
        P = np.linalg.matrix_power(transition_matrix, power)
        P_hat = np.linalg.matrix_power(transition_matrix, power+1)
       
        if(np.array_equal(P,P_hat)):
            return P[0]
        else:
            power = power+1

# State transition probabilities
P = np.array([[.9, 0, .1, 0], [.8, 0, .2, 0], [0, .5, 0, .5], [0, .1,0,.9]])
a = steady_state_power(P)
a = np.around(a,decimals=2)
print("Steady state: "+str(a))

#Probability it'll snow 3 days from today
b = a[1] + a[3]
print("Probability it'll snow 3 days from today "+str(b))