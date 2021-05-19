def MAP_decision(p, mu,sigma, W):
    results = []
    for i in range(len(p)):
        temp = []
        threshold = (sigma**2/(2*mu))*np.log(p[i]/(1-p[i])) 
        for j in range(10):
            W_sum = sum(W[:,j])
            if W_sum <= threshold:
                temp.append(0)
            else:
                temp.append(1)  
        print("For 𝑃(H_0) " + str(p[i]) + ", the hypotheses selected are :: " + str(temp) + "\n")
        results.append(temp)
    return results