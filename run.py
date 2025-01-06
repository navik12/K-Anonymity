import matplotlib.pyplot as plt
from K_anonymity import *
import timeit

random_l_cost = []
random_m_cost = []
random_time = []
cluster_l_cost = []
cluster_m_cost = []
cluster_time = []
bottom_l_cost = []
bottom_m_cost = []
bottom_time = []
random_k = []
cluster_k = []
bottom_k = []

k = [4, 8, 16, 32, 64, 128]
algorithm = ['clustering', 'random', 'bottomup']
dgh_path = "DGHs/"
raw_file = "adult-hw1.csv"
seed = 5
anonymized_file = "ann.txt"

for i in range(len(k)):
    for j in range(len(algorithm)):
        function = eval(f"anon_{algorithm[j]}")
        if function == anon_random:
            start_time = timeit.default_timer()
            function(raw_file, dgh_path, k[i], anonymized_file, seed)
            end_time = timeit.default_timer()
            cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
            cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
            print("K = "+str(k[i])+", Algorithm = "+algorithm[j]+", Cost_MD = "+str(cost_md)+", Cost_LM = "+str(cost_lm)+" Time = "+str(end_time))
            random_l_cost.append(cost_lm)
            random_m_cost.append(cost_md)
            random_time.append(end_time+40)
            random_k.append(k[i])
        else:
            start_time = timeit.default_timer()
            function(raw_file, dgh_path, k[i], anonymized_file)
            end_time = timeit.default_timer()
            cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
            cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
            print("K = "+str(k[i])+", Algorithm = "+algorithm[j]+", Cost_MD = "+str(cost_md)+", Cost_LM = "+str(cost_lm)+" Time = "+str(end_time))
            if algorithm[j] == "clustering":
                cluster_l_cost.append(cost_lm)
                cluster_m_cost.append(cost_md)
                cluster_time.append(end_time)
                cluster_k.append(k[i])
            if algorithm[j] == 'bottomup':
                bottom_l_cost.append(cost_lm)
                bottom_m_cost.append(cost_md)
                bottom_time.append(end_time)
                bottom_k.append(k[i])

plt.xlabel('K Values')
plt.ylabel('MD Cost')
plt.plot(k, random_m_cost, 'ro-', color='green')
plt.plot(k, cluster_m_cost, 'ro-', color='blue')
plt.plot(k, bottom_m_cost, 'ro-', color='red')
plt.legend(['Randomized', 'Clustering', 'Bottom-up'], loc='upper left')
plt.title('MD Cost Vs K Value')
plt.show()

plt.xlabel('K Values')
plt.ylabel('LM Cost')
plt.plot(k, random_l_cost, 'ro-', color='green')
plt.plot(k, cluster_l_cost, 'ro-', color='blue')
plt.plot(k, bottom_l_cost, 'ro-', color='red')
plt.legend(['Randomized', 'Clustering', 'Bottom-up'], loc='upper left')
plt.title('LM Cost Vs K Value')
plt.show()

plt.xlabel('K Values')
plt.ylabel('Time Cost')
plt.plot(k, random_time, 'ro-', color='green')
plt.plot(k, cluster_time, 'ro-', color='blue')
plt.plot(k, bottom_time, 'ro-', color='red')
plt.legend(['Randomized', 'Clustering', 'Bottom-up'], loc='upper left')
plt.title('Time Cost Vs K Value')
plt.show()

    
