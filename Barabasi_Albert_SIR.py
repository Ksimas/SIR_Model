import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import copy

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

def func(y, t, N, beta, gamma):
    # S, I, R values assigned from vector
    S, I, R = y
    # differential equations
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def draw_graph():
    nx.draw_networkx_nodes(G, pos, nodelist=[x for x in range(0, N, 1) if SN[x] == b'S'], node_color='b', label='S')
    nx.draw_networkx_nodes(G, pos, nodelist=[x for x in range(0, N, 1) if SN[x] == b'I'], node_color='r', label='I')
    nx.draw_networkx_nodes(G, pos, nodelist=[x for x in range(0, N, 1) if SN[x] == b'R'], node_color='g', label='R')
    nx.draw_networkx_edges(G, pos)


def save_figure(day_number):
    plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    draw_graph()
    plt.xlabel(r'$\tau=0.3$, $\gamma = 0.1$, $N = 50$', size = 40)
    plt.title('Day: {}'.format(day_number), size=50)
    plt.legend(loc='upper right', prop={'size': 30})
    if day_number <= 9:
        plt.savefig("Graph_0{0}.png".format(day_number))
    else:
        plt.savefig("Graph_{0}.png".format(day_number))
    plt.close()

#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------#

T = 100  # time [days]
time_of_simulation = np.arange(0, T, 1)  # time used for the plot
N = 50  # total population

b = 2  # number of contacts each individuals
beta = 0.3 # probability of infection 
gamma = 0.1 # probability of recover

number_of_repetitions = 2
II = np.zeros((T, number_of_repetitions))
RR = np.zeros((T, number_of_repetitions))
SS = np.zeros((T, number_of_repetitions))


repeat = 0
while repeat < number_of_repetitions:
    
    I = np.zeros(T)  # number of infected individuals
    I[0] = 1
    I_deterministic = np.zeros(T) 
    I_deterministic[0] = 1

    R = np.zeros(T)  # number of recovered individuals
    R[0] = 0
    R_deterministic = np.zeros(T) 
    R_deterministic[0] = 0

    S = np.zeros(T)  # number of susceptible individuals
    S[0] = N - I[0] - R[0]
    S_deterministic = np.zeros(T) 
    S_deterministic[0] = N - I_deterministic[0] - R_deterministic[0]

    SN = np.chararray(N)  # status of nodes (S/I/R)
    SN[:] = 'S'  # at the beginning every node has status S

    # first random infected
    r = np.random.randint(0, N)
    SN[r] = 'I'  # change status from S -> I

    G = nx.barabasi_albert_graph(N, b)  # create graph

    pos = nx.spring_layout(G)  
    save_figure(0)

    Tend = T
    for t in range(1, T, 1):  # MONTE CARLO STEPS
        SN_temp = copy.copy(SN)
        I[t] = I[t - 1]
        R[t] = R[t - 1]
        index_of_node = 0
        
        if I[t] == 0:  # condition checking if infection died out
            save_figure(t)
            Tend = t
            break

        for node_status in SN:
            if node_status == b'I':  # if node is infected we are looking for his susceptible neighbors
                nearest_s_neighbors = []
                for neighbor in G.neighbors(index_of_node):
                    if SN_temp[neighbor] == b'S':   # if node is Susceptible
                        nearest_s_neighbors.append(neighbor)

                if nearest_s_neighbors:  # if susceptible neighbors exist
                    for neighbor in nearest_s_neighbors:  
                        r = np.random.uniform(0, 1) # the infected node tries to infect neighbors
                        if r < beta:
                            SN_temp[neighbor] = 'I' # change status to Infected
                            I[t] += 1 

                r = np.random.uniform(0,1)  # attempt to recovery
                if r < gamma:
                    SN_temp[index_of_node] = 'R'  # change status to R
                    I[t] -= 1
                    R[t] += 1


            index_of_node += 1

        SN = copy.copy(SN_temp)
        save_figure(t)  # create a plot (graph)


    if Tend < T:  # if simulation ended ealier than T
        I[Tend:] = I[Tend]
        R[Tend:] = R[Tend]
    S[:] = N - I[:] - R[:]

    SS[:,repeat] = S
    II[:,repeat] = I
    RR[:,repeat] = R

    repeat += 1

II_avg = [sum(i)/repeat for i in II]
SS_avg = [sum(i)/repeat for i in SS]
RR_avg = [sum(i)/repeat for i in RR]

# deterministic model
y = S_deterministic[0], I_deterministic[0], R_deterministic[0]
SIR = odeint(func, y, time_of_simulation, args=(N, beta*(2*b - b/N - b/(N**2)), gamma))
S_deterministic, I_deterministic, R_deterministic = SIR.T


# plot for average results
plt.figure()
plt.plot(time_of_simulation, SS_avg, lw=1.5, color='blue', label='<S>')
plt.plot(time_of_simulation, II_avg, lw=1.5, color='red', label='<I>')
plt.plot(time_of_simulation, RR_avg, lw=1.5, color='green', label='<R>')
plt.plot(time_of_simulation, S_deterministic, lw=1.5, linestyle=':', color='magenta', label='S deterministic')
plt.plot(time_of_simulation, I_deterministic, lw=1.5, linestyle=':', color='darkorange', label='I deterministic')
plt.plot(time_of_simulation, R_deterministic, lw=1.5, linestyle=':', color='limegreen', label='R deterministic')
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend(loc='upper right')
plt.show()