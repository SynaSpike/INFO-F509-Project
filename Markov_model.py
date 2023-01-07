import numpy as np
from math import factorial as fact
from math import exp
from random import randint
from random import randrange



#Population size
Z = 200 #Number of individuals in the population
Zr = 40 #Number of rich individuals in the population
Zp = Z - Zr #Number of poor individuals in the population
N = 6 #groups of size N
Nr = Zr/N #Number of rich individuals in each group
Np = Zp/N #Number of poor individuals in each group

#Initial endowment (br > bp)
br = 2.5 #Initial endowment of the rich
bp = 0.625 #Initial endowment of the poor

b_hat = (br*Zr + bp*Zp) / Z #Average endowment of the population

#Population of Cs and Ds
Icr = 15
Icp = 50
Idr = Zr - Icr #nbr rich Ds in the population
Idp = Zp - Icp #nbr poor Ds in the population

#Contributions
c = 0.25 #fraction of the endowment contributed by Cs to help solve the group task

Cr = c*br #Contribution of the rich Cs
Cp = c*bp #Contribution of the poor Cs

#Lost if target not meet
lost_Cr = br*(1-c) #Lost of the rich Cs if next intermediate target is not meet
lost_Cp = bp*(1-c) #Lost of the poor Cs if next intermediate target is not meet
lost_Dr = br #Lost of the rich Ds if next intermediate target is not meet
lost_Dp = bp #Lost of the poor Ds if next intermediate target is not meet

M = 3*c*b_hat #positive integer between O and N
Mcb = M*c*b_hat #Threshold  for the target to be met

r = 0.2 #Perception of risk (varying between 0 and 1)

h = 0 #Homophily parameter (varying between 0 and 1)
        #When h = 1, individuals are restricted to influence by those of the same wealth status
        #When h = 0, no wealth discrimination takes place


############## Functions#############

def add_lists(list1, list2):
    result = [x1 + x2 for (x1, x2) in zip(list1, list2)]
    return result

def mult_lists(list1, list2):
    result = [x1 * x2 for (x1, x2) in zip(list1, list2)]
    return result

def diff_lists(list1, list2):
    result = [x1 - x2 for (x1, x2) in zip(list1, list2)]
    return result
def get_column(matrix, index):
    return [row[index] for row in matrix]

def O_(k):
    res = 0
    if k >=0:
        res = 1
    return res

def payoff_DR(Jr,Jp):
    ''' Payoff for rich defector player'''
    delta = Cr * Jr + Cp * Jp - Mcb
    return br * (O_(delta) + (1 - r) * (1 - O_(delta)))

def payoff_DP(Jr,Jp,Cr=Cr, Cp=Cp, Mcb=Mcb,bp=bp):
    ''' Payoff for poor defector player'''
    delta = Cr * Jr + Cp * Jp - Mcb
    return bp * (O_(delta) + (1 - r) * (1 - O_(delta)))

def binom(n,r):
    '''binomila product'''
    if n < r:
        return 0
    else:
        return fact(n)//fact(r)//fact(n-r)

def fitness(wealth, strat, Ir,Ip, N=N, Z=Z):
    '''fitness for rich cooperator'''
    res = 0
    if wealth == 'R' and strat == 'C':
        for jr in range(0,(N)):
            for jp in range (0,(N-jr)):
                P_C_R = payoff_DR(jr+1,jp) - Cr
                res += binom(Ir - 1, jr) * binom(Ip, jp) * binom(Z-Ir-Ip, N-1-jr-jp) * P_C_R
        return res * (binom(Z-1,N-1))**(-1)

    elif wealth == 'P' and strat == 'C':
        for jr in range(0, (N)):
            for jp in range(0, (N - jr)):
                P_C_P = payoff_DP(jr,jp + 1) - Cp
                res += binom(Ir, jr) * binom(Ip - 1, jp) * binom(Z - Ir - Ip, N - 1 - jr - jp) * P_C_P
        return res * (binom(Z -1, N - 1)) ** (-1)

    elif wealth == 'R' and strat == 'D':
        for jr in range(0, (N)):
            for jp in range(0, (N - jr)):
                P_D_R = payoff_DR(jr,jp)
                res += binom(Ir, jr) * binom(Ip, jp) * binom(Z - 1 - Ir - Ip, N - 1 - jr - jp) * P_D_R
        return res * (binom(Z - 1, N - 1)) ** (-1)

    elif wealth == 'P' and strat == 'D':
        for jr in range(0, (N)):
            for jp in range(0, (N - jr)):
                P_D_P = payoff_DP(jr, jp)
                res += binom(Ir, jr) * binom(Ip, jp) * binom(Z - 1 - Ir - Ip, N - 1 - jr - jp) * P_D_P
        return res * (binom(Z - 1, N - 1)) ** (-1)


def transition(k,X, Idr, Icr, Idp, Icp,u=1/Z,beta=3,h=0):
    '''
    The transition probabilities gives the probability that an individual
    with strategy X ∈ C;D in the subpopulation k ∈ R;P changes
    to a different strategy Y ∈ C;D, both from the same subpopulation k
    and from the other population l (that is, l = P if k = R, and l = R if k = P)
    :param k: rich or poor {'R','P'}
    :param X: cooperator or defector {'C','D'}
    :param u: mutation probability μ
    :param beta: β controls the intensity of selection
    :return: list with transition probabilities from a strategy X to Y
            Y = [RD, RC, PD, PC]
    '''
    if k == 'R'and X == 'C':
        #RC TO RD
        a = (Idr/(Zr-1 + (1-h)*Zp))*(1 + exp(beta * (fitness('R','C',Icr, Icp)-fitness('R','D',Icr, Icp))))**(-1)
        b = ((1-h)*Idp/(Zr-1 + (1-h)*Zp))*(1 + exp(beta * (fitness('R','C',Icr, Icp)-fitness('P','D',Icr, Icp))))**(-1)

        return [Icr/ Z * ((1-u)*(a+b)+u), 1 -(Icr/ Z * ((1-u)*(a+b)+u)),0,0]

    elif k == 'R'and X == 'D':
        #RD to RC
        a = (Icr / (Zr - 1 + (1 - h) * Zp)) * (1 + exp(beta * (fitness('R','D',Icr, Icp) - fitness('R','C',Icr, Icp))))**(-1)
        b = ((1 - h) * Icp / (Zr - 1 + (1 - h) * Zp)) * (1 + exp(beta * (fitness('R','D', Icr, Icp) - fitness('P', 'C',Icr, Icp))))**(-1)

        return [1 - (Idr / Z * ((1-u)*(a+b)+u)),Idr / Z * ((1-u)*(a+b)+u),0,0]

    elif k == 'P'and X == 'C':
        #PC to PD
        a = (Idp/(Zp-1 + (1-h)*Zr))*(1 + exp(beta * (fitness('P','C',Icr, Icp)-fitness('P','D',Icr, Icp))))**(-1)
        b = ((1-h)*Idr/(Zp-1 + (1-h)*Zr))*(1 + exp(beta * (fitness('P','C',Icr, Icp)-fitness('R','D',Icr, Icp))))**(-1)

        return [0,0, Icp / Z * ((1-u)*(a+b)+u),1 -(Icp / Z * ((1-u)*(a+b)+u))]

    elif k == 'P'and X == 'D':
        #PD to PC
        a = (Icp/(Zp-1 + (1-h)*Zr))*(1 + exp(beta * (fitness('P','D',Icr, Icp)-fitness('P','C',Icr, Icp))))**(-1)
        b = ((1-h)*Icr/(Zp-1 + (1-h)*Zr))*(1 + exp(beta * (fitness('P','D',Icr, Icp)-fitness('R','C',Icr, Icp))))**(-1)

        return [0,0, 1 - (Idp / Z * ((1-u)*(a+b)+u)),Idp / Z * ((1-u)*(a+b)+u)]

print(transition('R','C',20,20,80,80))
def GoS(Idr, Icr, Idp, Icp):
    ''' Function to compute the gradient of selection.
    For each configuration i =fiR; iPg,
    we compute the most likely path each subpopulation k∈fR; Pg will
    follow, resorting to the probability to increase (decrease) by one, in
    each time step, the number of cooperators for that configuration
    i of the population, which we denote by T+i;k (T−i;k),
    :param Idr: Nbr of rich defectors
    :param Idp: Nbr of poor defectors
    :param Icr: Nbr of rich cooperators
    :param Icp: Nbr of poor cooperators
    :return: the gradient of selection ∇i (GoS)
    '''
    Tpr = transition('R','D', Idr, Icr, Idp, Icp)[1]
    Tnr = transition('R', 'C', Idr, Icr, Idp, Icp)[0]
    Tpp = transition('P', 'D', Idr, Icr, Idp, Icp)[3]
    Tnp = transition('P', 'C', Idr, Icr, Idp, Icp)[2]

    return(Tpr-Tnr, Tpp-Tnp)

def prevalence(Idr, Icr, Idp, Icp):
    return [Idr/Z,Icr/Z,Idp/Z,Icp/Z]

def prevalence_all(Zr=Zr, Zp=Zp, Z=Z):

    '''Calculate the prevalence of each state at each time'''
    pvlc = []
    for Icr in range(0, Zr):
        for Icp in range(0, Zp):
            Idr = Zr - Icr
            Idp = Zp - Icp
            sub_pvlc = []
            sub_pvlc.append([Icr/Z,Icp/Z,Idr/Z,Idp/Z])
            pvlc.append(sub_pvlc)
    return (pvlc)

def markov_model (Icr, Icp, Zr=Zr, Zp=Zp,Z=Z):
    '''
    :param Icr: nbr of rich cooperators
    :param Icp: nbr of poor cooperators
    :param Zr: nbr of rich in population
    :param Zp: nbr of poor in population
    :param Z: size of population
    :return: list of new population [Idr, Icr, Idp, Icp]
    '''
    Idr = Zr - Icr
    Idp = Zp - Icp
    population = [Idr, Icr, Idp, Icp]
    strat = ['RD','RC','PD','PC']
    prev = prevalence(Idr, Icr, Idp, Icp)
    transition_matrix = []
    for elem in strat:
        transition_matrix.append(transition(elem[0],elem[1], Idr, Icr, Idp, Icp))

    proba = np.dot(prev,transition_matrix)
    print(transition_matrix)
    print(proba)
    print(sum(proba))

    new_pop = Z*proba
    new_pop_int = []
    for elem in new_pop:
        x = round(elem)
        new_pop_int.append(x)
    return population,new_pop_int


def equilibrium(Icr, Icp):
    prev_pop = markov_model(Icr,Icp)[0]
    new_pop = markov_model(Icr,Icp)[1]
    while prev_pop != new_pop:
        equilibrium(new_pop[1],new_pop[3])
    return(prev_pop, new_pop)

def transition_matrix(Idr, Icr, Idp, Icp):
    a =transition('R', 'C', Idr, Idp, Icr, Icp)
    b = transition('R', 'D', Idr, Idp, Icr, Icp)
    c =transition('P', 'C', Idr, Idp, Icr, Icp)
    d = transition('P', 'D', Idr, Idp, Icr, Icp)
    return [a,b,c,d]

def W(p,q):
    dico_p = {}
    dico_q ={}
    i= -1
    for Icr in range(0,Zr):
        for Icp in range(0,Zp):
            i +=1
            dico_p[i]=(Icr, Icp)
            dico_q[i]=(Icr, Icp)
    return(dico_p[p],dico_q[q])

def markov(initial_state, Zr=Zr, Zp=Zp, Z=Z):
    strat = ['RD','RC','PD','PC']
    res = [0,0,0,0]
    Icr = initial_state[0]
    Icp = initial_state[1]
    Idr = Zr - Icr
    Idp = Zp-Icp

    #Matrice de transtion dans les 4 scénarios possibles
    tm = [transition(elem[0], elem[1], Idr, Icr, Idp, Icp) for elem in strat]
    tm_crp = [transition(elem[0], elem[1], Idr-1, Icr+1, Idp, Icp) for elem in strat]
    tm_crm = [transition(elem[0], elem[1], Idr + 1, Icr - 1, Idp, Icp) for elem in strat]
    tm_cpp = [transition(elem[0], elem[1], Idr, Icr, Idp - 1, Icp + 1) for elem in strat]
    tm_cpm = [transition(elem[0], elem[1], Idr, Icr, Idp + 1, Icp - 1) for elem in strat]

    pi = prevalence(Idr, Icr, Idp, Icp)
    pip_crp = prevalence(Idr - 1, Icr + 1, Idp, Icp)
    pip_crm = prevalence(Idr + 1, Icr - 1, Idp, Icp)
    pip_cpp = prevalence(Idr, Icr, Idp - 1, Icp + 1)
    pip_cpm = prevalence(Idr, Icr, Idp + 1, Icp - 1)

    #CR +1
    Tiip = tm_crp[1]
    Tipi = tm[0]
    print(Tipi)
    print(pi)
    print(mult_lists(pi,Tipi))
    print('CR +1:',diff_lists(mult_lists(pip_crp,Tiip),mult_lists(pi,Tipi)))
    res = add_lists(res,diff_lists(mult_lists(pip_crp,Tiip),mult_lists(pi,Tipi)))

    #CR -1
    Tiip = tm_crm[0]
    Tipi = tm[1]
    print('CR -1:',diff_lists(mult_lists(pip_crm,Tiip),mult_lists(pi,Tipi)))
    res = add_lists(res,diff_lists(mult_lists(pip_crm,Tiip),mult_lists(pi,Tipi)))

    #CP +1
    Tiip = tm_cpp[3]
    Tipi = tm[2]
    print('CP +1:',diff_lists(mult_lists(pip_cpp,Tiip),mult_lists(pi,Tipi)))
    res = add_lists(res,diff_lists(mult_lists(pip_cpp,Tiip),mult_lists(pi,Tipi)))

    # CP -1
    Tiip = tm_cpm[2]
    Tipi = tm[3]
    print('CP -1:',diff_lists(mult_lists(pip_cpm,Tiip),mult_lists(pi,Tipi)))
    res = add_lists(res,diff_lists(mult_lists(pip_cpm,Tiip),mult_lists(pi,Tipi)))

    return res

def stationary_distribution(Zr=Zr,Zp=Zp):
    strat = ['RD', 'RC', 'PD', 'PC']
    for Icr in range(0, Zr):
        for Icp in range(0, Zp):
            Idr = Zr - Icr
            Idp = Zp - Icp
            transition_matrix = [transition(elem[0], elem[1], Idr, Icr, Idp, Icp) for elem in strat]
            eigenvalues, eigenvectors = np.linalg.eig(transition_matrix)
            index = np.where(eigenvalues == 1)[0][0]
            stationary_distribution = eigenvectors[:, index]
            print('--------------', (Icr, Icp), '--------------')
            print("Eigenvalues:", eigenvalues)
            print("Eigenvectors:", eigenvectors)
            print("Stationary distribution:", stationary_distribution)

stationary_distribution()
#stationary_distribution()

'''for Icr in range(0,Zr):
    for Icp in range (0,Zp):
        res = markov((Icr,Icp))
        if res == [0,0,0,0]:
            print('YOOUUUPIIIII --- ', (Icr, Icp),' ---')'''


'''def markov_v2(initial_state, Zr=Zr, Zp=Zp, Z=Z):
    strat = ['RD','RC','PD','PC']
    res = [0,0,0,0]
    Icr = initial_state[0]
    Icp = initial_state[1]
    Idr = Zr - Icr
    Idp = Zp-Icp

    population = [Icr, Icp, Idr, Idp]

    tm = [transition(elem[0], elem[1], Idr, Icr, Idp, Icp) for elem in strat]
    pi = prevalence(Idr, Icr, Idp, Icp)

    proba = np.dot(pi, tm)
    new_pop_float = Z * proba
    new_pop = []
    for elem in new_pop_float:
        x = round(elem)
        new_pop.append(x)
    print(population, new_pop)

    tmp = [transition(elem[0], elem[1], new_pop[0], new_pop[1], new_pop[2], new_pop[3]) for elem in strat]
    pip = prevalence(new_pop[0], new_pop[1], new_pop[2], new_pop[3])
    probap = np.dot(pip, tmp)
    new_pop_float_p = Z * probap
    new_pop_p = []
    for elem in new_pop_float_p:
        x = round(elem)
        new_pop_p.append(x)

    print(new_pop, new_pop_p)

    res = proba - probap

    return res

print(markov_v2((20, 80)))'''




###### Main code ######

'''strat = ('RD','RC','PD','PC')
trans_matrix = []
W= {}
p = 0
q = 0
for Icr in range(0,Zr):
    p +=1
    for Icp in range(0,Zp):
        q +=1
        
        Idr = Zr - Icr
        Idp = Zp - Icp

        print('--------------',(Icr,Icp),'--------------')

        ########## Fitness #########
        print('------ Fitness ------')
        fit = []
        for elem in strat:
            fit.append(fitness(elem[0],elem[1],Icr, Icp))
        print('RD: ',fit[0])
        print('RC: ',fit[1])
        print('PD: ',fit[2])
        print('PC: ',fit[3])

        ########## Gradient of selection #########

        print('------ Gradient of selection ------')
        print(GoS(Idr, Idp, Icr, Icp))

        ########## Transition probabilities #########

        print('------ Transition matrix ------')
        tm = transition_matrix(Idr, Icr, Idp, Icp)
        print(tm)
        trans_matrix.append(tm)

        eigenvalues, eigenvectors = np.linalg.eig(np.array(sub_list))
        print('eigenvalues:',eigenvalues)
        print('eigenvectors:',eigenvectors)
        print(eigenvalues[0], eigenvalues[2])
        print(np.where(eigenvalues == 1))
        index = np.where(eigenvalues == 1)[0][0]
        eigenvector = eigenvectors[:, index]
        print('eigenvector:',eigenvector)'''


print('XXXXXX',transition('R','P',20,20,80,80))