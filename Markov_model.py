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


def transition(k,X, Idr, Idp, Icr, Icp,u=1/Z,beta=3,h=0):
    '''
    The transition probabilities gives the probability that an individual
    with strategy X ∈ C;D in the subpopulation k ∈ R;P changes
    to a different strategy Y ∈ C;D, both from the same subpopulation k
    and from the other population l (that is, l = P if k = R, and l = R if k = P)
    :param k: rich or poor {'R','P'}
    :param X: cooperator or defector {'C','D'}
    :param u: mutation probability μ
    :param beta: β controls the intensity of selection
    :return: transition probabilities from a strategy X to Y
    '''
    if k == 'R'and X == 'C':
        #RC TO RD
        a = (Idr/(Zr-1 + (1-h)*Zp))*(1 + exp(beta * (fitness('R','C',Icr, Icp)-fitness('R','D',Icr, Icp))))**(-1)
        b = ((1-h)*Idp/(Zr-1 + (1-h)*Zp))*(1 + exp(beta * (fitness('R','C',Icr, Icp)-fitness('P','D',Icr, Icp))))**(-1)

        #RC to RC
        c = (Icr / (Zr - 1 + (1 - h) * Zp)) * (1 + exp(beta * (fitness('R', 'C', Icr, Icp) - fitness('R', 'C', Icr, Icp))))**(-1)
        d = ((1-h)*Icp/(Zr-1 + (1-h)*Zp))*(1 + exp(beta * (fitness('R','C',Icr, Icp)-fitness('P','C',Icr, Icp))))**(-1)

        return Icr/ Z * ((1-u)*(a+b)+u), Icr/ Z * ((1-u)*(c+d)+u)

    elif k == 'R'and X == 'D':
        #RD to RC
        a = (Icr / (Zr - 1 + (1 - h) * Zp)) * (1 + exp(beta * (fitness('R','D',Icr, Icp) - fitness('R','C',Icr, Icp))))**(-1)
        b = ((1 - h) * Icp / (Zr - 1 + (1 - h) * Zp)) * (1 + exp(beta * (fitness('R','D', Icr, Icp) - fitness('P', 'C',Icr, Icp))))**(-1)

        #RD to RD
        c = (Idr / (Zr - 1 + (1 - h) * Zp)) * (1 + exp(beta * (fitness('R', 'D', Icr, Icp) - fitness('R', 'D', Icr, Icp))))**(-1)
        d = ((1-h)*Idp/(Zr-1 + (1-h)*Zp))*(1 + exp(beta * (fitness('R','D',Icr, Icp)-fitness('P','D',Icr, Icp))))**(-1)
        return Idr / Z * ((1-u)*(a+b)+u), Idr/ Z * ((1-u)*(c+d)+u)

    elif k == 'P'and X == 'C':
        #PC to PD
        a = (Idp/(Zp-1 + (1-h)*Zr))*(1 + exp(beta * (fitness('P','C',Icr, Icp)-fitness('P','D',Icr, Icp))))**(-1)
        b = ((1-h)*Idr/(Zp-1 + (1-h)*Zr))*(1 + exp(beta * (fitness('P','C',Icr, Icp)-fitness('R','D',Icr, Icp))))**(-1)

        #PC to PC
        c = (Icp / (Zr - 1 + (1 - h) * Zp)) * (1 + exp(beta * (fitness('P', 'C', Icr, Icp)-fitness('P', 'C', Icr, Icp))))**(-1)
        d = ((1 - h) * Icr / (Zr - 1 + (1 - h) * Zp)) * (1 + exp(beta * (fitness('P', 'C', Icr, Icp) - fitness('R', 'C', Icr, Icp))))**(-1)

        return Icp / Z * ((1-u)*(a+b)+u),Icp / Z * ((1-u)*(c+d)+u)

    elif k == 'P'and X == 'D':
        #PD to PC
        a = (Icp/(Zp-1 + (1-h)*Zr))*(1 + exp(beta * (fitness('P','D',Icr, Icp)-fitness('P','C',Icr, Icp))))**(-1)
        b = ((1-h)*Icr/(Zp-1 + (1-h)*Zr))*(1 + exp(beta * (fitness('P','D',Icr, Icp)-fitness('R','C',Icr, Icp))))**(-1)

        #PD to PD
        c = (Idp / (Zr - 1 + (1 - h) * Zp)) * (1 + exp(beta * (fitness('P', 'D', Icr, Icp) - fitness('P', 'D', Icr, Icp)))) ** (-1)
        d = ((1 - h) * Idr / (Zr - 1 + (1 - h) * Zp)) * (1 + exp(beta * (fitness('P', 'D', Icr, Icp) - fitness('R', 'D', Icr, Icp)))) ** (-1)

        return Idp / Z * ((1-u)*(a+b)+u), Idp / Z * ((1-u)*(c+d)+u)


def simulate_markov_process(num_steps, Idr, Idp, Icr, Icp):
    """
    Simulates a Markov process over a two-dimensional space.
    """
    for i in range(num_steps):
        players = ['RD','PD','RC','PC']
        x = randint(0,3)
        player = players[x] #choose an aleatoire player
        k = player[0]
        X = player[1]
        probability = transition(k,X,Idr = Idr,Idp = Idp, Icr = Icr, Icp = Icp)
        if probability > 0.5:
            if k == 'R' and X == 'C':
                Icr = Icr - 1
                Idr = Idr + 1
            elif k == 'R' and X == 'D':
                Icr = Icr - 1
                Idr = Idr + 1
            elif k == 'P' and X == 'C':
                Icp = Icp - 1
                Idp = Idp + 1
            elif k == 'P' and X == 'D':
                Icp = Icp - 1
                Idp = Idp + 1
    return [Icr, Icp]

def GoS(Idr, Idp, Icr, Icp):
    Tpr = transition('R','D', Idr, Idp, Icr, Icp)[0]
    Tnr = transition('R', 'C', Idr, Idp, Icr, Icp)[0]
    Tpp = transition('P', 'D', Idr, Idp, Icr, Icp)[0]
    Tnp = transition('P', 'C', Idr, Idp, Icr, Icp)[0]
    return(Tpr-Tnr, Tpp-Tnp)


# Simulate the Markov process for 100 steps
#initial_state = [Icr, Icp]
#final_state = simulate_markov_process(100)

#print(f"Initial state: {initial_state}")
#print(f"Final state: {final_state}")


matrix = []
for Icr in range(0,Zr):
    for Icp in range(0,Zp):
        sub_list = []
        Idr = Zr - Icr
        Idp = Zp - Icp
        print('--------------',(Icr,Icp),'--------------')
        print(GoS(Idr, Idp, Icr, Icp))
        #sub_list.append(transition('R','C',Idr, Idp, Icr, Icp)[0])
        #sub_list.append(transition('R', 'D',Idr, Idp, Icr, Icp)[0])
        #sub_list.append(transition('P', 'C',Idr, Idp, Icr, Icp)[0])
        #sub_list.append(transition('P', 'D',Idr, Idp, Icr, Icp)[0])
        #matrix.append(sub_list)




        #print('transition prob. RC --> RD = ', transition('R','C',Idr, Idp, Icr, Icp))
        #print('transition prob. RD --> RC = ', transition('R', 'D',Idr, Idp, Icr, Icp))
        #print('transition prob. PC --> PD = ', transition('P', 'C',Idr, Idp, Icr, Icp))
        #print('transition prob. PD --> PC = ', transition('P', 'D',Idr, Idp, Icr, Icp))
        #print('--------------',(Icr,Icp),'--------------')