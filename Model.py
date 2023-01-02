'''
This file is used to code for the main model of the article: https://www.pnas.org/doi/full/10.1073/pnas.1323479111
Vasconcelos VV, Santos FC, Pacheco JM, Levin SA. 2014
Climate policies under wealth inequality. Proc. Natl Acad. Sci. USA 111, 2212-2216.
(doi:10.1073/pnas.1323479111) Crossref, PubMed, ISI, Google Scholar
'''

from math import factorial as fact
from math import exp
from random import randint

################## Global variables

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
F_cr = 0.5 #fraction of cooperators of the rich individuals
F_cp = 0.5 #fraction of cooperators of the poor individuals

Ir = int(F_cr * Zr) #nbr rich Cs in the population
Ip = int(F_cp * Zp) #nrb of poor Cs in the population
Icr = int(F_cr * Zr) #nbr rich Cs in the population
Icp = int(F_cp * Zp) #nrb of poor Cs in the population
Idr = int((1-F_cr) * Zr) #nbr rich Ds in the population
Idp = int((1-F_cp)*Zp) #nbr poor Ds in the population

Jr = int(F_cr * Nr) #nbr rich Cs in each group
Jp = int(F_cp * Np) #nbr rich Cs in each group

Ds = N - Jr - Jp #Number of defectors in each group
Cs = N - Ds #Number of cooperators in each group

#Contributions
c = 0.25 #fraction of the endowment contributed by Cs to help solve the group task

Cr = c*br #Contribution of the rich Cs
Cp = c*bp #Contribution of the poor Cs
c_tot = ((c * br) * Jr) + ((c * bp) * Jp) #Total amount of contributions Iin each group

#Lost if target not meet
lost_Cr = br*(1-c) #Lost of the rich Cs if next intermediate target is not meet
lost_Cp = bp*(1-c) #Lost of the poor Cs if next intermediate target is not meet
lost_Dr = br #Lost of the rich Ds if next intermediate target is not meet
lost_Dp = bp #Lost of the poor Ds if next intermediate target is not meet

M = 5 #positive integer between O and N
Mcb = M*c*b_hat #Threshold  for the target to be met

r = 0 #Perception of risk (varying between 0 and 1)

h = 0 #Homophily parameter (varying between 0 and 1)
        #When h = 1, individuals are restricted to influence by those of the same wealth status
        #When h = 0, no wealth discrimination takes place

############## Functions#############

def O_(k):
    res = 0
    if k >=0:
        res = 1
    return res

def payoff_DR(Jr=Jr,Jp=Jp,Cr=Cr, Cp=Cp, Mcb=Mcb,br=br):
    ''' Payoff for rich defector player'''
    delta = Cr * Jr + Cp * Jp - Mcb
    return br * (O_(delta) + (1 - r) * (1 - O_(delta)))

def payoff_DP(Jr=Jr,Jp=Jp,Cr=Cr, Cp=Cp, Mcb=Mcb,bp=bp):
    ''' Payoff for poor defector player'''
    delta = Cr * Jr + Cp * Jp - Mcb
    return bp * (O_(delta) + (1 - r) * (1 - O_(delta)))

def binom(n,r):
    '''binomila product'''
    return fact(n)//fact(r)//fact(n-r)


def fitness(wealth, strat, ir=Ir,ip=Ip,Cr=Cr,Z=Z,N=N):
    '''fitness for rich cooperator'''
    res = 0
    if wealth == 'R' and strat == 'C':
        for jr in range(0,(N)):
            for jp in range (0,(N-jr)):
                P_C_R = payoff_DR(Jr=jr+1, Jp=jp) - Cr
                res += binom(ir - 1, jr) * binom(ip, jp) * binom(Z-ir-ip, N-1-jr-jp) * P_C_R
        return res * (binom(Z-1,N-1))**(-1)
    elif wealth == 'P' and strat == 'C':
        for jr in range(0, (N)):
            for jp in range(0, (N - jr)):
                P_C_P = payoff_DP(Jr=jr, Jp=jp + 1) - Cp
                res += binom(ir, jr) * binom(ip - 1, jp) * binom(Z - ir - ip, N - 1 - jr - jp) * P_C_P
        return res * (binom(Z - 1, N - 1)) ** (-1)
    elif wealth == 'R' and strat == 'D':
        for jr in range(0, (N)):
            for jp in range(0, (N - jr)):
                P_D_R = payoff_DR(Jr=jr, Jp=jp)
                res += binom(ir, jr) * binom(ip, jp) * binom(Z - 1 - ir - ip, N - 1 - jr - jp) * P_D_R
        return res * (binom(Z - 1, N - 1)) ** (-1)
    elif wealth == 'P' and strat == 'D':
        for jr in range(0, (N)):
            for jp in range(0, (N - jr)):
                P_D_P = payoff_DP(Jr=jr, Jp=jp)
                res += binom(ir, jr) * binom(ip, jp) * binom(Z - 1 - ir - ip, N - 1 - jr - jp) * P_D_P
        return res * (binom(Z - 1, N - 1)) ** (-1)


def fermi(beta, strategy):
    '''
    Under pairwise comparison, each individual
of strategy X adopts the strategy Y of a randomly selected
member of the population, with probability given by the Fermi
function
    :param beta: control the intensity of the selection
    :param strategy: rich Cs (RC), poor Cs (PC), rich Ds (RD), or poor Ds (PD)
    :return: probability
    '''
    strat_pop = ['PC','PD','RC','RD']
    strat = strat_pop[randint(0,4)]

    if strategy == 'PC':
        res = 1 + exp(beta * (fitness(wealth='P',strat='C')-fitness(strat[0],strat[1])))**(-1)
    elif strategy == 'PD':
        res = 1 + exp(beta * (fitness(wealth='P',strat='D')-fitness(strat[0],strat[1])))**(-1)

    elif strategy == 'RC':
        res = 1 + exp(beta * (fitness(wealth='R',strat='C')-fitness(strat[0],strat[1])))**(-1)

    elif strategy == 'RD':
        res = 1 + exp(beta * (fitness(wealth='R',strat='D')-fitness(strat[0],strat[1])))**(-1)

    return res

def transition(k,X,u=0.5,beta=3):
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
        a = (Idr/(Zr-1 + (1-h)*Zp))*(1 + exp(beta * (fitness(wealth='R',strat='C')-fitness(wealth='R',strat='D'))**(-1)))
        b = ((1-h)*Idp/(Zr-1 + (1-h)*Zp))*(1 + exp(beta * (fitness(wealth='R',strat='C')-fitness(wealth='P',strat='D'))**(-1)))
        return Icr/ Z * ((1-u)*(a+b)+u)

    elif k == 'R'and X == 'D':
        a = (Icr / (Zr - 1 + (1 - h) * Zp)) * (1 + exp(beta * (fitness(wealth='R', strat='D') - fitness(wealth='R', strat='C')) ** (-1)))
        b = ((1 - h) * Icp / (Zr - 1 + (1 - h) * Zp)) * (1 + exp(beta * (fitness(wealth='R', strat='D') - fitness(wealth='P', strat='C')) ** (-1)))
        return Idr / Z * ((1-u)*(a+b)+u)

    elif k == 'P'and X == 'C':
        a = (Idp/(Zp-1 + (1-h)*Zr))*(1 + exp(beta * (fitness(wealth='P',strat='C')-fitness(wealth='P',strat='D'))**(-1)))
        b = ((1-h)*Icr/(Zp-1 + (1-h)*Zr))*(1 + exp(beta * (fitness(wealth='P',strat='C')-fitness(wealth='R',strat='D'))**(-1)))
        return Icp / Z * ((1-u)*(a+b)+u)

    elif k == 'P'and X == 'D':
        a = (Icp/(Zp-1 + (1-h)*Zr))*(1 + exp(beta * (fitness(wealth='P',strat='D')-fitness(wealth='P',strat='C'))**(-1)))
        b = ((1-h)*Icr/(Zp-1 + (1-h)*Zr))*(1 + exp(beta * (fitness(wealth='P',strat='D')-fitness(wealth='R',strat='C'))**(-1)))
        return Idp / Z * ((1-u)*(a+b)+u)




############## Main Code ##############

# Calculating payoffs

delta = Cr*Jr + Cp*Jp - Mcb

P_D_R = payoff_DR() #Payoff for a rich Ds   ### Sûrement à passer en fonction
P_D_P = payoff_DP() #Payoff for a poor Ds
P_C_R = P_D_R - Cr #Payoff for a rich Cs
P_C_P = P_D_P - Cp #Payoff for a poor Cs

print('Payoff = ')
print('FDR =',P_D_R)
print('FDR =',P_D_P)
print('FCR =',P_C_R)
print('FCP =',P_C_P)

# Calculating the average payoff of a given strategy

F_C_R = fitness(wealth='R', strat='C')
F_D_R = fitness(wealth='R', strat='D')
F_C_P = fitness(wealth='P', strat='C')
F_D_P = fitness(wealth='P', strat='D')


print('Fitness = ')
print('FCR =',F_C_R)
print('FDR =',F_D_R)
print('FCP =',F_C_P)
print('FDP =',F_D_P)

# Evolution in time of the number of individuals adopting a given strategy
#==== gain-loss equation involving the transition rates between all accessible configurations


# Stationary distribution
#==== obtained by reducing the master equation to an eigenvector search problem


#Compute the most likely path the population will follow, resorting the probability to increase (decrease) the number of individuals adopting a strategy


#Compute the fraction of group that reach a total Mcb in contributions== successfully achieve the public good (ag(i))


#Compute the average group achievment (ng) = averaging over all possibles configurations i, each weighted with the corresponding stationary distribution

