""""
This file is used to code for the main model of the article: https://www.pnas.org/doi/full/10.1073/pnas.1323479111
Vasconcelos VV, Santos FC, Pacheco JM, Levin SA. 2014
Climate policies under wealth inequality. Proc. Natl Acad. Sci. USA 111, 2212-2216.
(doi:10.1073/pnas.1323479111) Crossref, PubMed, ISI, Google Scholar
"""

################## Global variables

#Population size
Z = 6 #Number of individuals
Zr = 3 #Number of rich individuals
Zp = Z - Zr #Number of poor individuals
N = 0 #Individuals are randomly sampled from the population and organize into groups of size N == a compléter

#Initial endowment
br = 1 #Initial endowment of the rich
bp = 0.5 #Initial endowment of the poor
print(br > bp) #br must be superior to bp

b = (br + bp) / 2 #Average endowment of the population

#Strategy
strategy_1 = "C" #Cooperators
strategy_2 = "D" #Defectors

#Population of Cs and Ds
Cs_r = 0 #fraction of cooperators of the rich individuals
Cs_p = 1 #fraction of cooperators of the poor individuals
Ds_r = 0 #fraction of defectors of the rich individuals
Ds_p = 1 #fraction of defectors of the poor individuals

Jr = Cs_r * Zr #Number of rich Cs
Jp = Cs_p * Zp #Number of poor Cs
Ds = Z - Jr - Jp #Number of defectors

#Contributions
c = 0.25 #fraction of the endowment contributed by the cooperators (Cs) to help solve the group task (contributions)
        #must be inferior to 1, b and br
cr = c*br #Contribution of the rich Cs
cp = c*bp #Contribution of the poor Cs
c_tot = ((c * br) * Cs_r) + ((c * bp) * Cs_p) #Total amount of contributions

#Lost if target not meet
lost_Cs_r = br*(1-c) #Lost of the rich Cs if next intermediate target is not meet
lost_Cs_p = bp*(1-c) #Lost of the poor Cs if next intermediate target is not meet
lost_Ds_r = br #Lost of the rich Ds if next intermediate target is not meet
lost_Ds_p = bp #Lost of the poor Ds if next intermediate target is not meet


Mcb = 1 #Threshold  for the target to be met

r = 0 #Perception of risk (varying between 0 and 1)

h = 0 #Homophily parameter (varying between 0 and 1)
        #When h = 1, individuals are restricted to influence by those of the same wealth status
        #When h = 0, no wealth discrimination takes place



############## Functions

def O_(k):
    res = 0
    if k >=0:
        res = 1
    return res


############## Main Code

# Calculating payoffs

delta = cr*Jr + cp*Jp - Mcb

P_D_R = br*(O_(delta)+(1-r)*(1-O_(delta))) #Payoff for a rich Ds
P_D_P = bp*(O_(delta)+(1-r)*(1-O_(delta))) #Payoff for a poor Ds
P_C_R = P_D_R - cr #Payoff for a rich Cs    === il y a une erreur dans le MM mais c'est corrigé dans S1 text
P_C_P = P_D_P - cp #Payoff for a poor Cs

# Calculating the average payoff of a given strategy

ir = 0
ip = 0
i = [ir,ip] #given configuration of strategies and wealth classes == à compléter

F_C_R = 0
F_D_R = 0
F_C_P = 0    ### A compléter
F_D_P = 0


# Evolution in time of the number of individuals adopting a given strategy
#==== gain-loss equation involving the transition rates between all accessible configurations


# Stationary distribution
#==== obtained by reducing the master equation to an eigenvector search problem


#Compute the most likely path the population will follow, resorting the probability to increase (decrease) the number of individuals adopting a strategy


#Compute the fraction of group that reach a total Mcb in contributions== successfully achieve the public good (ag(i))


#Compute the average group achievment (ng) = averaging over all possibles configurations i, each weighted with the corresponding stationary distribution

