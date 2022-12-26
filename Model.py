""""
This file is used to code for the main model of the article: https://www.pnas.org/doi/full/10.1073/pnas.1323479111
Vasconcelos VV, Santos FC, Pacheco JM, Levin SA. 2014
Climate policies under wealth inequality. Proc. Natl Acad. Sci. USA 111, 2212-2216.
(doi:10.1073/pnas.1323479111) Crossref, PubMed, ISI, Google Scholar
"""

Z = 6 #Number of individuals
Zr = 3 #Number of rich individuals
Zp = Z - Zr #Number of poor individuals

br = 1 #Initial endowment of the rich
bp = 0.5 #Initial endowment of the poor
print(br > bp) #br must be superior to bp

strategy_1 = "C" #Cooperators
strategy_2 = "D" #Defectors

Cs_r = 0 #fraction of cooperators of the rich individuals
Cs_p = 1 #fraction of cooperators of the poor individuals
Ds_r = 0 #fraction of defectors of the rich individuals
Ds_p = 1 #fraction of defectors of the poor individuals

c = 0.25 #fraction of the endowment contributed by the cooperators (Cs) to help solve the group task (contributions)

lost_Cs_r = br*(1-c) #Lost of the rich Cs if next intermediate target is not meet
lost_Cs_p = bp*(1-c) #Lost of the poor Cs if next intermediate target is not meet
lost_Ds_r = br #Lost of the rich Ds if next intermediate target is not meet
lost_Ds_p = bp #Lost of the poor Ds if next intermediate target is not meet

c_tot = ((c * br) * Cs_r) + ((c * bp) * Cs_p) #Total amount of contributions

Mcb = 1 #Threshold  for the target to be met

b = br + bp #Average endowment of the population