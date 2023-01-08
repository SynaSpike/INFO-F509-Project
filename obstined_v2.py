# Try to replicate some of Vasconcelos et al 2014
# NOTE to self: the reason the p values sometimes don't line up with the gradient field is because the corners and edges are sticky
# homophily makes the edges sticky

from scipy.special import comb as comb
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib


############################### Definitions #################################

# indicator function for whether or not the group has met or exceeded the threshold
# a function of the number of rich cooperators jR and poor cooperators jP in the group
def Theta_fnc (jR, jP,Mcb):
    return 1 if Cr * jR + Cp * jP - Mcb >= 0 else 0
# payoffs
def PiDR (jR, jP,r,Mcb):
    return br * (Theta_fnc(jR, jP,Mcb) + (1 - r) * (1 - Theta_fnc(jR, jP,Mcb)))  # rich defectors
def PiDP (jR, jP,r,Mcb):
    return bp * (Theta_fnc(jR, jP,Mcb) + (1 - r) * (1 - Theta_fnc(jR, jP,Mcb)))  # poor defectors
def PiCR (jR, jP,r,Mcb):
    return br * (Theta_fnc(jR, jP,Mcb) + (1 - r) * (1 - Theta_fnc(jR, jP,Mcb))) - Cr  # rich cooperators
def PiCP (jR, jP,r,Mcb):
    return bp * (Theta_fnc(jR, jP,Mcb) + (1 - r) * (1 - Theta_fnc(jR, jP,Mcb))) - Cp  # poor cooperators

# fitness expected values are a function of the composition of the population (rich and poor cooperators)

# rich cooperators
def fCR (iR, iP,r,Mcb):
    return (1/comb(Z-1, N-1)) * sum( sum(comb(iR-1, jR) * comb(iP, jP) * comb(Z-iR-iP, N-1-jR-jP) * PiCR(jR+1, jP,r,Mcb)
            for jP in range(N-jR) ) for jR in range(N))

# poor cooperators
def fCP (iR, iP,r,Mcb):
    return(1/comb(Z-1, N-1)) * sum(sum(comb(iR, jR) * comb(iP-1, jP) * comb(Z-iR-iP, N-1-jR-jP) * PiCP(jR, jP+1,r,Mcb)
            for jP in range(N-jR) ) for jR in range(N) )

# rich defectors
def fDR (iR, iP,r,Mcb):
    return (1/comb(Z-1, N-1)) * sum( sum(comb(iR, jR) * comb(iP, jP) * comb(Z-1-iR-iP, N-1-jR-jP) * PiDR(jR, jP,r,Mcb)
            for jP in range(N-jR) ) for jR in range(N) )

# poor defectors
def fDP (iR, iP,r,Mcb):
    return (1/comb(Z-1, N-1)) * sum( sum(comb(iR, jR) * comb(iP, jP) * comb(Z-1-iR-iP, N-1-jR-jP) * PiDP(jR, jP,r,Mcb)
            for jP in range(N-jR) ) for jR in range(N) )

def fXk(iR, iP, X, k,r,Mcb):
    if X == 'C' and k == 'R':
        res = fCR(iR, iP,r,Mcb)
    elif X == 'C' and k == 'P':
        res = fCP(iR, iP,r,Mcb)
    elif X == 'D' and k == 'R':
        res = fDR(iR, iP,r,Mcb)
    elif X == 'D' and k == 'P':
        res = fDP(iR, iP,r,Mcb)
    else:
        res = None

    return (res)


# Fermi function
# chance that the candidate will imitate the other's strategy is proportional to how much higher the other's payoff is compared to their own
def Fe (iR, iP, X1, k1, X2, k2,r,beta,Mcb):
    return 1 + np.exp(beta * (fXk(iR, iP, X1, k1,r,Mcb) - fXk(iR, iP, X2, k2,r,Mcb)))


def T(iR, iP, X, k,r,h,mu,beta,Mcb):
    '''
    Probability of transition of a k-wealth individual (k in {R, P}) from strategy
    X (X in {C, D}) to Y (opposite of X)
    '''

    Y = 'C' if X == 'D' else 'D'
    l = 'R' if k == 'P' else 'P'

    Zk = Zr if k == 'R' else Zp
    Zl = Zr if l == 'R' else Zp

    ik = iR if k == 'R' else iP
    iXk = ik if X == 'C' else Zk - ik

    il = iR if l == 'R' else iP
    iYl = il if Y == 'C' else Zl - il
    iYk = ik if Y == 'C' else Zk - ik

    TXYk = (iXk / Z) * (mu + (1 - mu) * (iYk / ((Zk - 1 + (1 - h) * Zl) * Fe(iR, iP, X, k, Y, k,r,beta,Mcb)) + (1 - h) * iYl / ((Zk - 1 + (1 - h) * Zl) * Fe(iR, iP, X, k, Y, l,r,beta,Mcb))))

    return (TXYk)


def transition(iR,iP,Zr,Zp,iR_obs, iP_obs,r,h,mu,beta,Mcb):

    if iR < Zr:
        TiR_gain = T(iR, iP, 'D', 'R',r,h,mu,beta,Mcb)
    else:
        TiR_gain = 0  # not possible to have more rich cooperators than rich individuals in the population

    if iR > iR_obs:
        TiR_loss = T(iR, iP, 'C', 'R',r,h,mu,beta,Mcb)

    else:
        TiR_loss = 0

    if iP < Zp:
        TiP_gain = T(iR, iP, 'D', 'P',r,h,mu,beta,Mcb)
    else:
        TiP_gain = 0

    if iP > iP_obs:
        TiP_loss = T(iR, iP, 'C', 'P',r,h,mu,beta,Mcb)
    else:
        TiP_loss = 0

    return[TiR_gain, TiR_loss, TiP_gain, TiP_loss]


def iV(Zr,Zp, iR_obs, iP_obs):
    iV = [(iR, iP) for iP in range(iP_obs,Zp + 1) for iR in range(iR_obs,Zr + 1)]  # list of states
    return iV

def transition_matrix(iV,Zr,Zp,iR_obs,iP_obs,r,h,mu,beta,Mcb):

    #transition probability matrix between population states (iR, iP) -> (iR', iP')

    i2idx = {i: idx for idx, i in enumerate(iV)}  # reverse dictionary from state to index of W below
    print(i2idx)
    len_iV = len(iV)
    W = np.zeros((len_iV, len_iV))

    for idx, i in enumerate(iV):

        iR, iP = i  # number of rich cooperators, number of poor cooperators
        trans = transition(iR, iP, Zr, Zp,iR_obs,iP_obs,r, h, mu, beta, Mcb)
        # calculate probabilities of each transition and population the matrix W
        if iR < Zr:
            a = trans[0]
            W[i2idx[(iR + 1, iP)], idx] = a
        else:
            a = 0

        if iR > iR_obs:
            b = trans[1]
            W[i2idx[(iR - 1, iP)], idx] = b
        else:
            b = 0

        if iP < Zp:
            c = trans[2]
            W[i2idx[(iR, iP + 1)], idx] = c
        else:
            c= 0

        if iP > iP_obs:
            d = trans[3]
            W[i2idx[(iR, iP - 1)], idx] = d
        else:
            d= 0
        # probability that the population state does not transition from (iR, iP)
        W[(idx, idx)] = 1 - a - b - c -d

    return W


def gradient(iV,iR_obs,iP_obs,r,h,mu,beta,Mcb):
    # transition probability matrix between population states (iR, iP) -> (iR', iP')
    # gradient of selection on rich and poor axis
    grad_iR = np.zeros((Zp + 1, Zr + 1))  # rich on the x-axis
    grad_iP = np.zeros((Zp + 1, Zr + 1))

    for idx, i in enumerate(iV):
        iR, iP = i  # number of rich cooperators, number of poor cooperators

        TiR_gain = transition(iR,iP,Zr,Zp,iR_obs,iP_obs,r,h,mu,beta,Mcb)[0]
        TiR_loss = transition(iR,iP,Zr,Zp,iR_obs,iP_obs,r,h,mu,beta,Mcb)[1]
        TiP_gain = transition(iR,iP,Zr,Zp,iR_obs,iP_obs,r,h,mu,beta,Mcb)[2]
        TiP_loss = transition(iR,iP,Zr,Zp,iR_obs,iP_obs,r,h,mu,beta,Mcb)[3]

        if iR < iR_obs or iP < iP_obs:
            grad_iR[iP, iR] = 0
            grad_iP[iP, iR] = 0
        else:
            # gradients used for a quiver plot below
            grad_iR[iP, iR] = TiR_gain - TiR_loss
            grad_iP[iP, iR] = TiP_gain - TiP_loss

    return (grad_iR,grad_iP)


def p(W):
    # get relative proportions of time spent in different states
    # ---
    eigenvalues, eigenvectors = np.linalg.eig(W)
    domIdx = np.argmax(np.real(eigenvalues)) # index of the dominant eigenvalue
    p= np.real(eigenvectors[:, domIdx])
    p = p / np.sum(p)  # normalise
    return p

def P(p,iV):
    P = np.zeros((Zp + 1, Zr + 1))  # rich on the x-axis
    for idx, pi in enumerate(p):
        iR, iP = iV[idx]
        P[iP, iR] = pi
    return P

def aG(iR,iP, Z, N,Mcb):
    return(sum(comb(iR,jR)*comb(iP,jP)*comb(Z-iR-iP, N-jR-jP)*Theta_fnc(jR,jP,Mcb)
               for jR in range(N + 1) for jP in range(N + 1 - jR))/comb(Z,N))



def nG(p,iV,Mcb):
    nG = 0
    for index, P_bar_i in enumerate(p):
        iR,iP = iV[index]
        nG += P_bar_i*aG(iR,iP,Z,N,Mcb)
    return nG

def graph(suffix, Zr, Zp, iV,P,r,h,mu,beta,Mcb, nG):
    ###### plot ######
    # ---

    fig, ax = plt.subplots(figsize=(3, 6))

    iRV = list(range(Zr + 1))
    iPV = list(range(Zp + 1))

    # ax.pcolor(iRV, iPV, P, cmap='coolwarm_r', alpha=0.5) #, vmin=0, vmax=1)
    # im = ax.imshow(P, extent=(0-0.5, ZR+0.5, 0-0.5, ZP+0.5), origin='lower', cmap='coolwarm_r') #, alpha=0.5) #, vmin=0, vmax=1)
    # im = ax.imshow(P, extent=(0-0.5, ZR+0.5, 0-0.5, ZP+0.5), origin='lower', cmap='coolwarm_r')
    im = ax.imshow(P, origin='lower', cmap='coolwarm_r', alpha=0.5)
    # fig.colorbar(im)
    ax.quiver(iRV, iPV, gradient(iV,iR_obs,iP_obs,r,h,mu,beta,Mcb)[0], gradient(iV,iR_obs,iP_obs,r,h,mu,beta,Mcb)[1])

    # ax.quiver(iRV, iPV, np.zeros( (ZP+1, ZR+1) ), grad_iP)
    # ax.quiver(iRV, iPV, grad_iR, np.zeros( (ZP+1, ZR+1) ))

    # ax.quiver(iRV, iPV, stay_iR, np.zeros( (ZP+1, ZR+1) ))
    # ax.quiver(iRV, iPV, np.zeros( (ZP+1, ZR+1) ), stay_iP)

    # ax.quiver(iRV, iPV, go_iR, np.zeros( (ZP+1, ZR+1) ))
    # ax.quiver(iRV, iPV, np.zeros( (ZP+1, ZR+1) ), go_iP)

    ax.set_xlim((-1, Zr + 1))
    ax.set_ylim((-1, Zp + 1))
    # ax.set_aspect('scaled')
    ax.set_xlabel(r'rich cooperators, $i_R$')
    ax.set_ylabel(r'poor cooperators, $i_P$')
    plt.axis('scaled')
    plt.tight_layout()
    plt.title("nG = "+ str(round(nG,2))+' h = '+str(h))

    #save
    strFile = suffix + '.pdf'
    if os.path.isfile(strFile):
       os.remove(strFile)   # Opt.: os.system("rm "+strFile)
    plt.savefig(strFile)
    plt.savefig(suffix + '.pdf')
    plt.close()

def axes(Zr,Zp,h,mu,beta):
    combination = iV(Zr,Zp)
    risk_list = [r * 0.10 for r in range(0, 11)]
    nG_list = []
    for r in risk_list:
        W = transition_matrix(combination, Zr,Zp,iR_obs,iP_obs,r,h,mu,beta,Mcb)
        pi = p(W)
        nG_list.append(nG(pi,combination,Mcb))
    return(risk_list,nG_list)

def plot_fig1(suffix,mu,beta):
    ax1 = axes(8,32,0,mu,beta)
    x1 = ax1[0]
    y1 = ax1[1]
    label1 = '- with inequality & h = 0'
    ax2 = axes(8, 32, 1, mu, beta)
    x2 = ax2[0]
    y2 = ax2[1]
    label2 = '- with inequality & h = 1'
    ax3 = axes(20, 20, 0, mu, beta)
    x3 = ax3[0]
    y3 = ax3[1]
    label3 = '- without inequality'

    plt.plot(x1,y1,label=label1)
    plt.plot(x2,y2,label=label2)
    plt.plot(x3, y3, label=label3)
    plt.legend(loc='best')

    # save
    strFile = suffix + '.pdf'
    if os.path.isfile(strFile):
        os.remove(strFile)  # Opt.: os.system("rm "+strFile)
    plt.savefig(strFile)
    plt.savefig(suffix + '.pdf')
    plt.close()


def new_plot(suffix, Zr, Zp, iV,grad,P,h, nG):

    fig, ax = plt.subplots(figsize=(3, 6))

    x = []
    for ip in range(Zp + 1):
        for ir in range(Zr + 1):
            x.append(ir)
    y = []
    for ip in range(Zp+ 1):
        for ir in range(Zr + 1):
            y.append(ip)

    customcmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["#DCDCDC", "black"])
    plt.scatter(x, y, c=P, alpha=0.85, cmap=customcmap, edgecolors="#A9A9A9")

    iRV = list(range(Zr + 1))
    iPV = list(range(Zp + 1))

    colors = np.zeros((Zp + 1, Zr + 1))
    grad_iR = grad[0]
    grad_iP = grad[1]

    customcolormap2 = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["#610484", "#5a2293",
                                                                                     "#5340a1", "#366695", "#128f81",
                                                                                     "#22a967", "#67b448", "#9fae31",
                                                                                     "#b17630", "#c33d30"])
    ax.streamplot(iRV, iPV, grad_iR, grad_iP, color=colors, density=.5, cmap=customcolormap2)

    ax.set_xlim((-1, Zr + 1))
    ax.set_ylim((-1, Zp + 1))
    ax.set_xlabel(r'rich cooperators, $i_R$')
    ax.set_ylabel(r'poor cooperators, $i_P$')
    plt.axis('scaled')
    plt.tight_layout()
    plt.title("nG = "+ str(round(nG,3))+' h = '+str(h))

    # save
    strFile = suffix + '.pdf'
    if os.path.isfile(strFile):
        os.remove(strFile)  # Opt.: os.system("rm "+strFile)
    plt.savefig(strFile)
    plt.savefig(suffix + '.pdf')
    plt.close()

###################### Parameters ############################

Z = 200 #Number of individuals in the population
Zr = 40 #Number of rich individuals in the population
Zp = Z - Zr #Number of poor individuals in the population
N = 6 #groups of size N

#Initial endowment (br > bp)
br = 2.5 #Initial endowment of the rich
bp = 0.625 #Initial endowment of the poor
b_hat = (br*Zr + bp*Zp) / Z #Average endowment of the population

#Contributions
c = 0.1 #fraction of the endowment contributed by Cs to help solve the group task

Cr = c*br #Contribution of the rich Cs
Cp = c*bp #Contribution of the poor Cs

M = 3 #positive integer between O and N
Mcb = M*c*b_hat #Threshold  for the target to be met

r = 0.2 #Perception of risk (varying between 0 and 1)
h = 1 #Homophily parameter (varying between 0 and 1)
        #When h = 1, individuals are restricted to influence by those of the same wealth status
        #When h = 0, no wealth discrimination takes place
mu = (1/Z) #if mu = 1 --> individu does not change to strategy
beta = 5

iR_obs = 0
iP_obs = round(Zp/10)

######################### Main code #########################

iV = iV(Zr,Zp,iR_obs,iP_obs)
print(iV)
W = transition_matrix(iV,Zr,Zp,iR_obs,iP_obs,r,h,mu,beta,Mcb)
grad = gradient(iV, iR_obs,iP_obs,r,h,mu,beta,Mcb)
p = p(W)
nG = nG(p,iV,Mcb)
P = P(p, iV)
new_plot('test_new_plot_obst',Zr,Zp,iV,grad,P,h,nG)


