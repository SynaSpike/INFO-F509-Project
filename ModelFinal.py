import matplotlib.pyplot as plt
from scipy.linalg import eig as eig
from scipy.special import comb as comb
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm


class ClimateGame:


    def __init__(self, popuplation_size: int, group_size: int, nb_rich: int, nb_poor:int,
                 fraction_endowment: float, homophily:float, risk:float, M:float, rich_end:int, poor_end:int, mu:float) -> None:

        self.population_size = popuplation_size  # Z
        self.group_size = group_size  # N
        self.rich = nb_rich  # Zr
        self.rich_end = rich_end # br
        self.poor = nb_poor  # Zp
        self.poor_end = poor_end # bp
        self.strategies = ["C", "D"] # Cooperator or Defector
        self.profiles = ["R", "P"]  # Rich or Poor
        self.fraction_endowment = fraction_endowment  # C
        self.homophily = homophily  # h
        self.risk = risk  # r
        self.treshold = M * fraction_endowment * ((((poor_end * self.poor) + (rich_end * self.rich)) /population_size))
        self.mu = mu # mutation



    def play(self, group_composition):

        """
        Calculates the payoff of each strategy/profile inside the group.

        Parameters: array giving the number of [Rich Coop, Rich Defect, Poor Coop, Poor  Defect]
        ----------
        group_composition: Union[List[int], numpy.ndarray]
            counts of each strategy inside the group.
        game_payoffs: numpy.ndarray
            container for the payoffs of each strategy
        """

        Cr = self.rich_end * self.fraction_endowment # Rich cooperation Cr = c * br
        Cp = self.poor_end * self.fraction_endowment # Poor Cooperation Cp = c * bp

        k = Cr * group_composition[0] + Cp * group_composition[2] - self.treshold # Cr Jr + Cp Jp - Mcb

        if k >= 0:
            heav = 1

        else:
            heav = 0

        payoffs = np.zeros(4) # [0, 0, 0, 0]

        payoffs[1] = self.rich_end * (heav + (1 - self.risk) * (1 - heav))
        payoffs[3] = self.poor_end * (heav + (1 - self.risk) * (1 - heav))

        payoffs[0] = payoffs[1] - Cr # Rich coop payoff = Rich Defector payoff - contribution
        payoffs[2] = payoffs[3] - Cp # Rich coop payoff = Poor Defector payoff - contribution

        return payoffs

    def calculate_fitness(self, ir, ip) -> float:

        # Rich Cooperator Fitness
        sum_1 = 0

        for jr in range(self.group_size):
            sum_2 = 0
            for jp in range(self.group_size-jr):
                payoff = self.play([jr + 1, 0, jp, 0])[0]
                # Do not care about the nbr of defector (does not affect payoff)
                sum_2 += comb(ir - 1, jr) * comb(ip, jp) * comb(self.population_size - ir - ip,
                                                                self.group_size - 1 - jr - jp) * payoff
            sum_1 += sum_2

        RC = comb(self.population_size - 1, self.group_size - 1) ** (-1) * sum_1

        # Rich Defector Fitness
        sum_1 = 0

        for jr in range(self.group_size):
            sum_2 = 0
            for jp in range(self.group_size-jr):
                payoff = self.play([jr, 0, jp, 0])[1]
                # Do not care about the nbr of defector (does not affect payoff)
                sum_2 += comb(ir, jr) * comb(ip, jp) * comb(self.population_size - 1 - ir - ip,
                                                            self.group_size - 1 - jr - jp) * payoff
            sum_1 += sum_2

        RD = comb(self.population_size - 1, self.group_size - 1) ** (-1) * sum_1

        # Poor Cooperator Fitness
        sum_1 = 0

        for jr in range(self.group_size):
            sum_2 = 0
            for jp in range(self.group_size-jr):
                payoff = self.play([jr, 0, jp + 1, 0])[2]
                # Do not care about the nbr of defector (does not affect payoff)
                sum_2 += comb(ir, jr) * comb(ip - 1, jp) * comb(self.population_size - ir - ip,
                                                                self.group_size - 1 - jr - jp) * payoff
            sum_1 += sum_2

        PC = comb(self.population_size - 1, self.group_size - 1) ** (-1) * sum_1

        # Poor Defector Fitness
        sum_1 = 0

        for jr in range(self.group_size):
            sum_2 = 0
            for jp in range(self.group_size-jr):
                payoff = self.play([jr, 0, jp, 0])[3]
                # Do not care about the nbr of defector (does not affect payoff)
                sum_2 += comb(ir, jr) * comb(ip, jp) * comb(self.population_size - 1 - ir - ip,
                                                            self.group_size - 1 - jr - jp) * payoff
            sum_1 += sum_2

        PD = comb(self.population_size - 1, self.group_size - 1) ** (-1) * sum_1


        return [RC, RD, PC, PD]


    def T(self, ir:int, ip:int, k:str, X:str):
        """
        Calculate the Transition Probability a k-wealth individual (k in {R, P}) from strategy
        X (X in {C, D}) to Y (opposite of X)
        :return:
        """
        fit = self.calculate_fitness(ir, ip)  # Dp Dr Cp Cr
        Dp = self.poor - ip  # Nbr of poor defector
        Dr = self.rich - ir  # Nbr of rich defector
        beta = 3.
        mu = 1.4 * 1 / self.population_size

        # Transition Cp -> Dp
        if k == "P" and X == "C":
            fermi_1 = (1 + np.exp(beta * (fit[2] - fit[3]))) ** -1  # Cp -> Dp
            fermi_2 = (1 + np.exp((beta * (fit[2] - fit[1])))) ** -1  # Cp -> Dr
            param1 = (Dp / (self.poor - 1 + (1 - self.homophily) * self.rich))
            param2 = ((1 - self.homophily) * Dr) / (self.poor - 1 + (1 - self.homophily) * self.rich)
            return (ip / self.population_size) * ((1 - mu) * (param1 * fermi_1 + param2 * fermi_2) + mu)

        # Transition Cr -> Dr
        if k == "R" and X == "C":
            fermi_1 = (1 + np.exp(beta * (fit[0] - fit[1]))) ** -1  # Cr -> Dr
            fermi_2 = (1 + np.exp((beta * (fit[0] - fit[3])))) ** -1  # Cr -> Dp
            param1 = (Dr / (self.rich - 1 + (1 - self.homophily) * self.poor))
            param2 = ((1 - self.homophily) * Dp) / (self.rich - 1 + (1 - self.homophily) * self.poor)
            return (ir / self.population_size) * ((1 - mu) * (param1 * fermi_1 + param2 * fermi_2) + mu)

        # Transition Dp -> Cp
        if k == "P" and X == "D":
            fermi_1 = (1 + np.exp(beta * (fit[3] - fit[2]))) ** -1  # Dp -> Cp
            fermi_2 = (1 + np.exp((beta * (fit[3] - fit[0])))) ** -1  # Dp -> Cr
            param1 = (ip / (self.poor - 1 + (1 - self.homophily) * self.rich))
            param2 = ((1 - self.homophily) * ir) / (self.poor - 1 + (1 - self.homophily) * self.rich)
            return (Dp / self.population_size) * ((1 - mu) * (param1 * fermi_1 + param2 * fermi_2) + mu)

        # Transition Dr -> Cr
        if k == "R" and X == "D":
            fermi_1 = (1 + np.exp(beta * (fit[1] - fit[0]))) ** -1  # Dr -> Cr
            fermi_2 = (1 + np.exp((beta * (fit[1] - fit[2])))) ** -1  # Dr -> Cp
            param1 = (ir / (self.rich - 1 + (1 - self.homophily) * self.poor))
            param2 = ((1 - self.homophily) * ip) / (self.rich - 1 + (1 - self.homophily) * self.poor)
            return (Dr / self.population_size) * ((1 - mu) * (param1 * fermi_1 + param2 * fermi_2) + mu)


    def Enum_config(self, ir, ip):

        i_conf = [(ir, ip) for ip in range(self.poor + 1) for ir in range(self.rich+1)]

        dico_conf = { i: idx for idx, i in enumerate(i_conf) }

        return dico_conf, i_conf

    def Generate_W_GoS(self, dico:dict, i_conf:list):
        """
        :param dico: Dico generated by Enum_config
        :param i_conf: List of all possible population conf (ir, ip)
        :return:
        """

        W = np.zeros(shape=(len(i_conf), len(i_conf)))

        grad_ir = np.zeros((self.poor + 1, self.rich + 1))  # rich on the x-axis
        grad_ip = np.zeros((self.poor + 1, self.rich+ 1))

        for idx, i in enumerate(i_conf):

            ir, ip = i

            if ir < self.rich:
                Tr_gain = self.T(ir, ip, "R", "D") # Transition from one rich defector toward one rich cooperator
                W[dico[(ir + 1, ip)], idx] = Tr_gain

            else:
                Tr_gain = 0

            if ir > 0:
                Tr_loss = self.T(ir, ip, "R", "C")
                W[dico[(ir - 1 , ip)], idx] = Tr_loss

            else:
                Tr_loss = 0

            if ip < self.poor:
                Tp_gain = self.T(ir, ip, "P", "D")
                W[dico[(ir, ip + 1)], idx] = Tp_gain

            else:
                Tp_gain = 0

            if ip > 0:
                Tp_loss = self.T(ir, ip, "P", "C")
                W[dico[(ir, ip - 1)], idx] = Tp_loss

            else:
                Tp_loss = 0

            W[(idx, idx)] = 1 - Tr_gain - Tr_loss - Tp_gain - Tp_loss

            grad_ir[ip, ir] = Tr_gain - Tr_loss
            grad_ip[ip, ir] = Tp_gain - Tp_loss

            eigs, leftv, rightv = eig(W, left=True, right=True)
            domIdx = np.argmax(np.real(eigs))  # index of the dominant eigenvalue
            L = np.real(eigs[domIdx])  # the dominant eigenvalue
            p = np.real(
                rightv[:, domIdx])  # the right-eigenvector is the relative proportions in classes at steady state
            p = p / np.sum(p)  # normalise it


        return W, grad_ir, grad_ip, p




    def Plot(self, grad_iR, grad_iP, p, i_conf):

        P = np.zeros((self.poor + 1, self.rich + 1))  # rich on the x-axis
        for idx, pi in enumerate(p):
            iR, iP = i_conf[idx]
            P[iP, iR] = pi

        fig, ax = plt.subplots(figsize=(3, 6))

        iR = list(range(self.rich + 1))
        iP = list(range(self.poor + 1))


        norm = Normalize()
        norm.autoscale(grad_iR)
        cmp = cm.inferno

        ax.quiver(iR, iP, grad_iR, grad_iP, color = cmp(norm(grad_iR)))

        ax.set_xlim((-1, self.rich + 1))
        ax.set_ylim((-1, self.poor + 1))

        ax.set_xlabel(r'rich cooperators, $i_R$')
        ax.set_ylabel(r'poor cooperators, $i_P$')
        plt.axis('scaled')
        plt.tight_layout()
        im = ax.imshow(P, origin='lower', cmap='gray', alpha=0.5)
        plt.show()


    def __str__(self) -> str:
        """
        This method should return a string representation of the game.
        """
        return "ClimateGame Object"

    pass

    def type(self) -> str:
        """
        This method should return a string representing the type of game.
        """
        pass


if __name__ == '__main__':

    population_size = 40
    group_size = 6
    nb_rich = 8
    nb_poor = 32
    rich_end = 2.5
    poor_end = 0.625
    mu = 1 / population_size
    fraction_endowment = 0.1
    homophily = 0.2
    risk = 0.3
    M = 3  # Between 0 and group_size


    Game = ClimateGame(popuplation_size = population_size, group_size = group_size, nb_rich = nb_rich, nb_poor = nb_poor,
                 fraction_endowment = fraction_endowment, homophily = homophily, risk = risk, M = M, rich_end = rich_end, poor_end  = poor_end, mu = mu)


dico_conf, i_conf = Game.Enum_config(4,10)
W, grad_ir, grad_ip, p =Game.Generate_W_GoS(dico= dico_conf, i_conf= i_conf)

Game.Plot(grad_ir, grad_ip, p, i_conf)


