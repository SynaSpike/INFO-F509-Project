""""
This file is used to code for the main model of the article: https://www.pnas.org/doi/full/10.1073/pnas.1323479111
Vasconcelos VV, Santos FC, Pacheco JM, Levin SA. 2014
Climate policies under wealth inequality. Proc. Natl Acad. Sci. USA 111, 2212-2216.
(doi:10.1073/pnas.1323479111) Crossref, PubMed, ISI, Google Scholar
"""
import random
import matplotlib.pyplot as plt

from scipy.special import binom

import egttools
import numpy as np
import PGGStrategy
import Player


class ClimateGame:


    def __init__(self, popuplation_size: int, group_size: int, nb_rich: int, nb_poor:int, strategies: list, profiles: list,
                 fraction_endowment: float, homophily:float, risk:float, M:float, rich_end:int, poor_end:int) -> None:
        self.population_size = popuplation_size  # Z
        self.group_size = group_size  # N
        self.nb_group = self.population_size // self.group_size
        self.rich = nb_rich  # Zr
        self.rich_end = rich_end
        self.poor_end = poor_end
        self.poor = nb_poor  # Zp
        self.strategies = strategies  # Ds or Cs
        self.profiles = profiles  # Poor or Rich
        self.nb_strategies = 4  # Dp Dr Cp Cr
        self.fraction_endowment = fraction_endowment  # C
        self.poor_coop = profiles[0].endowment * fraction_endowment # Cooperation of the Poor
        self.rich_coop = profiles[1].endowment * fraction_endowment # Cooperation of the rich
        self.homophily = homophily  # h
        self.risk = risk  # r
        self.treshold = M * fraction_endowment * ((((profiles[0].endowment * self.poor) + (profiles[1].endowment * self.rich)) /self.population_size))
        self.rich_pg = (self.rich / self.population_size) *  self.group_size  # Rich per group
        self.poor_pg = self.group_size - self.rich_pg   # Poor per group

        self.nb_group_configurations_ = egttools.calculate_nb_states(self.group_size, self.nb_strategies)
        self.payoffs_ = np.zeros(shape=(self.nb_strategies,self.nb_group))

        self.population = [] #  Player population
        self.groups = self.sample(nb_rich / popuplation_size)

        #  Initialize payoff matrix
        self.calculate_payoffs()

    def sample(self, wealth_ratio):
        """
        :return: np.ndarray shape(nb.groups,2) of number of poor and rich.
        """

        num_rich = self.rich
        num_poor = self.poor
        self.population = []

        for i in range(num_rich):
            self.population.append(Player.Player(wealth = 1, endowment= self.rich_end, strategy= random.choice(self.strategies)))
        for i in range(num_poor):
            self.population.append(Player.Player(wealth = 0, endowment= self.poor_end, strategy= random.choice(self.strategies)))

        random.shuffle(self.population)

        groups = [self.population[i:i + self.group_size] for i in range(0, self.group_size*self.nb_group, self.group_size)]

        return groups


    def get_comp(self, group):

        comp = np.zeros(self.nb_strategies)

        for plr in group:

            if plr.wealth == 0 and plr.strategy.action == 0:
                comp[0] += 1
            if plr.wealth == 1 and plr.strategy.action == 0:
                comp[1] += 1
            if plr.wealth == 0 and plr.strategy.action == 1:
                comp[2] += 1
            if plr.wealth == 1 and plr.strategy.action == 1:
                comp[3] += 1

        return comp

    def get_nbr_cat(self, population: list):
        """
        :param population: List of Player population
        :return: the number of (wealth; strat)
        """

        Dp = 0
        Dr = 0
        Cp = 0
        Cr = 0

        for plr in range(population):

            if plr.wealth == 0 and plr.strategy.action == 0:
                Dp += 1
            if plr.wealth == 1 and plr.strategy.action == 0:
                Dr += 1
            if plr.wealth == 0 and plr.strategy.action == 1:
                Cp += 1
            if plr.wealth == 1 and plr.strategy.action == 1:
                Cr += 1

        return [Dp, Dr, Cp, Cr]


    def calculate_payoffs(self) -> np.ndarray:
        """
        :return: payoff array Dp, Dr, Cp, Cr
        """
        payoff_container = np.zeros(self.nb_strategies)  # 4 different strategies Dp, Dr, Cp, Cr

        for i in range(self.nb_group):

            group_composition = self.get_comp(self.groups[i])

            # Get group composition
            self.play(group_composition, payoff_container) # Update the container with the new payoff following group_comp


            for strategy_index, strategy_payoff in enumerate(payoff_container):
                self.payoffs_[strategy_index, i] = strategy_payoff


            # Reinitialize payoff vector
            payoff_container[:] = 0
        return self.payoffs_

    def play(self, group_composition, game_payoffs: np.ndarray) -> None:
        """
        Calculates the payoff of each strategy inside the group.

        Parameters
        ----------
        group_composition: Union[List[int], numpy.ndarray]
            counts of each strategy inside the group.
        game_payoffs: numpy.ndarray
            container for the payoffs of each strategy
        """

        k = self.rich_coop * group_composition[3] + self.poor_coop * group_composition[2] - self.treshold

        if k >= 0:
            heav = 1

        else:
            heav = 0


        game_payoffs[0] = self.profiles[0].endowment * (heav + (1 - self.risk) * (1 - heav))
        game_payoffs[1] = self.profiles[1].endowment * (heav + (1 - self.risk) * (1 - heav))

        game_payoffs[2] = game_payoffs[0] - self.poor_coop
        game_payoffs[3] = game_payoffs[1] - self.rich_coop

        return game_payoffs

    def calculate_fitness(self, ir, ip) -> float:
        """
        # strategy_index: int, pop_size: int, population_state: np.ndarray
        This method should return the fitness of strategy
        with index `strategy_index` for the given `population_state`.
        :param: ir nbr of rich
        :param: ip nbr of poor
        :format:
        [ Dp, Dr, Cp, Cr ]

        """
    # rich cooperators
        sum_1 = 0

        for jr in range(self.group_size):
            sum_2 = 0
            for jp in range(self.group_size-jr-1, -1, -1):
                payoff = self.play([0, 0, jp, jr + 1], np.zeros(4))[3] # Rich coop
                # Do not care about the nbr of defector (does not affect payoff)
                sum_2 += binom(ir - 1, jr) * binom(ip, jp) * binom(self.population_size - ir - ip,
                                                                   self.group_size - 1 - jr - jp) * payoff
            sum_1 += sum_2

        RC = binom(self.population_size - 1, self.group_size - 1)**(-1) * sum_1


    # rich defectors
        sum_1 = 0

        for jr in range(self.group_size):
            sum_2 = 0
            for jp in range(self.group_size-jr-1, -1, -1):
                payoff = self.play([0, 0, jp, jr], np.zeros(4))[1]  # Rich defector
                # Do not care about the nbr of defector (does not affect payoff)
                sum_2 += binom(ir, jr) * binom(ip, jp) * binom(self.population_size - 1 - ir - ip,
                                                                   self.group_size - 1 - jr - jp) * payoff
            sum_1 += sum_2

        RD = binom(self.population_size - 1, self.group_size - 1)**(-1) * sum_1

    # poor cooperators
        sum_1 = 0

        for jr in range(self.group_size):
            sum_2 = 0
            for jp in range(self.group_size-jr-1, -1, -1):
                payoff = self.play([0, 0, jp + 1, jr], np.zeros(4))[2]  # Poor coop
                # Do not care about the nbr of defector (does not affect payoff
                sum_2 += binom(ir, jr) * binom(ip - 1, jp) * binom(self.population_size - ir - ip,
                                                               self.group_size - 1 - jr - jp) * payoff
            sum_1 += sum_2

        PC = binom(self.population_size - 1, self.group_size - 1) ** (-1) * sum_1

    # poor defectors
        sum_1 = 0

        for jr in range(self.group_size):
            sum_2 = 0
            for jp in range(self.group_size-jr-1, -1, -1):

                payoff = self.play([0, 0, jp, jr], np.zeros(4))[0]  # Poor defector
                # Do not care about the nbr of defector (does not affect payoff
                sum_2 += binom(ir, jr) * binom(ip, jp) * binom(self.population_size - 1 - ir - ip,
                                                                   self.group_size - 1 - jr - jp) * payoff
            sum_1 += sum_2

        PD = binom(self.population_size - 1, self.group_size - 1) ** (-1) * sum_1

        return [PD, RD, PC, RC]


    def transition_probabilities(self, ir, ip, rounding:bool):
        """
        This function is used to return T
        :return: T prob
        :format:
        --------------
       Dp |Dp Cp Dr Cr |
       Cp |Dp Cp Dr Cr |
       Dr |Dp Cp Dr Cr |
       Cr |Dp Cp Dr Cr |
        -------------
        """
        fit = self.calculate_fitness(ir, ip) # Dp Dr Cp Cr
        Dp = self.poor - ip # Nbr of poor defector
        Dr = self.rich - ir # Nbr of rich defector
        beta = 3.
        mu = 1.4 * 1/self.population_size


        T_prob = np.zeros(shape=(self.nb_strategies, self.nb_strategies))

        # Transition Cp -> Dp
        fermi_1 = (1 + np.exp(beta * (fit[2]- fit[0])))** -1 # Cp -> Dp
        fermi_2 = (1 + np.exp((beta * (fit[2] - fit[1])))) ** -1 # Cp -> Dr
        param1 = (Dp / (self.poor - 1 + (1 - self.homophily)* self.rich))
        param2 =  ((1 - self.homophily) * Dr) / (self.poor - 1 + (1 - self.homophily) * self.rich)
        T_prob[1,0] = ( ip / self.population_size ) * ( ( 1 - mu) * ( param1 * fermi_1  + param2 * fermi_2 ) + mu )

        # Transition Cr -> Dr
        fermi_1 = (1 + np.exp(beta * (fit[3] - fit[1]))) ** -1  # Cr -> Dr
        fermi_2 = (1 + np.exp((beta * (fit[3] - fit[0])))) ** -1  # Cr -> Dp
        param1 = (Dr / (self.rich - 1 + (1 - self.homophily) * self.poor))
        param2 = ((1 - self.homophily) * Dp) / (self.rich - 1 + (1 - self.homophily) * self.poor)
        T_prob[3,2] = (ir / self.population_size) * ((1 - mu) * (param1 * fermi_1 + param2 * fermi_2) + mu)

        # Transition Dp -> Cp
        fermi_1 = (1 + np.exp(beta * (fit[0] - fit[2]))) ** -1  # Dp -> Cp
        fermi_2 = (1 + np.exp((beta * (fit[0] - fit[3])))) ** -1  # Dp -> Cr
        param1 = (ip / (self.poor - 1 + (1 - self.homophily) * self.rich))
        param2 = ((1 - self.homophily) * ir) / (self.poor - 1 + (1 - self.homophily) * self.rich)
        T_prob[0,1] = (Dp / self.population_size) * ((1 - mu) * (param1 * fermi_1 + param2 * fermi_2) + mu)

        # Transition Dr -> Cr
        fermi_1 = (1 + np.exp(beta * (fit[1] - fit[3]))) ** -1  # Dr -> Cr
        fermi_2 = (1 + np.exp((beta * (fit[1] - fit[2])))) ** -1  # Dr -> Cp
        param1 = (ir / (self.rich - 1 + (1 - self.homophily) * self.poor))
        param2 = ((1 - self.homophily) * ip) / (self.rich - 1 + (1 - self.homophily) * self.poor)
        T_prob[2,3] = (Dr / self.population_size) * ((1 - mu) * (param1 * fermi_1 + param2 * fermi_2) + mu)


        T_prob[0:1,2:3] = 0 # Poor -> Rich
        T_prob[2:3, 0:1] = 0 # Rich -> Poor
        T_prob[0,0] = 1 - T_prob[0,1]
        T_prob[1,1] = 1 - T_prob[1,0]
        T_prob[2, 2] = 1 - T_prob[2, 3]
        T_prob[3, 3] = 1 - T_prob[3, 2]

        if rounding:
            # Rounding 4 numbers
            for i in range(len(T_prob)):
                for j in range(len(T_prob[i])):
                    T_prob[i,j] = round(T_prob[i,j], 4)

        return T_prob

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
            fermi_1 = (1 + np.exp(beta * (fit[2] - fit[0]))) ** -1  # Cp -> Dp
            fermi_2 = (1 + np.exp((beta * (fit[2] - fit[1])))) ** -1  # Cp -> Dr
            param1 = (Dp / (self.poor - 1 + (1 - self.homophily) * self.rich))
            param2 = ((1 - self.homophily) * Dr) / (self.poor - 1 + (1 - self.homophily) * self.rich)
            return (ip / self.population_size) * ((1 - mu) * (param1 * fermi_1 + param2 * fermi_2) + mu)

        # Transition Cr -> Dr
        if k == "R" and X == "C":
            fermi_1 = (1 + np.exp(beta * (fit[3] - fit[1]))) ** -1  # Cr -> Dr
            fermi_2 = (1 + np.exp((beta * (fit[3] - fit[0])))) ** -1  # Cr -> Dp
            param1 = (Dr / (self.rich - 1 + (1 - self.homophily) * self.poor))
            param2 = ((1 - self.homophily) * Dp) / (self.rich - 1 + (1 - self.homophily) * self.poor)
            return (ir / self.population_size) * ((1 - mu) * (param1 * fermi_1 + param2 * fermi_2) + mu)

        # Transition Dp -> Cp
        if k == "P" and X == "D":
            fermi_1 = (1 + np.exp(beta * (fit[0] - fit[2]))) ** -1  # Dp -> Cp
            fermi_2 = (1 + np.exp((beta * (fit[0] - fit[3])))) ** -1  # Dp -> Cr
            param1 = (ip / (self.poor - 1 + (1 - self.homophily) * self.rich))
            param2 = ((1 - self.homophily) * ir) / (self.poor - 1 + (1 - self.homophily) * self.rich)
            return (Dp / self.population_size) * ((1 - mu) * (param1 * fermi_1 + param2 * fermi_2) + mu)

        # Transition Dr -> Cr
        if k == "R" and X == "D":
            fermi_1 = (1 + np.exp(beta * (fit[1] - fit[3]))) ** -1  # Dr -> Cr
            fermi_2 = (1 + np.exp((beta * (fit[1] - fit[2])))) ** -1  # Dr -> Cp
            param1 = (ir / (self.rich - 1 + (1 - self.homophily) * self.poor))
            param2 = ((1 - self.homophily) * ip) / (self.rich - 1 + (1 - self.homophily) * self.poor)
            return (Dr / self.population_size) * ((1 - mu) * (param1 * fermi_1 + param2 * fermi_2) + mu)



    def population_configuration(self):

        pop_conf = []

        for i in range(self.rich + 1):

            for j in range(self.poor + 1):

                pop_conf.append((i,j))

        return pop_conf

    def MarkovProcess(self, i_conf):

        P_conf_0 = [self.poor - i_conf[1], i_conf[1], self.rich - i_conf[0], i_conf[0]]


        # Adding one rich coop (Dr -> Cr)
        P_conf_1 = [x + y for x, y in zip(P_conf_0, [0,0,-1,1])]
        T_1_1 = self.transition_probabilities(i_conf[0] + 1, i_conf[1], rounding=False)[2,:] # Transitioning from i' to i -> Cr -> Dr
        T_1_2 = self.transition_probabilities(i_conf[0], i_conf[1], rounding=False) [3,:] # T from i to i' -> Dr-> Cr
        param_1_1 = [x * y for x, y in zip(T_1_1, P_conf_1)] # Matrice products more convenient
        param_1_2 = [x * y for x, y in zip(T_1_2, P_conf_0)]
        param1 = [x - y for x, y in zip(param_1_1, param_1_2)]

        print("T_1_1: ", T_1_1)
        print("T_1_2: ", T_1_2)
        print("param_1_1: ", param_1_1)
        print("param_1_2: ", param_1_2)
        print("P_conf_0: ", P_conf_0)
        print("P_conf_1: ", P_conf_1)
        print("param: ", param1)


        # Adding one rich defector (Cr -> Dr)
        P_conf_2 =  [x + y for x, y in zip(P_conf_0, [0,0,1,-1])]
        T_2_1 = self.transition_probabilities(i_conf[0] - 1, i_conf[1], rounding=False)[3,:]
        T_2_2 =self.transition_probabilities(i_conf[0], i_conf[1], rounding=False) [2,:]
        param_2_1 = [x * y for x, y in zip(T_2_1, P_conf_2)]
        param_2_2 = [x * y for x, y in zip(T_2_2, P_conf_0)]
        param2 = [x - y for x, y in zip(param_2_1, param_2_2)]


        # Adding one poor coop (Dp -> Cp )
        P_conf_3 = [x + y for x, y in zip(P_conf_0, [-1,1,0,0])]
        T_3_1 = self.transition_probabilities(i_conf[0], i_conf[1] + 1, rounding=False)[0,:]
        T_3_2 = self.transition_probabilities(i_conf[0], i_conf[1], rounding=False)[1,:]
        param_3_1 = [x * y for x, y in zip(T_3_1, P_conf_3)]
        param_3_2 = [x * y for x, y in zip(T_3_2, P_conf_0)]
        param3 = [x - y for x, y in zip(param_3_1, param_3_2)]

        # Adding one poor defector (Cp -> Dp )
        P_conf_4 =  [x + y for x, y in zip(P_conf_0, [1,-1,0,0])]
        T_4_1 = self.transition_probabilities(i_conf[0], i_conf[1] - 1, rounding=False)[1, :]
        T_4_2 = self.transition_probabilities(i_conf[0], i_conf[1], rounding=False)[0, :]
        param_4_1 = [x * y for x, y in zip(T_4_1, P_conf_4)]
        param_4_2 = [x * y for x, y in zip(T_4_2, P_conf_0)]
        param4 = [x - y for x, y in zip(param_4_1, param_4_2)]

        delta_finale = [x + y + w + z for x, y, w, z in zip(param1, param2, param3, param4)]
        delta_finale_arr = [round(i) for i in delta_finale]
        P_t = [x + y for x, y in zip(P_conf_0, delta_finale_arr)]

        print("Conf 0: ", P_conf_0)
        print("Conf 1: ", P_conf_1)
        print("Conf 2: ", P_conf_2)
        print("Conf 3: ", P_conf_3)
        print("Conf 4: ", P_conf_4)



        return P_conf_0, P_t, delta_finale_arr

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

        print(i_conf)
        for idx, i in enumerate(i_conf):
            print("idx : ", idx)
            print("i: ", i)

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

        return W, grad_ir, grad_ip




    def Plot(self, grad_iR, grad_iP ):

        fig, ax = plt.subplots(figsize=(3, 6))

        iR = list(range(self.rich + 1))
        iP = list(range(self.poor + 1))

        ax.quiver(iR, iP, grad_iR, grad_iP)

        ax.set_xlim((-1, self.rich + 1))
        ax.set_ylim((-1, self.poor + 1))

        ax.set_xlabel(r'rich cooperators, $i_R$')
        ax.set_ylabel(r'poor cooperators, $i_P$')
        plt.axis('scaled')
        plt.tight_layout()
        plt.show()

    @staticmethod

    def fermi(beta:float, fitness_diff:float):

        return (1 + np.exp(beta*fitness_diff))**(-1)

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

    strategy_defect = PGGStrategy.PGGStrategy(0)
    strategy_coop = PGGStrategy.PGGStrategy(1)
    strategies = [strategy_defect, strategy_coop]

    player_p = Player.Player(0, 0.625, strategies)
    player_r = Player.Player(1, 2.5, strategies)

    population_size = 40
    group_size = 6
    nb_rich = 8
    nb_poor = 32

    profiles = [player_p, player_r]
    fraction_endowment = 0.1
    homophily = 0.
    risk = 0.2
    M = 3  # Between 0 and group_size

    Game = ClimateGame(popuplation_size= population_size, group_size= group_size,  nb_rich= nb_rich, nb_poor=nb_poor, strategies= strategies,
                       profiles= profiles, fraction_endowment= fraction_endowment, homophily= homophily, risk= risk, M= M,
                       rich_end= 2.5, poor_end= 0.625)

"""

    print("------------------------ PAYOFF ---------------------")
    print("PAYOFF REACHED TRESHOLD: ", Game.play([0,0,4,2], [0,0,0,0]), "\n")
    print("PAYOFF UNREACHED TRESHOLD: ", Game.play([0, 0, 0, 0], [0, 0, 0, 0]), "\n")
    print("------------------------ FITNESSES ---------------------")
    print(Game.calculate_fitness(39, 159), "\n")
    print("----------------------- FULL TRANSITION PROB ----------------")
    print(Game.transition_probabilities(20, 60, rounding=False), "\n")
    print("----------------------- TRANSITION PROB ----------------")
    print(Game.transition_probabilities(20, 60, rounding= False)[:,3], "\n")
    print("----------------------- NUMBER OF CONFIG ----------------")
    print(len(Game.population_configuration()),"\n")
    print("----------------------- TRANSITION MATRIX ----------------")
    #print(Game.transition_matrix())
    print("----------------------- MARKOV PROCESS ----------------")
    #a,b,c = Game.MarkovProcessV2((20, 60))
    #print(" Initial Conf : ", a)
    #print(" Next Conf: ", b)
    #print(" Delta Conf: ", c)

    print("----------------------- T COMPARISON ----------------")
    print(Game.transition_probabilities(4,10, rounding=False), "\n")

    print("----------------------- CONF COMPARISON ----------------")
    print(len(Game.Enum_config(4,10)), "\n")

    print("----------------------- T COMPARISON ----------------")
    print(Game.T(4, 10,"R", "D"), "\n")
    print("----------------------- DICO COMPARISON ----------------")
    print(Game.Enum_config(4,10))

"""
a,b = Game.Enum_config(4,10)
W, grad_iR, grad_iP = Game.Generate_W_GoS(a, b)
Game.Plot(grad_iR, grad_iP)






