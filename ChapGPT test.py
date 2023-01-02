

# code from chapGPT

import random

class PublicGoodGame:
    def init(self, population_size, wealth_ratio, group_size, initial_endowment, threshold, mutation_prob):
        self.population_size = population_size
        self.wealth_ratio = wealth_ratio
        self.group_size = group_size
        self.initial_endowment = initial_endowment
        self.threshold = threshold
        self.mutation_prob = mutation_prob

        self.population = []
        self.create_population()

    def create_population(self):
        num_rich = int(self.population_size * self.wealth_ratio)
        num_poor = self.population_size - num_rich

        for i in range(num_rich):
          self.population.append(Individual(True, self.initial_endowment[0]))
        for i in range(num_poor):
          self.population.append(Individual(False, self.initial_endowment[1]))

    def play_game(self):
        groups = self.create_groups()
        for group in groups:
            contributions = []
            for individual in group:
                if individual.strategy == "cooperate":
                    contribution = individual.endowment * self.contribution_fraction
                    individual.endowment -= contribution
                    contributions.append(contribution)
                else:
                    contributions.append(0)

            total_contribution = sum(contributions)
            if total_contribution >= self.threshold:
                for i, individual in enumerate(group):
                    individual.endowment += contributions[i]
            else:
                for individual in group:
                    individual.endowment = 0

    def create_groups(self):
        random.shuffle(self.population)
        groups = [self.population[i:i+self.group_size] for i in range(0, len(self.population), self.group_size)]
        return groups

    def update_strategies(self):
        for individual in self.population:
            other = random.choice(self.population)
            prob = self.fermi_function(individual.fitness, other.fitness)
            if random.uniform(0, 1) < prob:
                individual.strategy = other.strategy
            if random.uniform(0, 1) < self.mutation_prob:
                individual.strategy = "cooperate" if individual.strategy == "defect" else "defect"

    def fermi_function(self, x, y):
        return 1 / (1 + math.exp(-self.k * (x - y)))

    def calculate_fitness(self):
        for individual in self.population:
            if individual.strategy == "cooperate":
                individual.fitness = individual.endowment - self.contribution_cost
            else:
                individual.fitness = individual.endowment

class Individual:
    def init(self, wealth, endowment):
        self.wealth = wealth
        self.endowment = endowment
        self.strategy = "cooperate" if random.uniform(0, 1) < 0.5 else "defect"
        self.fitness = 0
