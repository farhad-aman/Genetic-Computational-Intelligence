import copy

import numpy as np

from dna import DNA


class Evolution:
    fitness = ([], [], [])

    def __init__(self):
        pass

    def next_population_selection(self, current_generation, next_generation_num):
        """
        Gets list of previous and current generation (μ + λ) and returns next_gen_num number of DNAs based on their
        fitness value.
        """
        selection = self.roulette_wheel(current_generation, next_generation_num)
        self.learning_curve(current_generation)
        return selection

    @staticmethod
    def sorting_players(generation):
        return sorted(generation, reverse=True, key=lambda dna: dna.fitness)

    def learning_curve(self, players):
        sorted_players = sorted(players, key=lambda player: player.fitness, reverse=True)
        best_fitness = sorted_players[0].fitness
        worst_fitness = sorted_players[len(sorted_players) - 1].fitness
        fitnesses = [player.fitness for player in players]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        with open('learning_curve.txt', 'a') as f:
            f.write(f'{best_fitness} {worst_fitness} {avg_fitness} \n')

    @staticmethod
    def roulette_wheel(current_generation, next_generation_num):
        fitness_sum = sum([dna.fitness for dna in current_generation])
        dna_probs = [dna.fitness / fitness_sum for dna in current_generation]
        return list(np.random.choice(current_generation, size=next_generation_num, p=dna_probs))

    def generate_new_population(self, generation_size, prev_players=None):
        """
        Gets survivors and returns a list containing generation_size number of children.
        :return: A list of children
        """
        is_first_generation = prev_players is None
        cross_over_prob = np.random.uniform(0, 1)
        mutation_prob = 0.8

        if is_first_generation:
            return [DNA() for _ in range(generation_size)]
        else:
            # TODO ( Parent selection and child generation )
            new_generation = []
            parents = self.q_tournament(2, prev_players, num_players)
            children = []
            # crossover
            i = 0
            while i < len(parents):
                if cross_over_prob >= np.random.uniform(0, 1):
                    new_generation[i], new_generation[i + 1] = self.cross_over(parents[i], parents[i + 1])
                i += 2
            # mutation
            for i in range(len(new_generation)):
                if mutation_prob >= np.random.uniform(0, 1):
                    new_generation[i] = self.mutate(new_generation[i])
            return new_generation

    def mutate(self, parent):
        child = self.clone_player(parent)
        sigma = 0.8
        for i in range(len(parent.nn.layer_sizes) - 1):
            child.nn.weights[i] += sigma * np.random.standard_normal(
                size=(parent.nn.layer_sizes[i + 1], parent.nn.layer_sizes[i]))
            child.nn.bias[i] += sigma * np.random.standard_normal(size=(parent.nn.layer_sizes[i + 1], 1))
        return child

    def cross_over(self, parent1, parent2):
        alpha = 0.1
        child1 = self.clone_player(parent1)
        child2 = self.clone_player(parent2)
        for i in range(len(parent1.nn.layer_sizes) - 1):
            child1.nn.weights[i] = alpha * parent1.nn.weights[i] + (1 - alpha) * parent2.nn.weights[i]
            child2.nn.weights[i] = alpha * parent2.nn.weights[i] + (1 - alpha) * parent1.nn.weights[i]
            child1.nn.bias[i] = alpha * parent1.nn.bias[i] + (1 - alpha) * parent2.nn.bias[i]
            child2.nn.bias[i] = alpha * parent2.nn.bias[i] + (1 - alpha) * parent1.nn.bias[i]
        return child1, child2

    def clone_player(self, player):
        """
    Gets a player as an input and produces a clone of that player.
    """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
