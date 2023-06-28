import math
import random

import numpy as np


class Evolution:
    def __init__(self, agent_num, mutation_prob, game):
        self.agent_num = ((agent_num + 3) // 4) * 4
        self.mutation_prob = mutation_prob
        self.game = game
        self.level_len = game.current_level_len
        self.current_generation = 0
        self.agents = []
        self.max_scores = []
        self.min_scores = []
        self.average_scores = []
        for i in range(agent_num):
            new_agent = self.create_random_agent(self.level_len)
            self.agents.append((new_agent, self.game.get_score(new_agent)))

    @staticmethod
    def create_random_agent(length):
        init_probs = {
            0: 0.4,
            1: 0.3,
            2: 0.3,
        }
        new_agent = ""
        for i in range(length):
            rnd = random.random() * (init_probs[0] + init_probs[1] + init_probs[2])
            if rnd < init_probs[0]:
                new_agent += "0"
            elif rnd < init_probs[0] + init_probs[1]:
                if len(new_agent) and new_agent[-1] == "1":
                    new_agent += "0"
                else:
                    new_agent += "1"
            elif rnd <= init_probs[0] + init_probs[1] + init_probs[2]:
                if len(new_agent) and new_agent[-1] == "1":
                    new_agent += "0"
                else:
                    new_agent += "2"
        return new_agent

    @staticmethod
    def remove_duplicate_actions(agent):
        list_agent = list(agent)
        for i in range(len(agent) - 1):
            if list_agent[i] == "1":
                list_agent[i + 1] = "0"
        return ''.join(list_agent)

    def mutate(self, agent):
        if random.random() > self.mutation_prob:
            return agent
        list_agent = list(agent)
        mutation_index = random.randint(0, len(agent) - 1)
        list_agent[mutation_index] = "0"
        return ''.join(list_agent)

    @staticmethod
    def cross_over_single_point(agent1, agent2):
        index = random.randint(0, len(agent1))
        return agent1[:index] + agent2[index:], agent2[:index] + agent1[index:]

    @staticmethod
    def cross_over_random(agent1, agent2):
        child1 = ""
        child2 = ""
        for i in range(len(agent1)):
            parent = random.randint(1, 2)
            if parent == 1:
                child1 += agent1[i]
                child2 += agent2[i]
            else:
                child1 += agent2[i]
                child2 += agent1[i]
        return child1, child2

    def next_generation(self, probs):
        next_generation_indices = np.random.choice(len(self.agents), len(self.agents) // 2, p=probs)
        next_generation = [self.agents[i] for i in next_generation_indices]
        parent_indices = np.random.choice(len(self.agents), len(self.agents) // 2, p=probs)
        parents = [self.agents[i] for i in parent_indices]
        children = []
        for i in range(len(parents)):
            cross_overed_children = self.cross_over_single_point(parents[i][0], parents[(i + 1) % len(parents)][0])
            children.append(cross_overed_children[0])
            children.append(cross_overed_children[1])
        for i in range(len(children)):
            agent = self.remove_duplicate_actions(self.mutate(children[i]))
            children[i] = (agent, self.game.get_score(agent))
        for i in range(len(parents)):
            agent = self.remove_duplicate_actions(self.mutate(next_generation[i][0]))
            next_generation[i] = (agent, self.game.get_score(agent))
        children = sorted(children, key=lambda x: x[1][1])[-len(parents):]
        self.agents = children + next_generation
        self.current_generation += 1

    def rescale(self, x):
        b = 1.5 * self.game.current_level_len
        return math.exp(x / b)

    def calculate_probs(self):
        n = len(self.agents)
        probs = [0] * n
        scores_sum = 0
        rescaled_scores_sum = 0
        min_score = 1000000
        max_score = -1000000
        scores = []
        for i in range(len(self.agents)):
            score = self.agents[i][1][1]
            scores.append(score)
            scores_sum += score
            max_score = max(max_score, score)
            min_score = min(min_score, score)
            rescaled_score = self.rescale(score)
            probs[i] = rescaled_score
            rescaled_scores_sum += rescaled_score
        probs = [x / rescaled_scores_sum for x in probs]
        sorted(scores)
        return probs, scores_sum / len(self.agents), min_score, max_score

    def converge(self, generation_limit):
        indices = []
        best_agent = None
        best_result = (False, -1000000)
        while self.current_generation < generation_limit:
            probs, avg, min_score, max_score = self.calculate_probs()
            generation_best_agent = self.agents[probs.index(max(probs))]
            generation_best_agent_score = generation_best_agent[1]
            if generation_best_agent_score[1] > best_result[1]:
                best_result = generation_best_agent_score
                best_agent = generation_best_agent
            indices.append(self.current_generation)
            self.min_scores.append(min_score)
            self.max_scores.append(max_score)
            self.average_scores.append(avg)
            self.next_generation(probs)
        return best_agent, indices, self.min_scores, self.max_scores, self.average_scores
