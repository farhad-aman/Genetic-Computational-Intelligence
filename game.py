import os

from matplotlib import pyplot as plt

from evolution import Evolution
from gui import GUI


class Game:
    def __init__(self, levels):
        self.levels = levels
        self.current_level_index = -1
        self.current_level_len = 0
        self.rewards = []

    def load_next_level(self):
        if self.current_level_index + 1 == len(self.levels):
            return False
        self.current_level_index += 1
        self.current_level_len = len(self.get_current_level()[1])
        return True

    def get_current_level(self):
        try:
            return self.levels[self.current_level_index]
        except Exception as e:
            print(e)

    # Used in get score
    WIN_REWARD = 4.0
    MAX_STEP_REWARD = 0.3
    DEATH_REWARD = -3.0

    # Used in get record
    FLAG_REWARD = 1.0
    MUSHROOM_REWARD = 2.0
    KILL_REWARD = 2.0
    JUMP_REWARD = -0.3

    def get_score(self, agent):
        self.rewards = [None for _ in range(self.current_level_len)]
        current_name, current_level = self.get_current_level()
        if current_level is None:
            return None
        result = self.get_score_record(agent, current_level)
        reward = result[3]  # Jump before flag
        reward += result[1] * self.MAX_STEP_REWARD  # Max number of step without dying
        reward += result[2] * self.DEATH_REWARD  # Negative reward if agent is dead
        reward += self.WIN_REWARD if result[2] == 0 else 0  # Positive reward for winning the game
        return result[2] == 0, round(reward, 4)

    def get_score_record(self, agent, level, index=0, air=False):
        if self.rewards[index] is not None:
            return self.rewards[index]
        action = agent[index]
        if index == self.current_level_len - 1:
            self.rewards[index] = (1, 1, 0, self.FLAG_REWARD if action == "1" else 0)
            return self.rewards[index]
        next_step = level[index + 1]
        reward = 0.0
        is_dead = False
        if (next_step == "G" and action != "1" and not air) or (next_step == "L" and action != "2"):
            is_dead = True
        if next_step == "G" and air:
            reward += self.KILL_REWARD
        if next_step == "M" and action != "1":
            reward += self.MUSHROOM_REWARD
        if action == "1":
            reward += self.JUMP_REWARD
        next_actions_score = self.get_score_record(agent, level, index + 1, action == "1")
        steps = 0 if is_dead else next_actions_score[0] + 1
        self.rewards[index] = (
            steps,
            max(steps, next_actions_score[1]),
            next_actions_score[2] + (1 if is_dead else 0),
            next_actions_score[3] + reward,
        )
        return self.rewards[index]


def main():
    PATH = "./levels/"
    _, _, filenames = next(os.walk(PATH))
    levels = [
        (
            os.path.basename(level.name).split(".")[0],
            level.read().strip(),
            level.close()
        )[0:2] for level in [open(PATH + name) for name in filenames]
    ]

    game = Game(levels)
    while game.load_next_level():
        evolution = Evolution(200, .2, game)
        level = game.get_current_level()
        print("==========================================================")
        print(level[0])
        best_agent, indices, min_scores, max_scores, average_scores = evolution.converge(200)
        score = best_agent[1]
        print(level[1])
        print(best_agent[0])
        print("Score: ", score[1])
        print("Win: ", score[0])
        print("==========================================================")

        GUI(best_agent[0], level[1], score, level[0])

        fig = plt.gcf()
        fig.canvas.manager.set_window_title(level[0])
        plt.plot(indices, max_scores, 'b')
        plt.plot(indices, average_scores, 'y')
        plt.plot(indices, min_scores, 'r')
        plt.legend(["Best", "Average", "Worst"])
        plt.show()


if __name__ == "__main__":
    main()
