from environment import BaseEnvironment

from gym.envs.toy_text.discrete import categorical_sample, DiscreteEnv
from gym.envs.registration import register, registry, spec
import gym
from collections import deque
import numpy as np
import sys

try:
    from io import StringIO
except ImportError:
    from io import StringIO
from gym import ObservationWrapper, Wrapper
from gym.wrappers.time_limit import TimeLimit
from gym import utils
from gym import spaces
from gym import Env
from gym.utils import seeding

MAPS = {
    "action_test": [
        "HHHHHHHHHHHHHHHG",
        "HHHHHHHFFFFHHHHF",
        "HHHHHHHFHHFHHHHF",
        "HHHFFFFFHHFFHHFF",
        "HHFFHHFHHHHFFHFH",
        "HHFHHHFFHHHHFFFH",
        "HHFFHHHFHHHFFHHH",
        "HHHFFFHHHHHFHHHH",
        "HHHFHFHHHHHFHHHH",
        "FFFFHFHHHHHHHHHH",
        "FHHHHFHHHHHHHHHH",
        "SHHHHHHHHHHHHHHH"
    ],
    "4x4_fl": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8_fl": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
    "4x4_adj_goal": [
        "HHHG",
        "AFAF",
        "FHHH",
        "SHHH",
    ],
    "4x4": [
        "HHHG",
        "FFFF",
        "FHHH",
        "SHHH",
    ],
    "5x6": [
        "HHHHGH",
        "HFFFFH",
        "HFHHHH",
        "HSHHHH",
        "HHHHHH",
    ],
    "9x9": [
        "HHHHHHHHG",
        "HHHHHHHHF",
        "HHHHHHHHF",
        "HHHHHHHHF",
        "FFFFFFFFF",
        "FHHHHHHHH",
        "FHHHHHHHH",
        "FHHHHHHHH",
        "SHHHHHHHH",
    ],
    "9x9_maze": [
        "HHHHHHHHG",
        "HHHHHFHHF",
        "HHHFFFHHF",
        "HHHFHHHHF",
        "FFFFFFFFF",
        "FHHHHHHHH",
        "FFFFFHHHH",
        "FHHHFHHHH",
        "SHHHHHHHH",
    ],
    "11x10_maze_montezuma": [
        "HHHHHHHHHG",
        "HHHHHHAHHF",
        "HHHHFFFHHF",
        "HHHHFHHHHF",
        "HFFFFFFFFF",
        "HFHHHHHHHH",
        "HFHWWWWWWW",
        "HFFFFFFFFF",
        "HFHWFFFFFF",
        "HSHWFFFFFF",
        "HHHWFFFFFF",
    ],
    "1x4": [
        "SFFG",
    ],
    "2x4": [
        "SFFG",
        "HHHH",
    ],
    "2x5": [
        "SFFFF",
        "HWGWH",
    ],
    "2x11": [
        "SFFFFFFFFFF",
        "HWWWWGWWWWH",
    ],
    "5x5_risk": [
        "WSWWW",
        "WFFFF",
        "WFWWF",
        "HAHWF",
        "WFGFF",
    ],
}


class CorridorEnv(Env):
    """
      The surface is described using a grid like the following

        HHHG
        FFFF
        SWHH
        AHHH

      S : starting point, safe
      F : frozen surface, safe
      W : wall, safe (no state change)
      H : hole, fall to your doom
      A : adjacent goal
      G : distant goal

      Note that an implicit wall surrounding the whole 
      grid surface is assumed. There is no need to explicitly
      specify such wall and you should indeed not do it.

      The episode ends when you reach the goal or fall in a hole.

      Types of reward functions (can be combined; as a list):

      standard: (always active)
      You receive a reward of 0.5 if you reach the adjacent goal, 
      1 if you reach the distant goal, and zero otherwise.

      negative_hole reward:
      -1 if you fall in a hole 

      positive_steps reward:
      0.1 if you advance w/o falling in a hole or hitting a wall.

      negative_steps reward:
      -0.1 for every elapsed time step

    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="9x9", n_actions=5, random_start=True, reward_type=["standard"],
                 is_slippery=False):
        """
        - nS: number of states
        - nA: number of actions
        - P: transitions (*)
        - isd: initial state distribution (**)
        (*) dictionary dict of dicts of lists, where
          P[s][a] == [(probability, nextstate, reward, done), ...]
        (**) list or array of length nS
        """
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.map_name = map_name
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        if type(reward_type) == str:
            self.reward_type = [reward_type]
        if not "standard" in reward_type:
            self.reward_type.append("standard")

        n_states = nrow * ncol

        if random_start:
            self.desc[(self.desc == b'S')] = b'F'
            isd = np.array((self.desc == b'S') | (self.desc == b'F')).astype('float64').ravel()
        else:
            isd = np.array(self.desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        # Transition probabilities. A nested dictionary containing, for each state s and action a,
        # a list of posible transitions. Each transition is defined by a tuple consisting of the
        # probability of reaching the next-state when executing a, the next-state, the reward and an
        # episode termination flag
        P = {s: {a: [] for a in range(n_actions)} for s in range(n_states)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            _col = col
            _row = row
            if a == 0:  # left
                _col = max(col - 1, 0)
            elif a == 1:  # down
                _row = min(row + 1, nrow - 1)
            elif a == 2:  # right
                _col = min(col + 1, ncol - 1)
            elif a == 3:  # up
                _row = max(row - 1, 0)

            # Stay at current state if we hit a wall
            if desc[_row, _col] == b'W':
                _row = row
                _col = col

            return (_row, _col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                letter = desc[row, col]
                for a in range(n_actions):
                    p_s_a = P[s][a]
                    if letter in b'GH':
                        p_s_a.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                p_s_a.append((1.0 / 3.0, newstate, self.reward, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            p_s_a.append((1.0, newstate, self.reward, done))

        if np.array(desc == b'A').any():
            self.exists_adjacent_goal = True
            self.adjacent_goal_done = False
            self.nS = 2 * n_states
        else:
            self.exists_adjacent_goal = False
            self.adjacent_goal_done = True
            self.nS = n_states
        self.nA = n_actions
        self.P = P
        self.isd = isd
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        self.lastaction = None  # for rendering

        self.seed()
        self.reset()

    def reward(self, newletter, row, col, newrow, newcol):
        rew = 0
        if newletter == b'A' and not self.adjacent_goal_done:
            self.desc[(self.desc == b'A')] = b'F'  # Remove the adjacent goal
            self.adjacent_goal_done = True
            rew = 0.5
        elif newletter == b'G' and self.adjacent_goal_done:
            rew = 1.0
        elif (newletter == b'H') and 'negative_hole' in self.reward_type:
            rew = -1.0

        if 'positive_steps' in self.reward_type and (newrow != row or newcol != col) and newletter in [b'F', b'A',
                                                                                                       b'G']:
            rew += 0.1

        if 'negative_steps' in self.reward_type:
            rew -= 0.1

        return rew

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, new_s, r, d = transitions[i]

        row, col = self.coordinates(self.s)
        new_row, new_col = self.coordinates(new_s)
        new_letter = self.desc[new_row, new_col]
        _r = r(new_letter, row, col, new_row, new_col)

        self.s = new_s
        self.lastaction = a
        new_s = new_s + self.exists_adjacent_goal * self.nS / 2
        return (new_s, _r, d, {"prob": p})

    def coordinates(self, s):
        row = s // self.ncol
        col = s % self.ncol
        return (row, col)

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.start_s = self.s
        self.lastaction = None
        return self.s + self.exists_adjacent_goal * self.nS / 2

    def render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO.StringIO() if mode == 'ansi' else sys.stdout

        row_start, col_start = self.start_s // self.ncol, self.start_s % self.ncol
        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row_start][col_start] = 'S'
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        if self.lastaction is not None:
            outfile.write("  ({})\n".format(self.get_action_meanings()[self.lastaction]))
        else:
            outfile.write("\n")

        outfile.write("\n".join("".join(row) for row in desc) + "\n")

        return outfile

    def get_action_meanings(self):
        return [["Left", "Down", "Right", "Up"][i] if i < 4 else "NoOp" for i in range(self.action_space.n)]


class CorridorEnv2(DiscreteEnv):
    """
    The surface is described using a grid like the following
  
      HHHG
      FFFF
      SWHH
      AHHH
  
    S : starting point, safe
    F : frozen surface, safe
    W : wall, safe (no state change)
    H : hole, fall to your doom
    A : adjacent goal
    G : distant goal
  
    Note that an implicit wall surrounding the whole 
    grid surface is assumed. There is no need to explicitly
    specify such wall and you should indeed not do it.
  
    The episode ends when you reach the goal or fall in a hole.
  
    Types of reward functions (can be combined; as a list):
  
    standard: (always active)
    You receive a reward of 0.5 if you reach the adjacent goal, 
    1 if you reach the distant goal, and zero otherwise.
  
    negative_hole reward:
    -1 if you fall in a hole 
  
    positive_steps reward:
    0.1 if you advance w/o falling in a hole or hitting a wall.
  
    negative_steps reward:
    -0.1 for every elapsed time step
  
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="9x9", n_actions=5, random_start=True, reward=["standard"],
                 is_slippery=False):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.map_name = map_name
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        # self.action_space = spaces.Discrete(n_actions)
        # self.observation_space = spaces.Discrete(desc.size)
        # self.observation_space = DiscreteSpace(desc.size)
        if type(reward) == str:
            reward = [reward]
        if not "standard" in reward:
            reward.append("standard")

        n_state = nrow * ncol

        self.adjacent_goal_done = False

        row_goal = np.where(desc == b'G')[0][0]
        col_goal = np.where(desc == b'G')[1][0]

        if random_start:
            isd = np.array((self.desc == b'S') | (self.desc == b'F')).astype('float64').ravel()
            self.desc[(self.desc == b'S')] = b'F'
        else:
            isd = np.array(self.desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        # Transition probabilities. A nested dictionary containing, for each state s and action a,
        # a list of posible transitions. Each transition is defined by a tuple consisting of the
        # probability of reaching the next-state when executing a, the next-state, the reward and an
        # episode termination flag
        P = {s: {a: [] for a in range(n_actions)} for s in range(n_state)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            _col = col
            _row = row
            if a == 0:  # left
                _col = max(col - 1, 0)
            elif a == 1:  # down
                _row = min(row + 1, nrow - 1)
            elif a == 2:  # right
                _col = min(col + 1, ncol - 1)
            elif a == 3:  # up
                _row = max(row - 1, 0)

            # Stay at current state if we hit a wall
            if desc[_row, _col] == b'W':
                _row = row
                _col = col

            return (_row, _col)

        def distance_to_goal(row, col):
            return np.sqrt((row - row_goal) ** 2 + (col - col_goal) ** 2)

        def get_reward(newletter, row, col, newrow, newcol):
            rew = 0
            if newletter == b'A' and not self.adjacent_goal_done:
                # self.desc[(self.desc == b'A')]= b'F' #Remove the adjacent goal
                self.adjacent_goal_done = True
                rew = 0.5
            elif newletter == b'G':  # and self.adjacent_goal_done:
                rew = 1.0
            elif (newletter == b'H') and 'negative_hole' in reward:
                rew = -1.0

            if 'positive_steps' in reward and (newrow != row or newcol != col) and newletter in [b'F', b'A', b'G']:
                rew += 0.1

            if 'negative_steps' in reward:
                rew -= 0.1

            return rew

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                letter = desc[row, col]
                for a in range(n_actions):
                    p_s_a = P[s][a]
                    if letter in b'GH':
                        p_s_a.append((1.0, s, 0, True))
                    else:
                        if letter == b'A' and is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                rew = get_reward(newletter, row, col, newrow, newcol)
                                p_s_a.append((1.0 / 3.0, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = get_reward(newletter, row, col, newrow, newcol)
                            p_s_a.append((1.0, newstate, rew, done))
                            # p_s_a.append((1.0/3.0, newstate, rew, done))

        super(CorridorEnv2, self).__init__(n_state, n_actions, P, isd)

    def reset(self):
        self.start_s = super(CorridorEnv2, self).reset()
        return self.start_s

    def render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO.StringIO() if mode == 'ansi' else sys.stdout

        row_start, col_start = self.start_s // self.ncol, self.start_s % self.ncol
        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row_start][col_start] = 'S'
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        if self.lastaction is not None:
            outfile.write("  ({})\n".format(self.get_action_meanings()[self.lastaction]))
        else:
            outfile.write("\n")

        outfile.write("\n".join("".join(row) for row in desc) + "\n")

        return outfile

    def get_action_meanings(self):
        return [["Left", "Down", "Right", "Up"][i] if i < 4 else "NoOp" for i in range(self.action_space.n)]


class ComplexActionSetCorridorEnv(DiscreteEnv):
    """
    The surface is described using a grid like the following
  
      HHHD
      FFFF
      SHHH
      AHHH
  
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    A : adjacent goal
    G : distant goal
  
    The episode ends when you reach the goal or fall in a hole.
  
    simple reward:
    You receive a reward of 0.5 if you reach the adjacent goal, 
    1 if you reach the distant goal, and zero otherwise.
  
    negative reward:
    You receive a reward of 0.5 if you reach the adjacent goal, 
    1 if you reach the distant goal, -1 if you fall in a hole 
    and zero otherwise.
  
    steps reward:
    You receive a reward of 0.5 if you reach the adjacent goal, 
    1 if you reach the distant goal, 0.1 if you advance w/o falling
    in a hole and zero otherwise.
  
    negative_and_steps reward:
    You receive a reward of 0.5 if you reach the adjacent goal, 
    1 if you reach the distant goal, 0.1 if you advance w/o falling
    in a hole, -1 if you fall in a hole and zero otherwise.
  
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="9x9", n_actions=8, random_start=True, reward="simple", is_slippery=False):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.map_name = map_name
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        # self.action_space = spaces.Discrete(n_actions)
        # self.observation_space = spaces.Discrete(desc.size)
        # self.observation_space = DiscreteSpace(desc.size)

        self.adjacent_goal_done = False if np.array(desc == b'A').any() else True

        n_state = nrow * ncol

        if random_start:
            self.desc[(self.desc == b'S')] = b'F'
            isd = np.array((self.desc == b'S') | (self.desc == b'F')).astype('float64').ravel()
        else:
            isd = np.array(self.desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(n_actions)} for s in range(n_state)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == 0:  # left
                col = max(col - 1, 0)
            elif a == 1:  # 1 down
                row = min(row + 1, nrow - 1)
            elif a == 2:  # 2 down
                row = min(row + 2, nrow - 1)
            elif a == 3:  # 3 down
                row = min(row + 3, nrow - 1)
            elif a == 4:  # right
                col = min(col + 1, ncol - 1)
            elif a == 5:  # 1 up
                row = max(row - 1, 0)
            elif a == 6:  # 2 up
                row = max(row - 2, 0)
            elif a == 7:  # 3 up
                row = max(row - 3, 0)

            return (row, col)

        def get_reward(newletter, row, col, newrow, newcol):
            if newletter == b'A' and not self.adjacent_goal_done:
                self.desc[(self.desc == b'A')] = b'F'  # Remove the adjacent goal
                self.adjacent_goal_done = True
                return 0.5
            elif newletter == b'G' and self.adjacent_goal_done:
                return 1.0
            elif (newletter == b'H') and (reward in ['negative', 'negative_and_steps']):
                return -1.0
            elif (newrow != row or newcol != col) and newletter == b'F' and (reward in ['steps', 'negative_and_steps']):
                return 0.1
            else:
                return 0.0

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(n_actions):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                rew = get_reward(newletter, row, col, newrow, newcol)
                                li.append((1.0 / 3.0, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = get_reward(newletter, row, col, newrow, newcol)
                            li.append((1.0, newstate, rew, done))
                            # li.append((1.0/3.0, newstate, rew, done))

        super(ComplexActionSetCorridorEnv, self).__init__(n_state, n_actions, P, isd)

    def render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO.StringIO() if mode == 'ansi' else sys.stdout

        row_start, col_start = self.start_s // self.ncol, self.start_s % self.ncol
        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row_start][col_start] = 'S'
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        if self.lastaction is not None:
            outfile.write("  ({})\n".format(self.get_action_meanings()[self.lastaction]))
        else:
            outfile.write("\n")

        outfile.write("\n".join("".join(row) for row in desc) + "\n")

        return outfile

    def get_action_meanings(self):
        return [["Left", "1-Down", "2-Down", "3-Down", "Right", "1-Up", "2-Up", "3-Up"][i] if i < 8 else "NoOp" for i in
                range(self.action_space.n)]


register(
    id='CorridorToy-v1',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '1x4',
        'n_actions': 4
    },
    max_episode_steps=100,
)

register(
    id='CorridorToy-v2',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '2x4',
        'n_actions': 4
    },
    max_episode_steps=100,
)

register(
    id='CorridorToy-v3',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '2x4',
        'n_actions': 4,
        'random_start': False,
        'reward_type': 'negative_hole',
        'is_slippery': False
    },
    max_episode_steps=100,
)

register(
    id='CorridorFLNonSkid-v0',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '4x4_fl',
        'n_actions': 4,
        'random_start': False,
        'reward_type': 'standard',
        'is_slippery': False
    },
    max_episode_steps=100,
)

register(
    id='CorridorFLNonSkid-v1',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '8x8_fl',
        'n_actions': 4,
        'random_start': True,
        'reward_type': 'standard',
        'is_slippery': False
    },
    max_episode_steps=20,
)

register(
    id='CorridorSmall-v1',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '4x4',
        'n_actions': 4,
        'random_start': False,
        'reward_type': 'standard',
        'is_slippery': False
    },
    max_episode_steps=100,
)

register(
    id='CorridorSmall-v2',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '4x4',
        'n_actions': 4,
        'reward_type': 'standard',
        'random_start': True,
        'is_slippery': False
    },
    max_episode_steps=100,
)

register(
    id='CorridorSmall-v3',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '4x4',
        'n_actions': 4,
        'reward_type': 'negative_hole',
        'random_start': True,
        'is_slippery': False
    },
    max_episode_steps=100,
)

register(
    id='CorridorSmall-v4',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '2x5',
        'n_actions': 4,
        'reward_type': 'negative_hole',
        'random_start': True,
        'is_slippery': False
    },
    max_episode_steps=100,
)

register(
    id='CorridorSmall-v5',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '2x11',
        'n_actions': 4,
        'reward_type': ['negative_hole', 'negative_steps'],
        'random_start': True,
        'is_slippery': False
    },
    max_episode_steps=100,
)

register(
    id='CorridorSmall-v6',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '5x5_risk',
        'n_actions': 4,
        'reward_type': ['negative_hole'],  # , 'negative_steps'],
        'random_start': False,
        'is_slippery': True
    },
    max_episode_steps=100,
)

register(
    id='CorridorSmall-v7',
    entry_point='corridor_emulator:CorridorEnv2',
    kwargs={
        'map_name': '5x5_risk',
        'n_actions': 4,
        'reward_type': ['negative_hole'],  # , 'negative_steps'],
        'random_start': True,
        'is_slippery': True
    },
    max_episode_steps=100,
)

register(
    id='CorridorSmall-v10',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '4x4',
        'n_actions': 10
    },
    max_episode_steps=100,
)
register(
    id='CorridorBig-v0',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '9x9',
        'n_actions': 4,
        'random_start': True,
        'reward_type': 'standard',
        'is_slippery': False
    },
    max_episode_steps=100,
)

register(
    id='CorridorBig-v5',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '9x9',
        'n_actions': 5
    },
    max_episode_steps=100,
)

register(
    id='CorridorBig-v10',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '9x9',
        'n_actions': 10
    },
    max_episode_steps=100,
)

register(
    id='MazeBig-v0',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '9x9_maze',
        'n_actions': 4,
        'random_start': True,
        'reward_type': 'standard',
        'is_slippery': False
    },
    max_episode_steps=100,
)

register(
    id='MazeBig-v1',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '9x9_maze',
        'n_actions': 4,
        'random_start': True,
        'reward_type': 'standard',
        'is_slippery': True
    },
    max_episode_steps=100,
)

register(
    id='MazeBig-v2',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': '11x10_maze_montezuma',
        'n_actions': 4,
        'random_start': False,
        'reward_type': 'negative_hole',
        'is_slippery': False
    },
    max_episode_steps=100,
)

register(
    id='CorridorActionTest-v0',
    entry_point='corridor_emulator:CorridorEnv',
    kwargs={
        'map_name': 'action_test',
        'n_actions': 4,
        'random_start': True,
        'reward_type': 'standard',
        'is_slippery': False
    },
    max_episode_steps=100,
)

register(
    id='CorridorActionTest-v1',
    entry_point='corridor_emulator:ComplexActionSetCorridorEnv',
    kwargs={
        'map_name': 'action_test',
        'n_actions': 8,
        'random_start': True,
        'reward_type': 'simple',
        'is_slippery': False
    },
    max_episode_steps=100,
)

MY_ENV_NAME = 'FrozenLakeNonskid8x8-v0'
try:
    spec(MY_ENV_NAME)
except:
    register(
        id=MY_ENV_NAME,
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '8x8', 'is_slippery': False},
        max_episode_steps=100,
        reward_threshold=0.78,  # optimum = .8196
    )

MY_ENV_NAME = 'FrozenLakeNonskid4x4-v0'
try:
    spec(MY_ENV_NAME)
except:
    register(
        id=MY_ENV_NAME,
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False},
        max_episode_steps=100,
        reward_threshold=0.78,  # optimum = .8196
    )

MY_ENV_NAME = 'FrozenLakeNonskid4x4-v1'
try:
    spec(MY_ENV_NAME)
except:
    register(
        id=MY_ENV_NAME,
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4_adj_goal', 'is_slippery': False},
        max_episode_steps=100,
        reward_threshold=0.78,  # optimum = .8196
    )


class PartiallyObservedCorridor(ObservationWrapper):
    def __init__(self, env=None):
        super(PartiallyObservedCorridor, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=5, shape=(1, 4))
        self.letter_id = {'S': 0, 'F': 1, 'H': 2, 'W': 3, 'G': 4, 'A': 5}

    def observation(self, obs):
        new_obs = np.zeros(self.observation_space.shape)

        e = self.env.unwrapped
        row = obs // e.ncol
        col = obs % e.ncol
        # North letter
        l = 'W' if row == 0 else e.desc[row - 1, col]
        new_obs[:, 0] = self.letter_id[l]
        # South letter
        l = 'W' if row == e.nrow - 1 else e.desc[row + 1, col]
        new_obs[:, 1] = self.letter_id[l]
        # East letter
        l = 'W' if col == e.ncol - 1 else e.desc[row, col + 1]
        new_obs[:, 2] = self.letter_id[l]
        # West letter
        l = 'W' if col == 0 else e.desc[row, col - 1]
        new_obs[:, 3] = self.letter_id[l]

        return new_obs


class OneHotObservation(ObservationWrapper):
    def __init__(self, env=None):
        super(OneHotObservation, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.env.observation_space.n,))

    def observation(self, obs):
        new_obs = np.zeros(self.env.observation_space.n)
        new_obs[int(obs)] = 1
        return new_obs

class FlattenObservation(ObservationWrapper):
    def __init__(self, env=None):
        super(FlattenObservation, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(np.prod(env.observation_space.shape),))

    def observation(self, obs):
        return obs.ravel()

class ObsStack(Wrapper):
    def __init__(self, env, k, axis=None):
        """Stack k last observations.
           If axis == None, create a new 0 dimension and concatenate along it
           Otherwise, concatenate observations along the given axis.
        """
        Wrapper.__init__(self, env)
        self.k = k
        self.axis = axis
        self.obs = deque([], maxlen=k)
        shp = list(env.observation_space.shape)
        dim = len(shp)
        if axis:
            assert axis < dim, "Axis {} is out of bounds for observations of dimension {}".format(axis, dim)
            self.stack = lambda x: np.concatenate(list(x), axis=axis)
            shp[axis] *= k
            self.observation_space = spaces.Box(low=0, high=255, shape=tuple(shp))
        else:
            self.stack = lambda x: np.stack(list(x), axis=0)
            self.observation_space = spaces.Box(low=0, high=255, shape=(k,) + tuple(shp))

    def reset(self):
        ob = self.env.unwrapped.reset()
        for _ in range(self.k):
            self.obs.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.obs.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.obs) == self.k
        return self.stack(self.obs)


class GymEnvironment(BaseEnvironment):
    def __init__(self, actor_id, game, seed, env_class=None, visualize=False, agent_history_length=1, random_start=False,
                 partially_observed=False):
        try:
            self.env = gym.make(game)
            try:
                self.desc = self.env.unwrapped.desc
            except:
                self.desc = None
        except (NameError, ImportError):
            assert env_class is not None, "The specified environment does not seem to be a registered Gym environment: env_class cannot be None."
            spec = registry.spec(game)
            self.env = env_class(**spec._kwargs)
            self.env.unwrapped._spec = spec
            self.desc = self.env.desc
            self.env = TimeLimit(self.env,
                                 max_episode_steps=self.env.spec.max_episode_steps,
                                 max_episode_seconds=self.env.spec.max_episode_seconds)
        self.env.seed(seed * (actor_id + 1))
        if partially_observed:
            self.env = PartiallyObservedCorridor(self.env)
        else:
            self.env = OneHotObservation(self.env)
        if agent_history_length > 1:
            self.env = ObsStack(self.env, agent_history_length)

        self.agent_history_length = agent_history_length

        self.num_actions = self.env.action_space.n
        self.gym_actions = list(range(self.env.action_space.n))
        self.visualize = visualize

        self.grid_shape = self.desc.shape

        self.game = game
        self.np_random, seed = seeding.np_random(seed)

    # def process_state(self, state):
    #     if isinstance(self.env.env, MyDiscreteEnv):
    #         s = np.zeros(self.env.observation_space.n)
    #         s[state] = 1
    #         return s
    #     return state

    @property
    def shape(self):
        return list(self.env.observation_space.shape)

    def get_legal_actions(self):
    	return self.gym_actions


    def action_name(self, action_id):
        if self.game in ['CorridorActionTest-v1']:
            return ["Left", "1-Down", "2-Down", "3-Down", "Right", "1-Up", "2-Up", "3-Up"][
                action_id] if action_id < 8 else "NoOp"
        else:
            return ["Left", "Down", "Right", "Up"][action_id] if action_id < 4 else "NoOp"

    def reset(self):
        s = self.env.reset()
        if self.visualize:
            self.env.render()
        return s  # self.process_state(s)

    def visualize_on(self):
        self.visualize = True

    def visualize_off(self):
        self.visualize = False

    def step(self, action):
        #s_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        s_t1, r_t, terminal, info = self.env.step(action)
        if self.visualize:
            self.env.render()
        # s_t1 = self.process_state(s_t1)
        return s_t1, r_t, terminal, info

    def get_noop(self):
        if self.game in ['CorridorActionTest-v1']:
            return 8
        else:
            return 4


    def reset_with_noops(self, noops=0):
        s = self.reset()
        if noops != 0:
            for _ in range(self.np_random.randint(0, noops)):
                s, _, _ = self.step(self.get_noop())
        return s
