from collections import namedtuple, deque
import random

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state"])
# class Transition(namedtuple("Transition", ["state", "action", "reward", "next_state"])):
#     def __getstate__(self):
#         # Serialize the namedtuple as a dictionary
#         return self._asdict()

#     def __setstate__(self, state):
#         # Recreate the namedtuple from the dictionary
#         self.__init__(**state)

TransitionV2 = namedtuple("TransitionV2", ["state", 'alpha','beta', "action", "reward", "next_state"])
# class TransitionV2(namedtuple("TransitionV2", ["state", 'alpha','beta', "action", "reward", "next_state"])):
#     def __getstate__(self):
#         # Serialize the namedtuple as a dictionary
#         return self._asdict()

#     def __setstate__(self, state):
#         # Recreate the namedtuple from the dictionary
#         self.__init__(**state)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ReplayMemoryV2(object):
    
        def __init__(self, capacity):
            self.memory = deque([], maxlen=capacity)
    
        def push(self, *args):
            """Save a transition"""
            self.memory.append(TransitionV2(*args))
    
        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)
    
        def __len__(self):
            return len(self.memory)