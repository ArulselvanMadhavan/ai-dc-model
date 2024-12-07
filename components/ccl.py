import simpy
from utils import EventData, Dtypes

class Ccl:
    def __init__(self, env, num_devices):
        self.env = env
        self.tokens = simpy.Container(env, init=0, capacity=num_devices)

    @staticmethod
    def oot_msg():
        return f"Ccl - No tokens available. {self.tokens.level}/{self.tokens.capacity})"

    def all_reduce(self, payload, dtype, op):
        if (self.tokens.level + 1) < self.tokens.capacity:
            yield self.tokens.put(1)
            yield self.env.timeout(1, value=EventData([op, "all_reduce_init"], self.env.now))
        else:
            raise Exception(Ccl.oot_msg)
