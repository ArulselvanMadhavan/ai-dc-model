import simpy
from utils import EventData, Dtypes, MICRO

class Ccl:
    def __init__(self, env, num_devices, bws):
        self.env = env
        self.tokens = simpy.Container(env, init=0, capacity=num_devices)
        self.bws = bws

    def evt_data(self, name):
        return EventData(name, self.env.now, "ccl")
    @staticmethod
    def oot_msg():
        return f"Ccl - No tokens available. {self.tokens.level}/{self.tokens.capacity})"

    def all_reduce(self, payload, dtype, op):
        if self.tokens.level < self.tokens.capacity:
            yield self.tokens.put(1)
            yield self.env.timeout(1, value=self.evt_data([op, "all_reduce_init"]))
            while True:
                if self.tokens.level == self.tokens.capacity:
                    size_in_bytes = payload * dtype.byte_size()
                    # Assume only high BW domain is used
                    comm_time = int((size_in_bytes / self.bws[0]) * MICRO)
                    yield self.env.timeout(comm_time, value=self.evt_data([op, "all_reduce_comm"]))
                    yield self.tokens.get(1)
                    yield self.env.timeout(1, value=self.evt_data([op, "all_reduce_complete"]))
                    break
                else:
                    yield self.env.timeout(1)
        else:
            raise Exception(Ccl.oot_msg)
