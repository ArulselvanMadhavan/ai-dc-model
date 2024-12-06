import simpy

class HPS:
    def __init__(self, env):
        self.rd_bw = simpy.Container(env, init=0, capacity=1000)
        self.wr_bw = simpy.Container(env, init=0, capacity=500)
