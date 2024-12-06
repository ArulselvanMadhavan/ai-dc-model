import simpy

class Link:
    def __init__(self, env, rd_bw, wr_bw):
        self.rsrc = simpy.Resource(env, capacity=1)
        self.rd_bw = simpy.Container(env, init=0, capacity=rd_bw)
        self.wr_bw = simpy.Container(env, init=0, capacity=wr_bw)
