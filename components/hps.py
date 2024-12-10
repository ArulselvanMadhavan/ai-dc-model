import simpy
from utils import next_cid, MICRO, EventData, ComponentType

class Hps:
    def __init__(self, env, hps_rd_bw, hps_wr_bw):
        self.rd_bw = simpy.Container(env, init=0, capacity=hps_rd_bw)
        self.wr_bw = simpy.Container(env, init=0, capacity=hps_wr_bw)
        self.rd_cid = next_cid()
        self.wr_cid = next_cid()
        self.wr_buf = simpy.Store(env, capacity=hps_wr_bw)
        self.env = env

    def evt_data(self, name, is_read=False):
        return EventData(name, self.env.now, ComponentType.HPS,
                         self.rd_cid if is_read else self.wr_cid,
                         0 if is_read else 1)

    def read(self, payload, dtype, op):
        pass

    def write(self, payload, dtype, op):
        size_in_bytes = payload * dtype.byte_size()
        hps_write = op + ["hps_write"]
        while size_in_bytes > 0:
            available = self.wr_bw.capacity - self.wr_bw.level
            if available == 0:
                yield self.env.timeout(100000, value=None)
            elif (size_in_bytes > available):
                yield self.wr_bw.put(available)
                size_in_bytes = size_in_bytes - available
                wr_time = int((available / self.wr_bw.capacity) * MICRO)
                yield self.env.timeout(wr_time, value=self.evt_data(hps_write, is_read=False))
                yield self.wr_bw.get(available)
            else:
                yield self.wr_bw.put(size_in_bytes)
                wr_time = int((size_in_bytes / self.wr_bw.capacity) * MICRO)
                yield self.env.timeout(wr_time, value=self.evt_data(hps_write, is_read=False))
                yield self.wr_bw.get(size_in_bytes)
                size_in_bytes = 0
