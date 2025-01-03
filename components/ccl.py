import simpy
from utils import *
from typing import Tuple
import numpy as np
from enum import Enum

class CollType(Enum):
    ALLREDUCE=1
    ALLGATHER=2
    SEND=3

class Ccl:
    def __init__(self, env, dev_split:Tuple[int, int],
                 bw_split:Tuple[int, int],
                 bw_eff:Tuple[int, int]):
        self.env = env
        self.dev_split = []
        num_devices = np.prod(dev_split)
        self.tokens = simpy.Container(env, init=0, capacity=num_devices)
        self.bw_split = [bw * eff for bw, eff in zip(bw_split, bw_eff)]
        self.cid = next_cid()
        self.dev_split = dev_split
        self.tid = next_ccl_id()
        self.comm_time = 0

    def evt_data(self, name):
        return EventData(name, self.env.now, ComponentType.CCL, self.cid, self.tid)

    @staticmethod
    def oot_msg():
        return f"Ccl - No tokens available. {self.tokens.level}/{self.tokens.capacity})"

    def send_comm_time(self, size_in_bytes):
        # Each PP group has TP=(n*HB) devices
        # Each HB has low bw conn
        # so, in theory , you can leverage n * low_bw conn to send BxSxE
        comm_lbw = int((size_in_bytes / self.bw_split[1]) * MICRO)
        return comm_lbw

    def all_reduce_comm_time(self, size_in_bytes):
        per_device_chunk = size_in_bytes / np.prod(self.dev_split)
        payload_hbw = per_device_chunk * self.dev_split[1]
        if self.bw_split[0] == 0:
            comm_hbw = 0
        else:
            comm_hbw = int((payload_hbw / self.bw_split[0]) * MICRO)
        # Assume ring in scale-out
        payload_lbw = per_device_chunk
        num_steps = self.dev_split[1] - 1
        comm_lbw = int((payload_lbw / self.bw_split[1]) * MICRO)
        comm_lbw = comm_lbw * num_steps
        comm_time = comm_hbw + comm_lbw
        # print("BxSxE(in GB):", size_in_bytes/GIGA)
        # print("payload(hbw):", payload_hbw / GIGA)
        # print("comm_time(hbw):", comm_hbw/MICRO)
        # print("payload(lbw):", payload_lbw / GIGA)
        # print("comm_time(lbw):", comm_lbw/MICRO)
        # print("total:", comm_time / MICRO)
        # all-gather lbw
        # reduce-scatter-hbw
        # 1 2 3 4 - [1p, 5p] [2p, 6p] [3p, 7p] [4p, 8p]
        # 5 6 7 8 - [5p, 1p] [6p, 2p] [7p, 3p] [8p, 4p]
        # reduce-scatter - lbw
        # 1 2 3 4 - [1] [2] [3] [4]
        # 5 6 7 8 - [5] [6] [7] [8]
        # all-gather - lbw
        # 1 2 3 4 - [1 5] [2 6] [3 7] [4 8]
        # 5 6 7 8 - [5 1] [6 2] [7 3] [8 4]
        # all-gather - hbw
        # 1 2 3 4 - [1 5 2 6 3 7 4 8]
        # 5 6 7 8
        return 2 * comm_time

    def all_gather_comm_time(self, size_in_bytes):
        # Assume that all-gather is only done in high bw domain - all2all
        return int((size_in_bytes / self.bw_split[0]) * MICRO)

    def run_collective(self, payload, dtype, op, coll_type):
        if self.tokens.level < self.tokens.capacity:
            yield self.tokens.put(1)
            yield self.env.timeout(1, value=self.evt_data(op + ["init"]))
            # while True:
            #     if self.tokens.level == self.tokens.capacity:
            size_in_bytes = payload * dtype.byte_size()
            match coll_type:
                case CollType.ALLREDUCE:
                    comm_time = self.all_reduce_comm_time(size_in_bytes)
                case CollType.ALLGATHER:
                    comm_time = self.all_gather_comm_time(size_in_bytes)
                case CollType.SEND:
                    comm_time = self.send_comm_time(size_in_bytes)
            self.comm_time += comm_time
            yield self.env.timeout(comm_time, value=self.evt_data(op + ["comm"]))
            yield self.tokens.get(1)
            yield self.env.timeout(1, value=self.evt_data(op + ["complete"]))
                #     break
                # else:
                #     print("Waiting?", self.tokens.level, self.tokens.capacity, op)
                #     yield self.env.timeout(100, value=None)
        else:
            print("PP:", op, self.tokens.level, self.tokens.capacity)
            raise Exception(Ccl.oot_msg)

    def all_reduce(self, payload, dtype, op):
        return self.run_collective(payload, dtype, op + ["all_reduce"], CollType.ALLREDUCE)

    def all_gather(self, payload, dtype, op):
        return self.run_collective(payload, dtype, op + ["all_gather"], CollType.ALLGATHER)

    def send(self, payload, dtype, op):
        return self.run_collective(payload, dtype, op + ["send"], CollType.SEND)
