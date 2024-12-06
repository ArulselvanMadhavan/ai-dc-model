import simpy
from components.xpu import Xpu, Dtypes, XpuSpecs
from trace import trace, monitor
from functools import partial

if __name__ == "__main__":
    env = simpy.Environment()
    data = []
    monitor = partial(monitor, data)
    trace(env, monitor)

    h100_specs = XpuSpecs((989000, 0.5), (3350, 0.7), (80, 0.85))
    h100 = Xpu(env, h100_specs)

    B = 32
    S = 512
    E = 12288
    dtype = Dtypes.FP16

    def wrapper():
        yield env.process(h100.mem_fill(E*E, dtype, "ExE"))
        yield env.process(h100.mem_fill(B*S*E, dtype, "BxSxE"))
        yield env.process(h100.matmul(B*S, E, E))

    env.process(wrapper())
    env.run()

    print(h100.mem_rem())
    for d in data:
        evt = d[2]
        if isinstance(evt, simpy.events.Timeout) and evt.value is not None:
            print(d[0], evt.value)
