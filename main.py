import simpy
from components.xpu import Xpu, Dtypes
from trace import trace, monitor
from functools import partial

if __name__ == "__main__":
    env = simpy.Environment()
    data = []
    monitor = partial(monitor, data)
    trace(env, monitor)

    xpu = Xpu(env, fp32_gflops=989000)
    # 32x512x12288 12288x12288
    B = 32
    S = 512
    E = 12288
    env.process(xpu.matmul(B*S, E, E))
    env.run()

    for d in data:
        evt = d[2]
        if isinstance(evt, simpy.events.Timeout) and evt.value is not None:
            print(d[0], evt.value)
