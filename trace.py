from functools import partial, wraps
import simpy
import numpy as np
from utils import ComponentType, EventData, CounterData, get_cid

def trace(env, callback):
    """Replace the ``step()`` method of *env* with a tracing function
    that calls *callbacks* with an events time, priority, ID and its
    instance just before it is processed.

    """
    def get_wrapper(env_step, callback):
        """Generate the wrapper for env.step()."""
        @wraps(env_step)
        def tracing_step():
            """Call *callback* for the next event if one exist before
            calling ``env.step()``."""
            if len(env._queue):
                t, prio, eid, event = env._queue[0]
                callback(t, prio, eid, event)
            return env_step()
        return tracing_step

    env.step = get_wrapper(env.step, callback)

def monitor(data, t, prio, eid, event):
    data.append((t, eid, event))

def dump_perfetto(component_types, component_mat, data):
    from stubs.protos import trace_pb2
    import uuid

    def tpkt_tevt():
        tpkt = trace_pb2.TracePacket()
        tpkt.trusted_packet_sequence_id = 0
        tevt = trace_pb2.TrackEvent()
        return tpkt, tevt

    def random_int64():
        return uuid.uuid4().int & (1 << 64) - 1

    def track_descriptors(component_types, component_mat, pkts):
        uuids = []
        thread_count = 0
        for c_id, ctype in enumerate(component_types):
            # Define proc
            tpkt = trace_pb2.TracePacket()
            tpkt.trusted_packet_sequence_id = 0
            tdesc = trace_pb2.TrackDescriptor()
            proc_uuid = random_int64()
            tdesc.uuid = proc_uuid
            pdesc = tdesc.process
            pdesc.pid = c_id
            pdesc.process_name = ctype
            tpkt.track_descriptor.CopyFrom(tdesc)
            pkts.append(tpkt)
            # define threads
            comp_list = component_mat[c_id]
            for _, component in enumerate(comp_list):
                thread_uuid = random_int64()
                uuids.append(thread_uuid)
                tpkt = trace_pb2.TracePacket()
                tpkt.trusted_packet_sequence_id = 0
                tdesc = trace_pb2.TrackDescriptor()
                tdesc.uuid = thread_uuid
                tdesc.parent_uuid = proc_uuid
                tdesc.name = component
                if "ctr_" in component:
                    tdesc.counter.CopyFrom(trace_pb2.CounterDescriptor())
                else:
                    tdesc.thread.pid = c_id
                    tdesc.thread.tid = thread_count
                    tdesc.thread.thread_name = component
                tpkt.track_descriptor.CopyFrom(tdesc)
                pkts.append(tpkt)
                thread_count += 1
        return uuids

    def get_timeout_evts(data):
        evts = []
        start_times = []
        end_times = []
        cevts = []
        for d in data:
            event = d[2]
            if isinstance(event, simpy.events.Timeout) and isinstance(event.value, EventData):
                evt = event.value
                evts.append(evt)
                start_times.append(evt.start_time)
                end_times.append(d[0])
            elif isinstance(event, simpy.events.Timeout) and isinstance(event.value, CounterData):
                evt = event.value
                cevts.append((d[0], evt))
        # group by start_times
        def sort_events(evts, etimes):
            ev_groups = {}
            et_groups = {}
            for i, e in enumerate(evts):
                if ev_groups.get(e.start_time, None) is None:
                    ev_groups[e.start_time] = [e]
                    et_groups[e.start_time] = [end_times[i]]
                else:
                    ev_groups[e.start_time].append(e)
                    et_groups[e.start_time].append(end_times[i])
            evt_out = []
            for k, etg in et_groups.items():
                evg = ev_groups[k]
                evg = list(reversed(evg))
                etg = list(reversed(etg))
                scope_end = 0
                stack = []
                for i in range(0, len(etg)):
                    has_next = (i + 1) < len(etg)
                    et = etg[i]
                    if et > scope_end:
                        scope_end = et
                    evt_out.append(("BEGIN", k, evg[i]))
                    n_et = etg[i + 1] if has_next else None
                    if has_next is False:
                        evt_out.append(("END", et, evg[i]))
                    if n_et is not None:
                        if n_et == scope_end:
                            evt_out.append(("END", et, evg[i]))
                        elif n_et < scope_end:
                            stack.append((et, evg[i]))
                [evt_out.append(("END", et, ev)) for (et, ev) in reversed(stack)]
            return evt_out
        return sort_events(evts, end_times), cevts

    def track_events(pkts, uuids, data):
        count = 0
        evts, cevts = get_timeout_evts(data)
        for (evt_type, timestamp, evt) in evts:
            if evt.tid == 0 or evt.cty == ComponentType.CCL or evt.cty == ComponentType.HPS:
                thread_uuid = uuids[evt.cid]
                evt_name = "_".join(evt.name)
                tpkt, tevt = tpkt_tevt()
                tevt.track_uuid = thread_uuid
                tevt.name = evt_name
                tevt.type = trace_pb2.TrackEvent.Type.TYPE_SLICE_BEGIN if evt_type == "BEGIN" else trace_pb2.TrackEvent.Type.TYPE_SLICE_END
                tpkt.timestamp = timestamp
                tpkt.track_event.CopyFrom(tevt)
                pkts.append(tpkt)
        for (now, evt) in cevts:
            if evt.tid == 0:
                thread_uuid = uuids[evt.cid]
                tpkt, tevt = tpkt_tevt()
                tevt.track_uuid = thread_uuid
                tevt.type = trace_pb2.TrackEvent.Type.TYPE_COUNTER
                tevt.double_counter_value = evt.count
                tpkt.timestamp = now
                tpkt.track_event.CopyFrom(tevt)
                pkts.append(tpkt)

    trace = trace_pb2.Trace()
    track_uuids = track_descriptors(component_types, component_mat, trace.packet)
    track_events(trace.packet, track_uuids, data)

    with open("trace.perfetto", "wb+") as f:
        f.write(trace.SerializeToString())
