import random, heapq, pandas as pd
from collections import deque

class ClinicSimulationHetero:
    def __init__(self, arrival_rate_per_hour=15, 
                 service_rates_per_hour=[10, 12, 14],
                 sim_time_minutes=8*60, seed=None):
        if seed is not None:
            random.seed(seed)
        self.arrival_rate = arrival_rate_per_hour / 60.0
        self.service_rates = [r/60.0 for r in service_rates_per_hour]
        self.num_doctors = len(service_rates_per_hour)
        self.sim_time = sim_time_minutes

    def _next_interarrival(self):
        return random.expovariate(self.arrival_rate)

    def _service_time(self, doctor_idx):
        rate = self.service_rates[doctor_idx]
        return random.expovariate(rate)

    def run(self):
        EVENT_ARRIVAL = 'arrival'
        EVENT_DEPARTURE = 'departure'
        event_queue = []
        current_time = 0.0

        heapq.heappush(event_queue, (current_time + self._next_interarrival(), EVENT_ARRIVAL, 0))
        doctors_busy = [False]*self.num_doctors
        doctors_total_busy_time = [0.0]*self.num_doctors
        doctors_last_busy_start = [None]*self.num_doctors
        queue = deque()
        records = []
        customer_id = 0

        def find_free_doctor():
            for i, busy in enumerate(doctors_busy):
                if not busy:
                    return i
            return None

        def find_free_doctor_rand():
            free_doctors = [i for i, busy in enumerate(doctors_busy) if not busy]
            return random.choice(free_doctors) if free_doctors else None

        while event_queue:
            time, etype, info = heapq.heappop(event_queue)
            if time > self.sim_time:
                break
            current_time = time

            if etype == EVENT_ARRIVAL:
                customer_id += 1
                arrival = current_time
                free = find_free_doctor_rand()
                if free is not None:
                    st = self._service_time(free)
                    start = current_time
                    end = start + st
                    doctors_busy[free] = True
                    doctors_last_busy_start[free] = start
                    heapq.heappush(event_queue, (end, EVENT_DEPARTURE, (customer_id, free)))
                    records.append({'id':customer_id,'arrival':arrival,'start':start,'end':end,
                                    'wait':0.0,'service':st,'doctor':free})
                else:
                    queue.append((customer_id, arrival))
                heapq.heappush(event_queue, (current_time + self._next_interarrival(), EVENT_ARRIVAL, 0))

            elif etype == EVENT_DEPARTURE:
                cid, sidx = info
                depart = current_time
                doctors_busy[sidx] = False
                doctors_total_busy_time[sidx] += depart - doctors_last_busy_start[sidx]
                doctors_last_busy_start[sidx] = None
                if queue:
                    cid2, arr2 = queue.popleft()
                    st2 = self._service_time(sidx)
                    start2 = depart
                    end2 = start2 + st2
                    doctors_busy[sidx] = True
                    doctors_last_busy_start[sidx] = start2
                    heapq.heappush(event_queue, (end2, EVENT_DEPARTURE, (cid2, sidx)))
                    wait = start2 - arr2
                    records.append({'id':cid2,'arrival':arr2,'start':start2,'end':end2,
                                    'wait':wait,'service':st2,'doctor':sidx})

        df = pd.DataFrame(records)
        if df.empty:
            return df, {}
        metrics = {
            'num_served': len(df),
            'avg_wait_min': df['wait'].mean(),
            'avg_service_min': df['service'].mean(),
            'avg_time_in_system': (df['end']-df['arrival']).mean(),
            'doctor_utilization': [doctors_total_busy_time[i]/self.sim_time for i in range(self.num_doctors)]
        }
        return df, metrics


sim = ClinicSimulationHetero(
    arrival_rate_per_hour=36,
    service_rates_per_hour=[6, 8, 12],
    sim_time_minutes=8*60,
    seed=42
)
df, metrics = sim.run()

print(metrics)
print(df)