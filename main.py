import random, heapq, pandas as pd, matplotlib.pyplot as plt
from collections import deque

class ClinicSimulation:
    def __init__(self, arrival_rate_per_hour=15,
                 service_rates_per_hour=[10, 12, 14],
                 sim_time_minutes=8*60, seed=None,
                 max_queue_length=None,
                 breaks=None):
        if seed is not None:
            random.seed(seed)
        self.arrival_rate = arrival_rate_per_hour / 60.0
        self.service_rates = [r/60.0 for r in service_rates_per_hour]
        self.num_doctors = len(service_rates_per_hour)
        self.sim_time = sim_time_minutes
        self.max_queue_length = max_queue_length
        self.breaks = breaks if breaks else [[] for _ in range(self.num_doctors)]

    def _next_interarrival(self):
        return random.expovariate(self.arrival_rate)

    def _service_time(self, doctor_idx):
        return random.expovariate(self.service_rates[doctor_idx])

    def run(self):
        EVENT_ARRIVAL = 'arrival'
        EVENT_DEPARTURE = 'departure'
        EVENT_BREAK_START = 'break_start'
        EVENT_BREAK_END = 'break_end'

        event_queue = []
        current_time = 0.0
        heapq.heappush(event_queue, (current_time + self._next_interarrival(), EVENT_ARRIVAL, None))

        doctors_busy = [False]*self.num_doctors
        doctors_on_break = [False]*self.num_doctors
        doctors_total_busy_time = [0.0]*self.num_doctors
        doctors_last_busy_start = [None]*self.num_doctors

        for i, doctor_breaks in enumerate(self.breaks):
            for bstart, bdur in doctor_breaks:
                heapq.heappush(event_queue, (bstart, EVENT_BREAK_START, i))
                heapq.heappush(event_queue, (bstart + bdur, EVENT_BREAK_END, i))

        queue = deque()
        records = []
        timeline = []
        customer_id = 0
        rejected_count = 0
        served_over_time = []
        rejected_over_time = []

        def find_free_doctor_rand():
            free_doctors = [i for i in range(self.num_doctors)
                            if not doctors_busy[i] and not doctors_on_break[i]]
            return random.choice(free_doctors) if free_doctors else None

        while event_queue:
            time, etype, info = heapq.heappop(event_queue)
            if time > self.sim_time:
                break
            current_time = time
            num_busy = sum(doctors_busy)
            timeline.append({'time': current_time,
                             'queue': len(queue),
                             'busy': num_busy,
                             'served': len(records),
                             'rejected': rejected_count})

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
                    if self.max_queue_length is None or len(queue) < self.max_queue_length:
                        queue.append((customer_id, arrival))
                    else:
                        rejected_count += 1
                        rejected_over_time.append((current_time, rejected_count))
                heapq.heappush(event_queue, (current_time + self._next_interarrival(), EVENT_ARRIVAL, None))

            elif etype == EVENT_DEPARTURE:
                cid, sidx = info
                depart = current_time
                doctors_busy[sidx] = False
                doctors_total_busy_time[sidx] += depart - doctors_last_busy_start[sidx]
                doctors_last_busy_start[sidx] = None
                if queue and not doctors_on_break[sidx]:
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
                served_over_time.append((current_time, len(records)))

            elif etype == EVENT_BREAK_START:
                sidx = info
                doctors_on_break[sidx] = True

            elif etype == EVENT_BREAK_END:
                sidx = info
                doctors_on_break[sidx] = False
                if queue and not doctors_busy[sidx]:
                    cid2, arr2 = queue.popleft()
                    st2 = self._service_time(sidx)
                    start2 = current_time
                    end2 = start2 + st2
                    doctors_busy[sidx] = True
                    doctors_last_busy_start[sidx] = start2
                    heapq.heappush(event_queue, (end2, EVENT_DEPARTURE, (cid2, sidx)))
                    wait = start2 - arr2
                    records.append({'id':cid2,'arrival':arr2,'start':start2,'end':end2,
                                    'wait':wait,'service':st2,'doctor':sidx})
                    served_over_time.append((current_time, len(records)))

        df = pd.DataFrame(records)
        timeline = pd.DataFrame(timeline)
        served_df = pd.DataFrame(served_over_time, columns=["time","served"])
        rejected_df = pd.DataFrame(rejected_over_time, columns=["time","rejected"])

        if df.empty:
            return df, {}, timeline, served_df, rejected_df
        metrics = {
            'num_served': len(df),
            'num_rejected': rejected_count,
            'avg_wait_min': df['wait'].mean(),
            'avg_service_min': df['service'].mean(),
            'avg_time_in_system': (df['end']-df['arrival']).mean(),
            'doctor_utilization': [doctors_total_busy_time[i]/self.sim_time for i in range(self.num_doctors)],
        }
        return df, metrics, timeline, served_df, rejected_df

    def plot_results(self, df, metrics, timeline, served_df, rejected_df, title="Symulacja", save=False):
        fig, axs = plt.subplots(3, 2, figsize=(12,12))
        fig.suptitle(title, fontsize=14)

        axs[0,0].plot(timeline['time'], timeline['queue'])
        axs[0,0].set_title("Długość Kolejki w Czasie")
        axs[0,0].set_xlabel("Czas (min)")
        axs[0,0].set_ylabel("Długość Kolejki")

        axs[0,1].plot(timeline['time'], timeline['busy'])
        axs[0,1].set_title("Zajęci Doktorzy w Czasie")
        axs[0,1].set_xlabel("Czas (min)")
        axs[0,1].set_ylabel("Zajęci Doktorzy")

        axs[1,0].bar(range(self.num_doctors), metrics['doctor_utilization'])
        axs[1,0].set_title("Czas Pracy Doktorów")
        axs[1,0].set_xlabel("Numer Doktora")
        axs[1,0].set_ylabel("Procent Czasu Pracy")

        axs[1,1].hist(df['wait'], bins=20)
        axs[1,1].set_title("Rozkład Czasu Oczekiwania")
        axs[1,1].set_xlabel("Czas Oczekiwania (min)")
        axs[1,1].set_ylabel("Ilość")

        axs[2,0].bar(['Obsłużeni','Odrzuceni'], [metrics['num_served'], metrics['num_rejected']],
                     color=['green','red'])
        axs[2,0].set_title("Obsłużeni vs Odrzuceni Pacjenci")
        axs[2,0].set_ylabel("Ilość")

        if not served_df.empty or not rejected_df.empty:
            if not served_df.empty:
                axs[2,1].plot(served_df['time'], served_df['served'], label='Obsłużeni', color='green')
            if not rejected_df.empty:
                axs[2,1].plot(rejected_df['time'], rejected_df['rejected'], label='Odrzuceni', color='red')
            axs[2,1].set_title("Obsłużeni i Odrzuceni Pacjenci w Czasie")
            axs[2,1].set_xlabel("Czas (min)")
            axs[2,1].set_ylabel("Suma")
            axs[2,1].legend()

        plt.tight_layout()

        if save:
            filename = f"wykres_{title}.png"
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            print(f"✅ Wykres zapisany jako: {filename}")
            plt.close(fig)
        else:
            plt.show()


def run_experiment(description, arrival_rate, breaks):
    print(f"\n{'='*80}\n{description}\n{'='*80}")
    sim = ClinicSimulation(
        arrival_rate_per_hour=arrival_rate,
        service_rates_per_hour=[6, 8, 10],
        sim_time_minutes=8*60,
        seed=42,
        max_queue_length=20,
        breaks=breaks
    )
    df, metrics, timeline, served_df, rejected_df = sim.run()
    print("Wyniki:", metrics)
    sim.plot_results(df, metrics, timeline, served_df, rejected_df,
                     title=description, save=True)
    return metrics


same_breaks = [[(240, 20)], [(240, 20)], [(240, 20)]]
staggered_breaks = [[(180, 20)], [(240, 20)], [(300, 20)]]

long_same_breaks = [[(240, 40)], [(240, 40)], [(240, 40)]]
long_staggered_breaks = [[(180, 40)], [(240, 40)], [(300, 40)]]

#m1 = run_experiment("Wszyscy lekarze mają przerwę w tym samym czasie (20 minut)", 30, same_breaks)
#m2 = run_experiment("Lekarze mają przesunięte przerwy (20 minut)", 30, staggered_breaks)
#m1 = run_experiment("Wszyscy lekarze mają przerwę w tym samym czasie (40 minut)", 30, long_same_breaks)
#m2 = run_experiment("Lekarze mają przesunięte przerwy (40 minut)", 30, long_staggered_breaks)
#m3 = run_experiment("Dużo pacjentów", 40, staggered_breaks)
#m4 = run_experiment("Mało pacjentów", 20, staggered_breaks)

def compare_experiment(description, arrival_rate, breaks_with, breaks_without):
    """
    Porównuje dwa scenariusze (z przerwami i bez) na wspólnych wykresach.
    """
    print(f"\n{'='*80}\n{description}\n{'='*80}")

    # Scenariusz z przerwami
    sim_with = ClinicSimulation(
        arrival_rate_per_hour=arrival_rate,
        service_rates_per_hour=[6, 8, 10],
        sim_time_minutes=8*60,
        seed=42,
        max_queue_length=20,
        breaks=breaks_with
    )
    df_with, metrics_with, timeline_with, served_with, rejected_with = sim_with.run()

    # Scenariusz bez przerw
    sim_without = ClinicSimulation(
        arrival_rate_per_hour=arrival_rate,
        service_rates_per_hour=[6, 8, 10],
        sim_time_minutes=8*60,
        seed=42,
        max_queue_length=20,
        breaks=breaks_without
    )
    df_no, metrics_no, timeline_no, served_no, rejected_no = sim_without.run()

    # ----- Tworzenie wykresów porównawczych -----
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(description, fontsize=14)

    # Długość kolejki
    axs[0,0].plot(timeline_with['time'], timeline_with['queue'], label='Z przerwami')
    axs[0,0].plot(timeline_no['time'], timeline_no['queue'], label='Bez przerw', linestyle='--')
    axs[0,0].set_title("Długość kolejki w czasie")
    axs[0,0].set_xlabel("Czas (min)")
    axs[0,0].set_ylabel("Długość kolejki")
    axs[0,0].legend()

    # Zajęci lekarze
    axs[0,1].plot(timeline_with['time'], timeline_with['busy'], label='Z przerwami')
    axs[0,1].plot(timeline_no['time'], timeline_no['busy'], label='Bez przerw', linestyle='--')
    axs[0,1].set_title("Zajęci lekarze w czasie")
    axs[0,1].set_xlabel("Czas (min)")
    axs[0,1].set_ylabel("Liczba zajętych lekarzy")
    axs[0,1].legend()

    # Obsłużeni pacjenci w czasie
    axs[1,0].plot(served_with['time'], served_with['served'], label='Z przerwami', color='green')
    axs[1,0].plot(served_no['time'], served_no['served'], label='Bez przerw', color='blue', linestyle='--')
    axs[1,0].set_title("Liczba obsłużonych pacjentów w czasie")
    axs[1,0].set_xlabel("Czas (min)")
    axs[1,0].set_ylabel("Obsłużeni")
    axs[1,0].legend()

    # Odrzuceni pacjenci w czasie
    axs[1,1].plot(rejected_with['time'], rejected_with['rejected'], label='Z przerwami', color='red')
    axs[1,1].plot(rejected_no['time'], rejected_no['rejected'], label='Bez przerw', color='orange', linestyle='--')
    axs[1,1].set_title("Liczba odrzuconych pacjentów w czasie")
    axs[1,1].set_xlabel("Czas (min)")
    axs[1,1].set_ylabel("Odrzuceni")
    axs[1,1].legend()

    plt.tight_layout()
    filename = f"porownanie_{description}.png"
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"✅ Wykres zapisany jako: {filename}")

    # ----- Wyświetlenie podstawowych metryk -----
    print("\n📊 Wyniki porównania:")
    print(f"{'':<25} {'Z przerwami':<15} {'Bez przerw':<15}")
    print(f"{'Obsłużeni':<25} {metrics_with['num_served']:<15} {metrics_no['num_served']:<15}")
    print(f"{'Odrzuceni':<25} {metrics_with['num_rejected']:<15} {metrics_no['num_rejected']:<15}")
    print(f"{'Śr. czas oczekiwania [min]':<25} {metrics_with['avg_wait_min']:<15.2f} {metrics_no['avg_wait_min']:<15.2f}")
    print(f"{'Śr. czas w systemie [min]':<25} {metrics_with['avg_time_in_system']:<15.2f} {metrics_no['avg_time_in_system']:<15.2f}")

    return metrics_with, metrics_no


# -------------------------------------------------------------
# 🔸 Porównanie dwóch ostatnich przypadków: dużo i mało pacjentów
# -------------------------------------------------------------
no_breaks = [[] for _ in range(3)]  # brak przerw

compare_experiment("Dużo Pacjentów", 40,
                   breaks_with=staggered_breaks,
                   breaks_without=no_breaks)

compare_experiment("Mało Pacjentów", 20,
                   breaks_with=staggered_breaks,
                   breaks_without=no_breaks)