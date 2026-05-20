from ortools.sat.python import cp_model
import time

class ThroughputOptimizer:
    """
    Railway Throughput Optimizer using Google OR-Tools CP-SAT Solver.
    """

    def optimize_train_schedule(self, trains, platforms=3, section_capacity=25, time_horizon=60):
        start_time = time.time()
        model = cp_model.CpModel()

        # decision variables
        # x[i, t] = 1 if train i starts at time t
        x = {}
        # scheduled[i] = 1 if train i is scheduled at all
        scheduled = {}

        for i, train in enumerate(trains):
            scheduled[i] = model.NewBoolVar(f'scheduled_{i}')
            arrival = train['arrival_time']
            duration = train['duration']

            # Potential start times (must be >= arrival)
            possible_starts = [t for t in range(time_horizon - duration + 1) if t >= arrival]

            for t in range(time_horizon - duration + 1):
                x[i, t] = model.NewBoolVar(f'x_{i}_{t}')
                if t < arrival:
                    model.Add(x[i, t] == 0)

            # Constraint: A train is scheduled if exactly one start time is chosen
            if not possible_starts:
                model.Add(scheduled[i] == 0)
                model.Add(sum(x[i, t] for t in range(time_horizon - duration + 1)) == 0)
            else:
                model.Add(sum(x[i, t] for t in possible_starts) == scheduled[i])
                # Ensure no other x[i, t] are set
                for t in range(time_horizon - duration + 1):
                    if t not in possible_starts:
                        model.Add(x[i, t] == 0)

        # 2. Block Occupancy Constraints (Capacity)
        for t in range(time_horizon):
            trains_at_t = []
            for i, train in enumerate(trains):
                duration = train['duration']
                for s in range(max(0, t - duration + 1), min(t + 1, time_horizon - duration + 1)):
                    trains_at_t.append(x[i, s])

            if trains_at_t:
                model.Add(sum(trains_at_t) <= section_capacity)

        # 3. Platform Constraints
        for t in range(time_horizon):
            trains_at_platform_at_t = []
            for i, train in enumerate(trains):
                duration = train['duration']
                for s in range(max(0, t - duration + 1), min(t + 1, time_horizon - duration + 1)):
                    trains_at_platform_at_t.append(x[i, s])

            if trains_at_platform_at_t:
                model.Add(sum(trains_at_platform_at_t) <= platforms)

        # Objective:
        # 1. Maximize scheduled trains (High reward)
        # 2. Minimize total weighted delay (Lower penalty)

        obj_terms = []
        for i, train in enumerate(trains):
            priority = train['priority']
            # Reward for scheduling
            obj_terms.append(scheduled[i] * 1000 * priority)

            arrival = train['arrival_time']
            for t in range(time_horizon - train['duration'] + 1):
                delay = t - arrival
                if delay > 0:
                    # Penalty for delay
                    obj_terms.append(x[i, t] * (-10 * delay * priority))

        model.Maximize(sum(obj_terms))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)

        scheduled_trains = []
        unresolved = 0

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for i, train in enumerate(trains):
                if solver.Value(scheduled[i]) == 1:
                    for t in range(time_horizon - train['duration'] + 1):
                        if solver.Value(x[i, t]) == 1:
                            scheduled_trains.append({
                                "id": train["id"],
                                "scheduled_start": t,
                                "delay": t - train["arrival_time"],
                                "duration": train["duration"],
                                "priority": train["priority"]
                            })
                            break
                else:
                    unresolved += 1
        else:
            unresolved = len(trains)

        return {
            "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE" if status == cp_model.FEASIBLE else "FAILED",
            "total_trains_scheduled": len(scheduled_trains),
            "scheduled_trains": scheduled_trains,
            "unresolved_conflicts": unresolved,
            "computation_time_ms": round((time.time() - start_time) * 1000, 2)
        }
