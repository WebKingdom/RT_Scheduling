import sys
import math
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd

# df = pd.DataFrame([
#     dict(Task="Job A", Start=0, Finish=2),
#     dict(Task="Job B", Start=2, Finish=3),
#     dict(Task="Job A", Start=10, Finish=12),
#     dict(Task="Job B", Start=15, Finish=16)
# ])


# Set debug level higher to output more information
DISABLED = 0
LOW = 1
MEDIUM = 2
HIGH = 3
DEBUG_LEVEL = HIGH


# TimeFrame class for holding a start and end time
class TimeFrame:
    def __init__(self, start_t=0, end_t=0):
        self.start_t = start_t
        self.end_t = end_t

    def __str__(self):
        return f"({self.start_t}, {self.end_t})"


# Task class for holding task name, ci, pi, and time frames
class Task:
    # constructor with optional start and end times
    def __init__(self, name, exec_t, period):
        self.name = name
        self.exec_t = exec_t
        self.period = period
        self.time_frames = []

    def getDict(self, index):
        tf = self.time_frames[index]
        if DEBUG_LEVEL >= HIGH: print(tf)
        return dict(Task=self.name, Start=tf.start_t, Finish=tf.end_t)

    def __str__(self):
        return f"Task {self.name}: ({self.exec_t}, {self.period}). TimeFrames: {self.time_frames}"


# Error message printer
def print_error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# RMS utilization test returns true if task set is schedulable, false otherwise (sufficient but not necessary)
def rms_utilization_test(tasks) -> bool:
    n = len(tasks)
    limit = float(n * (2.0 ** (1.0 / n) - 1.0))
    sum = 0.0
    for task in tasks:
        sum += float(task.exec_t) / float(task.period)
    return sum <= limit

# RMS exact analysis test (completion time test) return true if task set is schedulable, false otherwise
def rms_exact_analysis_test(tasks) -> bool:
    # iterate over tasks from lowest to highest priority (period) and workload will only include subset of tasks that have not passed the test yet
    for i in range(len(tasks) - 1, -1, -1):
        task = tasks[i]
        cur_time = 0.0
        workload = compute_workload(tasks[0:i+1], cur_time, initial_workload=True)
        while ((cur_time != workload) and (workload <= task.period)):
            cur_time = workload
            workload = compute_workload(tasks[0:i+1], cur_time)

        if (cur_time == workload):
            if DEBUG_LEVEL >= MEDIUM:
                print(str(task) + " schedulable by exact analysis due to: " +
                      str(cur_time) + " == " + str(workload))
        elif (workload > task.period):
            if DEBUG_LEVEL >= MEDIUM:
                print(str(task) + " NOT schedulable by exact analysis due to: " +
                      str(workload) + " > " + str(task.period))
            return False
    return True


# computes the workload for the given task set at the given time
def compute_workload(tasks, cur_time, initial_workload=False) -> float:
    workload = 0.0
    for task in tasks:
        if (initial_workload):
            workload += float(task.exec_t)
        else:
            workload += float(task.exec_t) * float(math.ceil(float(cur_time) / float(task.period)))
    return workload


# EDF utilization test return true if task set is shedulable, false otherwise
def edf_utilization_test(tasks) -> bool:
    sum = 0.0
    for task in tasks:
        sum += float(task.exec_t) / float(task.period)
    return sum <= 1.0


# Returns the LCM of all task periods
def get_lcm_period(tasks) -> int:
    n = len(tasks)
    if (n <= 0):
        print_error("Error: len(tasks) <= 0. Must have 1+ tasks for scheduling.")
        return -1
    lcm = tasks[0].period
    for i in range(1, n):
        lcm = lcm * tasks[i].period // math.gcd(lcm, tasks[i].period)
    return lcm


# generates the RMS schedule for the given task set
def generate_rms_schedule(tasks):
    n = len(tasks)
    if (n <= 0):
        print_error("Error: len(tasks) <= 0. Must have 1+ tasks for scheduling.")
        return -1
    tasks = sorted(tasks, key=lambda task: task.period)
    sched_util_test = rms_utilization_test(tasks)
    sched_exact_test = rms_exact_analysis_test(tasks)
    print("RMS utilization test: " + str(sched_util_test))
    print("RMS exact analysis test: " + str(sched_exact_test))
    
    if (sched_util_test or sched_exact_test):
        # current time and task index
        cur_t = 0
        ti = 0
        deferred_tasks = []
        # array of length n holding (cur_t / period) at the corresponding task index
        multiples = [0] * n
        lcm = get_lcm_period(tasks)
        if DEBUG_LEVEL >= MEDIUM: print("LCM = " + str(lcm))
        
        while (cur_t < lcm):
            end_t = cur_t + tasks[ti].exec_t
            # update multiples
            for i in range(0, len(multiples)):
                new_multiple = end_t / tasks[i]
                if (new_multiple != multiples[i] and i < ti):
                    print("Task at index: " + str(i) + " should be scheduled instead of: " + str(ti))
                    deferred_tasks.append(tasks[ti])
                    # TODO
                
            if (ti == 0):
                # highest priority task must be scheduled
                tasks[ti].time_frames.append(TimeFrame(cur_t, end_t))
                cur_t = end_t
                ti = (ti + 1) % len(tasks)
            else:
                # TODO schedule current task based on next task's start time and its priority
                print("TODO")
                
            
    else:
        print("Task set is not schedulable by RMS.")


# generates the EDF schedule for the given task set
def generate_edf_schedule(tasks):
    tasks = sorted(tasks, key=lambda task: task.period)
    sched_exact_test = edf_utilization_test(tasks)
    print("EDF utilization test: " + str(sched_exact_test))
    # TODO


# get tasks and sort from high to low priority
# tasks = [Task("T1", 1, 8), Task("T2", 2, 6), Task("T3", 4, 24)]
# tasks = [Task("T1", 3, 12), Task("T2", 3, 12), Task("T3", 8, 16)]
tasks = [Task("T1", 2, 8), Task("T2", 3, 12), Task("T3", 4, 16)]

generate_rms_schedule(tasks)
generate_edf_schedule(tasks)

# A = (2, 10), B = (1, 15) where (ci, pi)
# tasks = [Task("T1", 2, 10), Task("T2", 1, 15)]

# tasks[0].time_frames.append(TimeFrame(0, 2))
# tasks[1].time_frames.append(TimeFrame(2, 3))

# tasks[0].time_frames.append(TimeFrame(10, 12))
# tasks[1].time_frames.append(TimeFrame(15, 16))

# dfs = []
# # for each task in tasks
# for task in tasks:
#     if DEBUG_LEVEL >= HIGH: print(task)
#     # add all time frames in the task to the DataFrame.
#     for i in range(0, len(task.time_frames)):
#         cur_dict = task.getDict(i)
#         if DEBUG_LEVEL >= MEDIUM: print(cur_dict)
#         dfs.append(pd.DataFrame([cur_dict]))

# df = pd.concat(dfs)
# if DEBUG_LEVEL >= LOW: print(df)

# # index_col='Resource',
# fig = ff.create_gantt(df, index_col='Task', showgrid_x=True,
#                       bar_width=0.4, show_colorbar=True, group_tasks=True)
# fig.update_layout(xaxis_type='linear', autosize=False, width=800, height=400)
# fig.show()
