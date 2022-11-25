import sys
import math
import copy
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
DEBUG_LEVEL = MEDIUM


# TimeFrame class for holding a start and end time
class TimeFrame:
    def __init__(self, start_t=0, end_t=0):
        self.start_t = start_t
        self.end_t = end_t

    def __str__(self):
        return f"({self.start_t}, {self.end_t})"


# Task class for holding task name, ci, pi, and time frames
class Task:
    def __init__(self, name: str, exec_t: int, period: int):
        self.name = name
        self.exec_t = exec_t
        self.remaining_t = exec_t
        self.period = period
        self.time_frames = []
        self.priority = -1

    def getDict(self, index: int):
        tf = self.time_frames[index]
        if DEBUG_LEVEL >= HIGH:
            print(tf)
        return dict(Task=self.name, Start=tf.start_t, Finish=tf.end_t)

    def __str__(self):
        return f"Task {self.name}: ({self.exec_t}, {self.period}). TimeFrames: {self.time_frames}"


# Preemption class for holding currently executing task, new task that will preempt the currently executing task, 
# and the current time
class Preemption():
    def __init__(self, exec_task: Task, new_task: Task, time: int):
        self.exec_task = exec_task
        self.new_task = new_task
        self.time = time

    def __str__(self):
        return f"Task {self.exec_task.name} preempted by {self.new_task.name} at time {self.time}"


# Error message printer
def print_error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# RMS utilization test returns true if task set is schedulable, false otherwise (sufficient but not necessary). Also sets priority of tasks from low (0) to high (n-1)
def rms_utilization_test(tasks: list[Task]) -> bool:
    n = len(tasks)
    limit = float(n * (2.0 ** (1.0 / n) - 1.0))
    sum = 0.0
    for i in range(0, n):
        sum += float(tasks[i].exec_t) / float(tasks[i].period)
        tasks[i].priority = i
    print("RMS Processor Utilization: " + str(sum * 100) + "%")
    return (sum <= limit, tasks)


# RMS exact analysis test (completion time test) return true if task set is schedulable, false otherwise
def rms_exact_analysis_test(tasks: list[Task]) -> bool:
    n = len(tasks)
    # iterate over tasks from lowest to highest priority (period) and workload will only include subset of tasks that have not passed the test yet
    for i in range(0, n):
        task = tasks[i]
        cur_time = 0.0
        workload = compute_workload(tasks[i:n], cur_time, initial_workload=True)
        while ((cur_time != workload) and (workload <= task.period)):
            cur_time = workload
            workload = compute_workload(tasks[i:n], cur_time)

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
def compute_workload(tasks: list[Task], cur_time: int, initial_workload=False) -> float:
    workload = 0.0
    for task in tasks:
        if (initial_workload):
            workload += float(task.exec_t)
        else:
            workload += float(task.exec_t) * float(math.ceil(float(cur_time) / float(task.period)))
    return workload


# EDF utilization test return true if task set is shedulable, false otherwise. Also sets priority of tasks from low (0) to high (n-1)
def edf_utilization_test(tasks: list[Task]) -> bool:
    sum = 0.0
    for i in range(0, len(tasks)):
        sum += float(tasks[i].exec_t) / float(tasks[i].period)
        tasks[i].priority = i
    print("EDF Processor Utilization: " + str(sum * 100) + "%")
    return (sum <= 1.0, tasks)


# Returns the LCM of all task periods
def get_lcm_period(tasks: list[Task]) -> int:
    n = len(tasks)
    if (n <= 0):
        print_error("Error: len(tasks) <= 0. Must have 1+ tasks for scheduling.")
        return -1
    lcm = tasks[0].period
    for i in range(1, n):
        lcm = lcm * tasks[i].period // math.gcd(lcm, tasks[i].period)
    return lcm


# inserts 1 task at the correct index into the task queue (ordered low to high priority)
def insert_task_q(task_q: list[Task], task: Task) -> list[Task]:
    task = copy.deepcopy(task)
    n = len(task_q)
    if (n == 0):
        task_q = [task]
        return task_q
    for i in range(n - 1, -1, -1):
        if (task.period < task_q[i].period):
            task_q.insert(i+1, task)
            return task_q
        elif (task.period == task_q[i].period):
            task_q.insert(i, task)
            return task_q


# !!! NOT USED: inserts the same priority tasks at the correct index into the task queue (ordered low to high priority)
def insert_tasks_q(task_q: list[Task], tasks: list[Task]) -> list[Task]:
    n = len(task_q)
    if (n == 0):
        task_q = tasks
        return task_q
    for i in range(n - 1, -1, -1):
        if (tasks[i].period < task_q[i].period):
            while (len(tasks) != 0):
                task_q.insert(i+1, tasks.pop())
            return task_q
        elif (tasks[i].period == task_q[i].period):
            while (len(tasks) != 0):
                task_q.insert(i, tasks.pop())
            return task_q


# generates the RMS schedule for the given task set
def generate_rms_schedule(tasks: list[Task]):
    n = len(tasks)
    if (n <= 0):
        print_error("Error: len(tasks) <= 0. Must have 1+ tasks for scheduling.")
        return -1
    # * must sort tasks from low (0) to high (len(tasks - 1)) priority so that utilization test can set priority
    tasks = sorted(tasks, key=lambda task: task.period, reverse=True)
    sched_util_test, tasks = rms_utilization_test(tasks)
    sched_exact_test = rms_exact_analysis_test(tasks)
    print("RMS utilization test: " + str(sched_util_test))
    print("RMS exact analysis test: " + str(sched_exact_test))
    preemptions = []

    if (sched_util_test or sched_exact_test):
        # current time and task index
        cur_t = 0
        # array of length n holding (cur_t / period) at the corresponding task index
        multiples = [0] * n
        task_q = copy.deepcopy(tasks)
        lcm = get_lcm_period(tasks)
        remainder = 0
        if DEBUG_LEVEL >= MEDIUM:
            print("LCM = " + str(lcm))

        while (cur_t < lcm):
            if (len(task_q) > 0):
                # pop highest priority task
                task = task_q[-1]
                end_t = cur_t + task.remaining_t

                # determine end_t due to higher priority tasks
                for i in range(0, len(multiples)):
                    new_multiple = int(end_t / tasks[i].period)
                    if (new_multiple != multiples[i]):
                        if (tasks[i].period < task.period):
                            # task has higher priority than current task being scheduled
                            # minimize end_t (current task will be preempted)
                            new_end_t = tasks[i].period * new_multiple
                            if (new_end_t < end_t):
                                end_t = new_end_t

                # add TimeFrame for current task, update remaining_t (if applicable), and update task_q
                tasks[task.priority].time_frames.append(TimeFrame(cur_t, end_t))
                remainder = task.remaining_t - (end_t - cur_t)
                cur_t = end_t
                if (remainder > 0):
                    task_q[-1].remaining_t = remainder
                    preemptions.append(Preemption(copy.deepcopy(task), Task("Temp", -1, -1), cur_t))
                elif (remainder == 0):
                    task_q.pop()
                else:
                    print_error("Error: remaining_t of task is less than 0.")
            else:
                # no task to schedule, increment time
                cur_t += 1
            # check if task multiples changed within new time frame and update task_q
            for i in range(0, len(multiples)):
                new_multiple = int(cur_t / tasks[i].period)
                if (new_multiple != multiples[i]):
                    # add to task_q and update multiples
                    task_q = insert_task_q(task_q, tasks[i])
                    multiples[i] = new_multiple
                    # update preemption new_task accordingly
                    if (remainder > 0):
                        preemptions[-1].new_task = copy.deepcopy(tasks[i])
    else:
        print("Task set is not schedulable by RMS.")
        return pd.DataFrame([dict(Task="Not RMS Schedulable", Start=0, Finish=0)])

    # create DataFrame used by Plotly
    dfs = []
    for task in tasks:
        if DEBUG_LEVEL >= HIGH:
            print(task)
        # add all time frames in the task to the DataFrame.
        for i in range(0, len(task.time_frames)):
            cur_dict = task.getDict(i)
            if DEBUG_LEVEL >= HIGH:
                print(cur_dict)
            dfs.append(pd.DataFrame([cur_dict]))
    df = pd.concat(dfs)
    if DEBUG_LEVEL >= LOW:
        print(df)
        print(str(len(preemptions)) + " preemptions.")
        for preemption in preemptions:
            print(preemption)
    return df


# generates the EDF schedule for the given task set
def generate_edf_schedule(tasks: list[Task]):
    # * must sort tasks from low (0) to high (len(tasks - 1)) priority so that utilization test can set priority
    tasks = sorted(tasks, key=lambda task: task.period, reverse=True)
    sched_exact_test, tasks = edf_utilization_test(tasks)
    print("EDF utilization test: " + str(sched_exact_test))
    # TODO

    dfs = []
    for task in tasks:
        if DEBUG_LEVEL >= HIGH:
            print(task)
        # add all time frames in the task to the DataFrame.
        for i in range(0, len(task.time_frames)):
            cur_dict = task.getDict(i)
            if DEBUG_LEVEL >= HIGH:
                print(cur_dict)
            dfs.append(pd.DataFrame([cur_dict]))
    df = pd.concat(dfs)
    if DEBUG_LEVEL >= LOW:
        print(df)
    return df


# get tasks and sort from high to low priority
# tasks = [Task("T1", 1, 8), Task("T2", 2, 6), Task("T3", 4, 24)]
# tasks = [Task("T1", 3, 12), Task("T2", 3, 12), Task("T3", 8, 16)]
tasks = [Task("T1", 2, 8), Task("T2", 3, 12), Task("T3", 5, 16), Task("T4", 4, 32), Task("T5", 6, 96)]
tasks_rms = copy.deepcopy(tasks)
tasks_edf = copy.deepcopy(tasks)

# show RMS schedule
df_rms = generate_rms_schedule(tasks_rms)
fig = ff.create_gantt(df_rms, index_col='Task', showgrid_x=True,
                      bar_width=0.4, show_colorbar=True, group_tasks=True)
fig.update_layout(xaxis_type='linear', autosize=False, width=800, height=400)
fig.show()

# show EDF schedule
# df_edf = generate_edf_schedule(tasks_edf)
# fig = ff.create_gantt(df_edf, index_col='Task', showgrid_x=True,
#                       bar_width=0.4, show_colorbar=True, group_tasks=True)
# fig.update_layout(xaxis_type='linear', autosize=False, width=800, height=400)
# fig.show()


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
