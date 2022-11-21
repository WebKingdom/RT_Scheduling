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
DEBUG_LEVEL = LOW

class TimeFrame:
    def __init__(self, start_t=0, end_t=0):
        self.start_t = start_t
        self.end_t = end_t

    def __str__(self):
        return f"({self.start_t}, {self.end_t})"


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


# A = (2, 10), B = (1, 15) where (ci, pi)
tasks = [Task("T1", 2, 10), Task("T2", 1, 15)]

tasks[0].time_frames.append(TimeFrame(0, 2))
tasks[1].time_frames.append(TimeFrame(2, 3))

tasks[0].time_frames.append(TimeFrame(10, 12))
tasks[1].time_frames.append(TimeFrame(15, 16))

dfs = []
# for each task in tasks
for task in tasks:
    if DEBUG_LEVEL >= HIGH: print(task)
    # add all time frames in the task to the DataFrame.
    for i in range(0, len(task.time_frames)):
        cur_dict = task.getDict(i)
        if DEBUG_LEVEL >= MEDIUM: print(cur_dict)
        dfs.append(pd.DataFrame([cur_dict]))

df = pd.concat(dfs)
if DEBUG_LEVEL >= LOW: print(df)

# index_col='Resource', 
fig = ff.create_gantt(df, index_col='Task', showgrid_x=True,
                      bar_width=0.4, show_colorbar=True, group_tasks=True)
fig.update_layout(xaxis_type='linear', autosize=False, width=800, height=400)
fig.show()
