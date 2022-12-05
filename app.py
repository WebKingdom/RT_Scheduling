import sys
import math
import copy
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtCore import Qt

# Set debug level higher to output more information
DISABLED = 0
LOW = 1
MEDIUM = 2
HIGH = 3
VERBOSITY = LOW

# Refer to https://www.pythonguis.com/tutorials/modelview-architecture/ for MVC
qt_ui_file = "RT_Visualizer.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qt_ui_file)


# Task class for holding task name, ci, pi, and time frames
class Task:
    def __init__(self, name: str, exec_t: int, period: int):
        self.name = name
        self.exec_t = exec_t        # constant once set
        self.remaining_t = exec_t   # changes depending on cur_t
        self.period = period        # constant once set
        self.deadline = period      # constant once set
        self.cur_deadline = period  # changes depending on cur_t
        self.priority = -1          # constant once set
        self.time_frames = list()   # changes depending on cur_t

    def getDict(self, index: int):
        # get TimeFrame tuple of: (start_t, end_t)
        tf = self.time_frames[index]
        if VERBOSITY >= HIGH:
            print(tf)
        return dict(Task=self.name, Start=tf[0], Finish=tf[1])

    def add_time_frame(tf):
        raise NotImplementedError

    def remove_time_frame(tf):
        raise NotImplementedError

    def __iter__(self):
        return TaskIter(self)

    def __str__(self):
        return f"Task {self.name}: ({self.exec_t}, {self.period})"


# Iterator class for Task class
class TaskIter():
    def __init__(self, task_class):
        self._task = task_class
        self._cur_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if (self._cur_index < len(self._task.time_frames)):
            cur_item = self._task.time_frames[self._cur_index]
            self._cur_index += 1
            return cur_item
        raise StopIteration


# Preemption class for holding currently executing task, new task that will preempt the currently executing task,
# and the current time
class Preemption():
    def __init__(self, exec_task: Task, new_task: Task, time: int):
        self.exec_task = exec_task
        self.new_task = new_task
        self.time = time

    def __str__(self):
        return f"Task {self.exec_task.name} preempted by {self.new_task.name} at time {self.time}"


# Visualizer tasks list model (for QListView holding tasks)
class TasksListModel(QtCore.QAbstractListModel):
    def __init__(self, *args, tasks=None, **kwargs):
        super(TasksListModel, self).__init__(*args, **kwargs)
        self.tasks = tasks or []

    # handles requests for data from the view and returns appropriate result
    def data(self, index, role):
        if (role == Qt.DisplayRole):
            task = self.tasks[index.row()]
            return str(task)

    # called by the view to get the rows in the current data
    def rowCount(self, index):
        return len(self.tasks)


# Main window MVC class
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.model = TasksListModel()
        self.listView.setModel(self.model)
        self.addTaskBtn.pressed.connect(self.addTask)
        self.deleteTaskBtn.pressed.connect(self.deleteTask)
        self.generateSchedBtn.pressed.connect(self.generateSched)

    def set_console(self, text):
        self.textEditConsole.setText(str(text))

    def append_console(self, text):
        self.textEditConsole.append(str(text))

    def clear_console(self):
        self.textEditConsole.setText("")

    def addTask(self):
        task_name = self.lineEditTaskName.text()
        task_exec_t = self.lineEditTaskExecTime.text()
        task_period = self.lineEditTaskPeriod.text()

        # Ensure strings are not empty
        if (task_name and task_exec_t and task_period):
            # add task to tasks list. It is a tuple of status and task (which will be converted to text using str())
            new_task = Task(str(task_name), int(task_exec_t), int(task_period))
            if (not self.task_name_exists(task_name)):
                self.model.tasks.append(new_task)
                # trigger refresh
                self.model.layoutChanged.emit()
                # clear inputs
                self.lineEditTaskName.setText("")
                self.lineEditTaskExecTime.setText("")
                self.lineEditTaskPeriod.setText("")
                self.set_console(str(new_task) + " added")
            else:
                self.set_console(str(new_task) + " NOT added! Ensure unique task names.")
        else:
            self.set_console("No task to add! Ensure input fields are not empty.")


    # returns true if task name exists in current tasks list, false otherwise
    def task_name_exists(self, task_name):
        for task in self.model.tasks:
            if (task.name == task_name):
                return True
        return False

    def deleteTask(self):
        indexes = self.listView.selectedIndexes()
        if (indexes):
            # indexes is a list of a single item in single-select mode
            index = indexes[0]
            # delete task, refresh, and clear selection
            to_delete = self.model.tasks[index.row()]
            del self.model.tasks[index.row()]
            self.model.layoutChanged.emit()
            self.listView.clearSelection()
            self.set_console(str(to_delete) + " deleted")
        else:
            self.set_console("No task to delete!")

    def generateSched(self):
        tasks = self.model.tasks
        if (len(tasks) > 0):
            self.set_console("Generating RMS schedule")
            self.show_schedule(self.generate_rms_schedule(tasks), max(len(tasks) * 100, 400), 800)
            self.append_console("")
            self.append_console("Generating EDF schedule")
            self.show_schedule(self.generate_edf_schedule(tasks), max(len(tasks) * 100, 400), 800)
        else:
            self.set_console("No tasks to schedule!")

    # shows the schedule in the DataFrame and height and width parameters

    def show_schedule(self, df, h, w):
        fig = ff.create_gantt(df, index_col='Task', showgrid_x=True, bar_width=0.3, show_colorbar=True, group_tasks=True)
        fig.update_layout(xaxis_type='linear', autosize=False, width=w, height=h)
        fig.show()


    # RMS utilization test returns true if task set is schedulable, false otherwise (sufficient but not necessary). Also sets priority of tasks from low (0) to high (n-1)
    def rms_utilization_test(self, tasks: list[Task]) -> bool:
        n = len(tasks)
        limit = float(n * (2.0 ** (1.0 / n) - 1.0))
        sum = 0.0
        for i in range(0, n):
            sum += float(tasks[i].exec_t) / float(tasks[i].period)
            tasks[i].priority = i
        self.append_console("RMS processor utilization: " + str(round(sum * 100, 3)) + "%")
        return (sum <= limit, tasks)


    # RMS exact analysis test (completion time test) return true if task set is schedulable, false otherwise
    def rms_exact_analysis_test(self, tasks: list[Task]) -> bool:
        n = len(tasks)
        # iterate over tasks from lowest to highest priority (period) and workload will only include subset of tasks that have not passed the test yet
        for i in range(0, n):
            task = tasks[i]
            cur_time = 0.0
            workload = self.compute_workload(tasks[i:n], cur_time, initial_workload=True)
            while ((cur_time != workload) and (workload <= task.period)):
                cur_time = workload
                workload = self.compute_workload(tasks[i:n], cur_time)

            if (cur_time == workload):
                if VERBOSITY >= MEDIUM:
                    self.append_console(str(task) + " schedulable by exact analysis due to: " +
                                        str(cur_time) + " == " + str(workload))
            elif (workload > task.period):
                if VERBOSITY >= MEDIUM:
                    self.append_console(str(task) + " NOT schedulable by exact analysis due to: " +
                                        str(workload) + " > " + str(task.period))
                return False
        return True


    # computes the workload for the given task set at the given time
    def compute_workload(self, tasks: list[Task], cur_time: int, initial_workload=False) -> float:
        workload = 0.0
        for task in tasks:
            if (initial_workload):
                workload += float(task.exec_t)
            else:
                workload += float(task.exec_t) * float(math.ceil(float(cur_time) / float(task.period)))
        return workload


    # EDF utilization test returns true if task set is schedulable, false otherwise. Also sets priority of tasks from low (0) to high (n-1)
    def edf_utilization_test(self, tasks: list[Task]) -> bool:
        sum = 0.0
        for i in range(0, len(tasks)):
            sum += float(tasks[i].exec_t) / float(tasks[i].period)
            tasks[i].priority = i
        self.append_console("EDF processor utilization: " + str(round(sum * 100, 3)) + "%")
        return (sum <= 1.0, tasks)


    # Returns the LCM of all task periods
    def get_lcm_period(self, tasks: list[Task]) -> int:
        n = len(tasks)
        if (n <= 0):
            self.append_console("Error: len(tasks) <= 0. Must have 1+ tasks for scheduling.")
            return -1
        lcm = tasks[0].period
        for i in range(1, n):
            # floor division
            lcm = lcm * tasks[i].period // math.gcd(lcm, tasks[i].period)
        return lcm


    # inserts 1 task at the correct index into the task queue for RMS (ordered low to high priority)
    def insert_task_q_rms(self, task_q: list[Task], task: Task) -> list[Task]:
        task = copy.deepcopy(task)
        if ((task_q is None) or ((task_q is not None) and (len(task_q) == 0))):
            task_q = [task]
            return task_q
        for i in range(len(task_q) - 1, -1, -1):
            if (task.period < task_q[i].period):
                task_q.insert(i+1, task)
                return task_q
            elif (task.period == task_q[i].period):
                task_q.insert(i, task)
                return task_q


    # inserts 1 task at the correct index into the task queue for EDF (ordered low to high priority)
    def insert_task_q_edf(self, task_q: list[Task], task: Task) -> list[Task]:
        task = copy.deepcopy(task)
        if ((task_q is None) or ((task_q is not None) and (len(task_q) == 0))):
            task_q = [task]
            return task_q
        for i in range(len(task_q) - 1, -1, -1):
            if (task.cur_deadline < task_q[i].cur_deadline):
                task_q.insert(i+1, task)
                return task_q
            elif (task.cur_deadline == task_q[i].cur_deadline):
                task_q.insert(i, task)
                return task_q


    # generates the RMS schedule for the given task set
    def generate_rms_schedule(self, tasks: list[Task]):
        n = len(tasks)
        if (n <= 0):
            self.append_console("Error: len(tasks) <= 0. Must have 1+ tasks for scheduling.")
            return pd.DataFrame([dict(Task="No tasks.", Start=0, Finish=0)])

        # * must sort tasks from low (0) to high (len(tasks) - 1) priority so that utilization test can set priority
        tasks = copy.deepcopy(sorted(tasks, key=lambda task: task.period, reverse=True))
        sched_util_test, tasks = self.rms_utilization_test(tasks)
        sched_exact_test = self.rms_exact_analysis_test(tasks)
        self.append_console("RMS utilization test: " + ("Pass" if sched_util_test else "Fail"))
        self.append_console("RMS exact analysis test: " + ("Pass" if sched_exact_test else "Fail"))
        # list of preemptions
        preemptions = []

        if (sched_util_test or sched_exact_test):
            # current time
            cur_t = 0
            # array of length n holding (cur_t / period) at the corresponding task priority (index)
            multiples = [0] * n
            task_q = copy.deepcopy(tasks)
            lcm = self.get_lcm_period(tasks)
            if VERBOSITY >= MEDIUM:
                self.append_console("LCM = " + str(lcm))

            while (cur_t < lcm):
                if ((task_q is not None) and (len(task_q) > 0)):
                    # get highest priority task
                    task = task_q[-1]
                    end_t = cur_t + task.remaining_t
                    # list of higher priority task indices than the current task being scheduled
                    task_indices = []

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
                                    task_indices = [i]
                                elif (new_end_t == end_t):
                                    # occurs when 2+ tasks have the same period/priority
                                    task_indices.append(i)

                    # update remaining_t, add TimeFrame tuple for current task (if applicable), and update task_q
                    remainder = task.remaining_t - (end_t - cur_t)
                    if (cur_t != end_t):
                        tasks[task.priority].time_frames.append((cur_t, end_t))
                    if (remainder > 0):
                        task_q[-1].remaining_t = remainder
                        if (cur_t != end_t):
                            preemptions.append(Preemption(copy.deepcopy(task), copy.deepcopy(tasks[task_indices[-1]]), end_t))
                    elif (remainder == 0):
                        # done scheduling highest priority task
                        task_q.pop()
                    else:
                        self.append_console("Error: remaining_t of task is less than 0.")
                    cur_t = end_t
                else:
                    # no task to schedule, increment time
                    cur_t += 1
                # update multiples based on cur_t and update task_q
                for i in range(0, len(multiples)):
                    new_multiple = int(cur_t / tasks[i].period)
                    if (new_multiple != multiples[i]):
                        task_q = self.insert_task_q_rms(task_q, tasks[i])
                        multiples[i] = new_multiple
        else:
            self.append_console("Task set is not schedulable by RMS.")
            return pd.DataFrame([dict(Task="Not RMS Schedulable", Start=0, Finish=0)])

        # create DataFrame used by Plotly
        dfs = []
        for task in tasks:
            if VERBOSITY >= HIGH:
                self.append_console(task)
            # add all time frame tuples in the task to the DataFrame.
            for i in range(0, len(task.time_frames)):
                cur_dict = task.getDict(i)
                if VERBOSITY >= HIGH:
                    self.append_console(cur_dict)
                dfs.append(pd.DataFrame([cur_dict]))
        df = pd.concat(dfs)
        self.append_console(str(len(preemptions)) + " preemptions")
        for p in preemptions:
            self.append_console(p)
        if VERBOSITY >= LOW:
            self.append_console(df)
        return df


    # generates the EDF schedule for the given task set
    def generate_edf_schedule(self, tasks: list[Task]):
        n = len(tasks)
        if (n <= 0):
            self.append_console("Error: len(tasks) <= 0. Must have 1+ tasks for scheduling.")
            return pd.DataFrame([dict(Task="No tasks.", Start=0, Finish=0)])

        # * must sort tasks from low (0) to high (len(tasks) - 1) priority so that utilization test can set priority
        tasks = copy.deepcopy(sorted(tasks, key=lambda task: task.cur_deadline, reverse=True))
        sched_util_test, tasks = self.edf_utilization_test(tasks)
        self.append_console("EDF utilization test: " + ("Pass" if sched_util_test else "Fail"))
        # list of preemptions
        preemptions = []

        if (sched_util_test):
            # current time
            cur_t = 0
            # array of length n holding (cur_t / period) at the corresponding task priority (index)
            multiples = [0] * n
            task_q = copy.deepcopy(tasks)
            lcm = self.get_lcm_period(tasks)
            if VERBOSITY >= MEDIUM:
                self.append_console("LCM = " + str(lcm))

            while (cur_t < lcm):
                if ((task_q is not None) and (len(task_q) > 0)):
                    # get highest priority task
                    task = task_q[-1]
                    end_t = cur_t + task.remaining_t
                    # list of higher priority task indices than the current task being scheduled
                    task_indices = []

                    # determine end_t due to higher priority tasks
                    for i in range(0, len(multiples)):
                        new_multiple = int(end_t / tasks[i].period)
                        if (new_multiple != multiples[i]):
                            if (int(tasks[i].period * new_multiple + tasks[i].deadline) < task.cur_deadline):
                                # task has higher priority (earlier deadline) than current task being scheduled
                                # minimize end_t (current task will be preempted)
                                new_end_t = tasks[i].period * new_multiple
                                if (new_end_t < end_t):
                                    end_t = new_end_t
                                    task_indices = [i]
                                elif (new_end_t == end_t):
                                    # occurs when 2+ tasks have the same deadline/priority
                                    task_indices.append(i)

                    # update remaining_t, add TimeFrame tuple for current task (if applicable), and update task_q
                    remainder = task.remaining_t - (end_t - cur_t)
                    if (cur_t != end_t):
                        tasks[task.priority].time_frames.append((cur_t, end_t))
                    if (remainder > 0):
                        task_q[-1].remaining_t = remainder
                        if (cur_t != end_t):
                            preemptions.append(Preemption(copy.deepcopy(task), copy.deepcopy(tasks[task_indices[-1]]), end_t))
                    elif (remainder == 0):
                        # done scheduling highest priority task
                        task_q.pop()
                    else:
                        self.append_console("Error: remaining_t of task is less than 0.")
                    cur_t = end_t
                else:
                    # no task to schedule, increment time
                    cur_t += 1
                for i in range(0, len(multiples)):
                    new_multiple = int(cur_t / tasks[i].period)
                    if (new_multiple != multiples[i]):
                        tasks[i].cur_deadline = int(tasks[i].period * new_multiple + tasks[i].deadline)
                        task_q = self.insert_task_q_edf(task_q, tasks[i])
                        multiples[i] = new_multiple
        else:
            self.append_console("Task set is not schedulable by EDF.")
            return pd.DataFrame([dict(Task="Not EDF Schedulable", Start=0, Finish=0)])

        dfs = []
        for task in tasks:
            if VERBOSITY >= HIGH:
                self.append_console(task)
            # add all time frame tuples in the task to the DataFrame.
            for i in range(0, len(task.time_frames)):
                cur_dict = task.getDict(i)
                if VERBOSITY >= HIGH:
                    self.append_console(cur_dict)
                dfs.append(pd.DataFrame([cur_dict]))
        df = pd.concat(dfs)
        self.append_console(str(len(preemptions)) + " preemptions")
        for p in preemptions:
            self.append_console(p)
        if VERBOSITY >= LOW:
            self.append_console(df)
        return df


# Error message printer
def print_error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# utility function for printing all the tasks from a list of tasks
# Ensures iterator is working
def print_tasks(tasks):
    iterator = iter(tasks)
    while True:
        try:
            e = next(iterator)
            print(e)
        except StopIteration:
            print("Done iterating over tasks")
            break


# Sample tasks
# tasks = [Task("T1", 1, 8), Task("T2", 2, 6), Task("T3", 4, 24)]
# tasks = [Task("T1", 3, 12), Task("T2", 3, 12), Task("T3", 8, 16)]
# tasks = [Task("T1", 2, 8), Task("T2", 3, 12), Task("T3", 4, 16)]
# tasks = [Task("T1", 2, 8), Task("T2", 3, 12), Task("T3", 5, 16), Task("T4", 4, 32), Task("T5", 6, 96)]
# tasks = [Task("T1", 1, 8), Task("T6", 1, 8), Task("T2", 3, 12), Task("T3", 5, 16), Task("T4", 4, 32), Task("T5", 3, 96), Task("T7", 2, 96)]
# tasks = [Task("T1", 1, 4), Task("T2", 3, 50), Task("T3", 3, 32), Task("T4", 1, 36), Task("T5", 5, 128)]


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
