import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd

# df = pd.DataFrame([
#     dict(Task="Job A", Start='2009-01-01', Finish='2009-02-28', Resource="Alex"),
#     dict(Task="Job B", Start='2009-03-05', Finish='2009-04-15', Resource="Alex"),
#     dict(Task="Job C", Start='2009-02-20', Finish='2009-05-30', Resource="Max"),
#     dict(Task="Job D", Start='2009-06-23', Finish='2009-09-10', Resource="Max")
# ])

# fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Resource")
# fig.update_yaxes(autorange="reversed")
# fig.show()

df = pd.DataFrame([
    dict(Task="Job A", Start=0, Finish=2, Resource="R1"),
    dict(Task="Job B", Start=2, Finish=3, Resource="R2"),
    dict(Task="Job A", Start=10, Finish=12, Resource="R1"),
    dict(Task="Job B", Start=20, Finish=21, Resource="R2")
])

fig = ff.create_gantt(df, index_col = 'Resource', showgrid_x=True, bar_width = 0.4, show_colorbar=True, group_tasks=True)
fig.update_layout(xaxis_type='linear', autosize=False, width=800, height=400)
fig.show()
