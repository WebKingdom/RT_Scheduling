import altair as alt
import pandas as pd

# allows VSCode to display graph
# alt.renderers.enable('mimetype')
# alt.renderers.enable('altair_viewer')


# * source data must be in pd.DataFrame format
source = pd.DataFrame([
    {"task": "A", "start": 1, "end": 3},
    {"task": "B", "start": 3, "end": 8},
    {"task": "C", "start": 8, "end": 10},
    {"task": "D", "start": 12, "end": 13},
    {"task": "D", "start": 12, "end": 13},
    {"task": "D", "start": 20, "end": 22}
])

# 
graph = alt.Chart(source).mark_bar().encode(
    x='start',
    x2='end',
    y='task'
)

graph.show()

# save graph to external source
# graph.save('graph1.html')
