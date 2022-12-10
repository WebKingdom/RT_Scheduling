# RT_Scheduling
An app for visualizing real-time scheduling algorithms.

# Setup
Use the commands below to set up the Real-Time Scheduling Algorithm Visualizer (RT-SAV) app.
## Windows:
```
git clone https://github.com/WebKingdom/RT_Scheduling 
cd RT_Scheduling
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
py -3 -m venv .venv
.venv\scripts\activate
pip3 install pandas
pip3 install jsonpickle
pip3 install plotly
pip3 install pyqt5
python app.py
```

## macOS:
```
git clone https://github.com/WebKingdom/RT_Scheduling 
cd RT_Scheduling
pip3 install virtualenv
virtualenv .venv
source .venv/bin/activate
pip3 install pandas
pip3 install jsonpickle
pip3 install plotly
pip3 install pyqt5
python app.py
```
