# ROB537-FinalProject
# Reinforcement Learning Policy for Autonomous Robotic Exploration in a 2D Environment 

## install Instructions: 

clone the repo with 
```
git clone git@github.com:roboTurt/ROB537-FinalProject.git
```
The spec-file.txt serves to create identical conda environments 

run: 
```
conda create --name "name of my environment" --file spec-file.txt 
```
activate the environment with 
```
conda activate "name of my environment"
```
install openai gym with 
```
pip install gym
```
next, install pytorch 
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

now you can run the test.py file with
```
python test.py
```
