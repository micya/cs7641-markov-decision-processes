```sh
conda create -n <env> python==3.10
conda activate <env>

pip install -r requirements.txt
```

[Frozen Lake](https://gym.openai.com/envs/FrozenLake-v0/):

```sh
# Value iteration
python FrozenLake/ValueIteration.py

# Policy iteration
python FrozenLake/PolicyIteration.py

# Q learning
python FrozenLake/QLearning.py
```

[Mountain Car](https://gym.openai.com/envs/MountainCar-v0/):

```sh
# Value iteration
python MountainCar/ValueIteration.py

# Policy iteration
python MountainCar/PolicyIteration.py

# Q learning
python MountainCar/QLearning.py
```
