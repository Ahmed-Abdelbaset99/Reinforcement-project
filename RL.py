import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

# ENVIRONMENT

class SmartElevatorEnv:
    def __init__(self, floors=5, max_steps=50):
        self.floors = floors
        self.max_steps = max_steps

    def reset(self):
        self.floor = np.random.randint(self.floors)
        self.target = np.random.randint(self.floors)
        while self.target == self.floor:
            self.target = np.random.randint(self.floors)
        self.steps = 0
        return (self.floor, self.target)

    def step(self, action):
        self.steps += 1
        reward = -0.5
        done = False

        old_dist = abs(self.floor - self.target)

        if action == 0:
            self.floor = min(self.floors - 1, self.floor + 1)
        elif action == 1:
            self.floor = max(0, self.floor - 1)

        new_dist = abs(self.floor - self.target)

        if new_dist < old_dist:
            reward += 1
        else:
            reward -= 1

        if self.floor == self.target:
            reward = 20
            done = True

        if self.steps >= self.max_steps:
            done = True

        return (self.floor, self.target), reward, done


# Q-LEARNING

class QLearning:
    def __init__(self, floors, actions):
        self.q = np.zeros((floors * floors, actions))
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.actions = actions
        self.floors = floors

    def encode(self, state):
        floor, target = state
        return floor * self.floors + target

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actions - 1)
        return np.argmax(self.q[s])

    def update(self, s, a, r, s2):
        self.q[s, a] += self.alpha * (r + self.gamma * np.max(self.q[s2]) - self.q[s, a])


# DQN

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self):
        self.model = DQN()
        self.target = DQN()
        self.target.load_state_dict(self.model.state_dict())

        self.opt = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = []
        self.gamma = 0.95
        self.batch = 32
        self.epsilon = 1.0

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        s = torch.FloatTensor([s])
        return torch.argmax(self.model(s)).item()

    def store(self, exp):
        self.memory.append(exp)
        if len(self.memory) > 5000:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch:
            return

        batch = random.sample(self.memory, self.batch)

        states = torch.FloatTensor([b[0] for b in batch])
        actions = torch.LongTensor([b[1] for b in batch])
        rewards = torch.FloatTensor([b[2] for b in batch])
        next_states = torch.FloatTensor([b[3] for b in batch])
        dones = torch.FloatTensor([b[4] for b in batch])

        q_values = self.model(states)
        next_q = self.target(next_states).detach()

        target = q_values.clone()

        for i in range(self.batch):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * torch.max(next_q[i])

        loss = nn.MSELoss()(q_values, target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


# TRAIN

def train_once(episodes=300):
    env = SmartElevatorEnv()
    q_agent = QLearning(5, 3)
    dqn = DQNAgent()

    q_rewards = []
    dqn_rewards = []

    # Q-Learning
    for ep in range(episodes):
        state = env.reset()
        s = q_agent.encode(state)
        total = 0
        done = False

        while not done:
            a = q_agent.act(s)
            state2, r, done = env.step(a)
            s2 = q_agent.encode(state2)

            q_agent.update(s, a, r, s2)
            s = s2
            total += r

        q_rewards.append(total)
        q_agent.epsilon = max(0.05, q_agent.epsilon * 0.995)

    # DQN
    for ep in range(episodes):
        s = env.reset()
        total = 0
        done = False

        while not done:
            a = dqn.act(s)
            s2, r, done = env.step(a)

            dqn.store((s, a, r, s2, done))
            dqn.train()

            s = s2
            total += r

        dqn_rewards.append(total)
        dqn.epsilon = max(0.05, dqn.epsilon * 0.995)

        if ep % 10 == 0:
            dqn.target.load_state_dict(dqn.model.state_dict())

    return q_rewards, dqn_rewards, q_agent


# MULTIPLE RUNS 

runs = 5
all_q = []
all_dqn = []

for i in range(runs):
    q_r, dqn_r, q_agent = train_once()
    all_q.append(q_r)
    all_dqn.append(dqn_r)

q_r = all_q[-1]
dqn_r = all_dqn[-1]

# PLOTS

def moving_avg(data, w=20):
    return np.convolve(data, np.ones(w)/w, mode='valid')

plt.plot(moving_avg(q_r), label="Q-Learning")
plt.plot(moving_avg(dqn_r), label="DQN")
plt.legend()
plt.title("Learning Curve")
plt.show()

sns.heatmap(q_agent.q)
plt.title("Q-table")
plt.show()

policy = np.argmax(q_agent.q, axis=1)
plt.bar(range(len(policy)), policy)
plt.title("Policy")
plt.show()

# METRICS

def final_perf(r):
    return np.mean(r[-50:])

def stability(runs_data):
    return np.std([np.mean(r[-50:]) for r in runs_data])

def sample_efficiency(r, threshold=15):
    for i, val in enumerate(r):
        if val >= threshold:
            return i
    return -1

def convergence_speed(r, window=20):
    ma = moving_avg(r, window)
    for i in range(len(ma)-1):
        if abs(ma[i] - ma[i+1]) < 0.1:
            return i
    return -1

print("\n========== METRICS ==========")

print("Q Final:", final_perf(q_r))
print("DQN Final:", final_perf(dqn_r))

print("Q Stability:", stability(all_q))
print("DQN Stability:", stability(all_dqn))

print("Q Sample Efficiency:", sample_efficiency(q_r))
print("DQN Sample Efficiency:", sample_efficiency(dqn_r))

print("Q Convergence:", convergence_speed(q_r))
print("DQN Convergence:", convergence_speed(dqn_r))