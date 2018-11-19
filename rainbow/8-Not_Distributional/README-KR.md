# Rainbow-Distributional

Last Edited: Nov 19, 2018 7:20 PM
Tags: RL

이제 지금까지 한 것 들을 조합해서 Rainbow을 만듭니다.

일단 바로 Rainbow 을 하면 복잡하기 때문에 Distributional 을 제외한 부분부터 조합하도록 합시다.

정리하면

1. DQN (기본)
2. Double DQN ( Q-value 을 계산하는 Net 과 액션 선택 Net 을 분리 )
3. Duel DQN ( Q = V + A, A 는 Advantage )
4. Multi Step ( Q-value 계산에서 여러 좀 더 뒤의 reward 도 사용 )
5. PER ( Replay Memory 에서 좀 더 의미 있는 sample 을 더 자주 사용 )
6. Nosiy Net ( Net 전체에 Noisy 을 주어서 explore 을 하도록 )

입니다.

## Memory.py

multi_step 과 PER 부분이 적용됩니다.

deque 보다는 list 가 좀더 많은 기능을 사용할 수 있기 때문에 deque 을 list 로 바꾸어 줍니다.

```python
def __init__(self, capacity):
        self.memory = []
        self.memory_probabiliy = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, next_state, action, reward, mask):
        """Saves a transition."""
        if len(self.memory) > 0:
            max_probability = max(self.memory_probabiliy)
        else:
            max_probability = small_epsilon

        if len(self.memory) < self.capacity:
            self.memory.append(Transition(state, next_state, action, reward, mask))
            self.memory_probabiliy.append(max_probability)
        else:
            self.memory[self.position] = Transition(state, next_state, action, reward, mask)
            self.memory_probabiliy[self.position] = max_probability

        self.position = (self.position + 1) % self.capacity
```

기본뼈대는 PER 과 같지만 N-step 전까지의 replay 안에서만 sample 을 구하고 sample 을 N-step 에 맞게 변형해주는 부분이 들어갑니다.

```python
def sample(self, batch_size, net, target_net, beta):
        memory_probabiliy = self.memory_probabiliy[:(len(self.memory_probabiliy) - n_step)]
        probability_sum = sum(memory_probabiliy)
        p = [probability / probability_sum for probability in memory_probabiliy]
        # print(len(self.memory_probabiliy))

        indexes = np.random.choice(np.arange(len(self.memory) - n_step), batch_size, p=p)
        transitions = []
        for transition_start_idx in indexes:
            state = self.memory[transition_start_idx].state
            action = self.memory[transition_start_idx].action
            next_state = self.memory[transition_start_idx + n_step].next_state
            mask = self.memory[transition_start_idx + n_step].mask
            reward = 0
            for step in range(0, n_step + 1):
                idx = transition_start_idx + step
                reward += (gamma ** step) * self.memory[idx].reward
            transitions.append(Transition(state, next_state, action, reward, mask))

        transitions_p = [p[idx] for idx in indexes]
        batch = Transition(*zip(*transitions))

        weights = [pow(self.capacity * p_j, -beta) for p_j in transitions_p]
        weights = torch.Tensor(weights).to(device)
        weights = weights / weights.max()
        td_error = QNet.get_td_error(net, target_net, batch.state, batch.next_state, batch.action, batch.reward, batch.mask)

        td_error_idx = 0
        for idx in indexes:
            self.memory_probabiliy[idx] = pow(abs(td_error[td_error_idx]) + small_epsilon, alpha).item()
            # print(pow(abs(td_error[td_error_idx]) + small_epsilon, alpha).item())
            td_error_idx += 1
        return batch, weights
```

## Model.py

Noisy Net, Double, Duel 이 적용됩니다. 또한 PER 에 필요한 td_error 을 쉽게 구하기 위해서 함수를 따로 빼줍니다.

일단 Noisy Net 을 정의해주고 Advantage 부분에 Noisy 을 적용해 줍니다. explore 위해서 Noisy Net을 적용하기 때문에 실제 action 에 관련있는 Adv 에만 적용해 주었습니다.

```python
def __init__(self, num_inputs, num_outputs):
        super(QNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_adv = NoisyLinear(128, num_outputs)
        self.fc_val = nn.Linear(128, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
```


​    
```python

@classmethod
    def get_td_error(cls, oneline_net, target_net, states, next_states, actions, rewards, masks):
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.Tensor(actions)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)

        pred = oneline_net(states).squeeze(1)
        _, action_from_oneline_net = oneline_net(next_states).squeeze(1).max(1)
        next_pred = target_net(next_states).squeeze(1)

        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + masks * gamma * next_pred.gather(1, action_from_oneline_net.unsqueeze(1)).squeeze(1)

        td_error = pred - target.detach()

        return td_error
```

explore 을 위한 Noisy Net 이기 때문에 action을 구할 때 마다 nosie 을 초기화 해줍니다.

```python
def get_action(self, input):
        self.fc_adv.reset_noise()
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.numpy()[0]
```