# Double DQN

Last Edited: Nov 19, 2018 6:06 PM
Tags: RL

## 논문

double: [https://arxiv.org/pdf/1509.06461.pdf](https://arxiv.org/pdf/1509.06461.pdf)

duel: [https://arxiv.org/pdf/1511.06581.pdf](https://arxiv.org/pdf/1511.06581.pdf)

## Double

그냥 DQN 식

$$loss = (Q(s,a) - r + \gamma Q'(s, argmax_{a'}Q'(s,a'))^2$$

Double DQN 식

$$loss = (Q(s,a) - r + \gamma Q'(s, argma_{a'}Q(s,a'))^2$$

Action 선택을 `target_net` 으로 하는지 `main_net` 으로 하는지의 차이만 있을 뿐이다.

DQN에서는 단순하게 `target_net`으로 Action을 선택했는데 이 경우에는 만약 `target_net`이 가장 큰 `qvalue`을 가지고 있는 `action`을 선택하면 그 `action`이 다시 `Q-value`을 증가 시키고 다시 그 `action`이 선택 되는 순환이 발생 할 수 있기 때문에 `action` 을 선택하는 `net` 과 `value` 을 평가하는 `net` 을 분리시킨다.

## 구현

```python
def train_model(cls, net, target_net, optimizer, batch, batch_size):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).float()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)

        pred = net(states).squeeze(1)
        _, action_from_net = net(next_states).squeeze(1).max(1)
        next_pred = target_net(next_states).squeeze(1)

        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + masks * gamma * next_pred.gather(1, action_from_net.unsqueeze(1)).squeeze(1)
```


​    
```python
        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

일단 `action` 을 `net` 의 `max` 로 구한다. 그 뒤 `target_net` 에서 그 `action` 값에 맞는 `Q-value`을 사용한다.