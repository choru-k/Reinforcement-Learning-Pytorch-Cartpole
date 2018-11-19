# Rainbow

Last Edited: Nov 19, 2018 8:32 PM

## Duel + Distributional

일단 지금까지에서 Q는 밑으로 정의하고

$$Q(s,a) = R(s,a) + \gamma \ argmax_{a'}Q(s',a')$$

Duel 에서 밑의 식을 사용하였습니다.

$$Q(s,a) = V(s) + A(s,a)$$

그리고 Distributional 에서는

$$Z(s,a)=R(s,a)+ \gamma \ Z(s',a')$$

입니다. 여기서 Z 는 확률분포 입니다. 이 두 개를 같이 적용 한다면

$$Z(s,a) = V(s) + A(s,a)$$

V와 A 을 확률 분포로써 적용 할 수 있을 것 같습니다. V 는 `(num_support)` 의 차원을 갖고 A 는 `(action_space, num_support)` 의 차원을 가질 것입니다.

```python
def forward(self, x):
        x = F.relu(self.fc(x))
        adv = self.fc_adv(x)
        val = self.fc_val(x)

        val = val.view(-1, 1, num_support)
        adv = adv.view(-1, self.num_outputs, num_support)
        z = val + (adv - adv.mean(1, keepdim=True))
        z = z.view(-1, self.num_outputs, num_support)
        p = nn.Softmax(dim=2)(z)
        return p
```

## Double + Distributional

oneline_net 에서 action을 구합니다.

```python
        z_space = online_net.z.repeat(batch_size, online_net.num_outputs, 1)
        prob_next_states_online = online_net(next_states)
        prob_next_states_target = target_net(next_states)
        Q_next_state = torch.sum(prob_next_states_online * z_space, 2)
        next_actions = torch.argmax(Q_next_state, 1)
        prob_next_states_action = torch.stack([prob_next_states_target[i, action, :] for i, action in enumerate(next_actions)])

```

## PER + Distributional

그 전의 PER 에서는 td_error 의 절댓 값의 비율이 경험의 중요성이 되었습니다. 

여기서는 Distributional 의 Loss값을 경험의 중요성이라고 고려합니다.

```python
td_error = QNet.get_loss(net, target_net, batch.state, batch.next_state, batch.action, batch.reward, batch.mask)
```



```python

@classmethod
    def get_loss(cls, oneline_net, target_net, states, next_states, actions, rewards, masks):
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.Tensor(actions).int()
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)

        z_space = oneline_net.z.repeat(batch_size, oneline_net.num_outputs, 1)
        prob_next_states = oneline_net(next_states)
        Q_next_state = torch.sum(prob_next_states * z_space, 2)
        next_actions = torch.argmax(Q_next_state, 1)
        prob_next_states_action = torch.stack([prob_next_states[i, action, :] for i, action in enumerate(next_actions)])

        m_prob = cls.get_m(rewards, masks, prob_next_states_action)
        m_prob = torch.tensor(m_prob)

        m_prob = m_prob / torch.sum(m_prob, dim=1, keepdim=True)
        expand_dim_action = torch.unsqueeze(actions, -1)
        p = torch.sum(oneline_net(states) * expand_dim_action.float(), dim=1)
        loss = -torch.sum(m_prob * torch.log(p + 1e-20), 1)

        return loss
```

나머지는 Distributional 이 없는 버전과 동일합니다.

