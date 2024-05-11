'''
Author: malonelin
Date: 2024.04.25
ref: https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char05%20DDPG/DDPG.py
'''
import torch, os, sys, random, argparse, glob
import numpy as np
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from loguru import logger as log
from tensorboardX import SummaryWriter

'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='test', type=str) # mode = 'train' or 'test'
parser.add_argument('--render', default=False, type=bool) # show UI or not
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
# parser.add_argument("--env_name", default="Pendulum-v0")
parser.add_argument("--env_name", default="LunarLanderContinuous-v2")
parser.add_argument('--tau',  default=0.001, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=100, type=int)

parser.add_argument('--lr_actor', default=1e-4, type=float)
parser.add_argument('--lr_critic', default=1e-3, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=1e5, type=int) # replay buffer size
parser.add_argument('--batch_size', default=64, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--max_length_of_trajectory', default=250, type=int) #
# parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--log_interval', default=20, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
# parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=100000, type=int) # num of games
parser.add_argument('--best_w_file', default='./w/actor_best.pth_ep32514_rw292.77_st177', type=str)
parser.add_argument('--best_w_file_name', default='actor_best.pth*rw29*', type=str)
parser.add_argument('--update_iteration', default=20, type=int)
args = parser.parse_args()

log.add('log/info_{time}.log')
log.info('hyparam:')
log.info(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
log.info(f'using:{device}')
script_name = os.path.basename(__file__)
env = gym.make(args.env_name,
               max_episode_steps = args.max_length_of_trajectory,
               render_mode = 'human' if (args.render and args.mode == 'test') else None)
env_val = gym.make(args.env_name,
                   max_episode_steps = args.max_length_of_trajectory,
                   render_mode = 'human')

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_Val = torch.tensor(1e-7).float().to(device) # min value

W_DIR = './w/'

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def full(self):
        return len(self.storage) == self.max_size
    
    def push(self, data):
        if self.full():
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
 
class OrnsteinUhlenbeckProcess(object):
    def __init__(self, size, theta=.15, mu=0., sigma=.2, n_steps=1, dt=1e-4,
                 random_state=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.n_steps = n_steps
        self.dt = dt
        self.size = size
        self.reset()
        self.random_state = np.random.RandomState() if random_state is None else random_state
 
    def reset(self):
        self.x = self.random_state.randn(self.size) * .01
 
    def sample(self):
        dx = self.theta * (self.mu - self.x) * self.dt + \
             self.sigma * np.sqrt(self.dt) * self.random_state.randn(self.size)
        self.x += dx
        return self.x

class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_actor)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr_critic)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter('tensorboardX')

        self.actor_target.eval()
        self.critic_target.eval()

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.num_val_test = 0

    @torch.no_grad()
    def select_action(self, state):
        self.actor.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def learn(self):
        for it in range(args.update_iteration):
            with torch.no_grad():
                # Sample replay buffer
                x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
                state = torch.FloatTensor(x).to(device)
                action = torch.FloatTensor(u).to(device)
                next_state = torch.FloatTensor(y).to(device)
                done = torch.FloatTensor(1-d).to(device)
                reward = torch.FloatTensor(r).to(device)

                # Compute the target Q value
                target_Q = self.critic_target(next_state, self.actor_target(next_state))
                target_Q = reward + (done * args.gamma * target_Q).detach()

            self.critic.train()
            # Get current Q estimate
            current_Q = self.critic(state, action)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            self.actor.train()
            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            with torch.no_grad():
                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), W_DIR + 'actor.pth')
        torch.save(self.critic.state_dict(), W_DIR + 'critic.pth')

    def save_best(self, ep, r, s):
        actor_w_name = format(f'actor_best.pth_ep{ep}_rw{r:.2f}_st{s:.0f}')
        torch.save(self.actor.state_dict(), W_DIR + actor_w_name)
        critic_w_name = format(f'critic_best.pth_ep{ep}_rw{r:.2f}_st{s:.0f}')
        torch.save(self.critic.state_dict(), W_DIR + critic_w_name)

    def load(self):
        self.actor.load_state_dict(torch.load(W_DIR + 'actor.pth'))
        self.critic.load_state_dict(torch.load(W_DIR + 'critic.pth'))
        log.info("====================================")
        log.info("model has been loaded...")
        log.info("====================================")

    def val_test(self, train_episode):
        avg_steps = 0
        avg_reward = 0
        test_ep = 0
        for test_ep in range(10):
            state, _ = env.reset()
            ep_r = 0
            for t in range(args.max_length_of_trajectory):
                action = self.select_action(state)
                state, reward, done, truncated, _ = env.step(np.float32(action))
                ep_r += reward
                if done or truncated:
                    avg_steps += (t + 1)
                    avg_reward += ep_r
                    ep_r = 0
                    break
        if test_ep > 0:
            avg_steps /= (test_ep + 1)
            avg_reward /= (test_ep + 1)
        if avg_reward > 200:
            self.save_best(train_episode, avg_reward, avg_steps)
        log.info(f'validate test:{test_ep + 1:>2}. avg_steps:{(avg_steps):>3} avg_reward:{avg_reward:>7.2f}')
        self.writer.add_scalar('val_test/avg_reward', avg_reward, self.num_val_test)
        self.writer.add_scalar('val_test/avg_steps', avg_steps, self.num_val_test)
        self.num_val_test += 1

    def human_test(self):
        state, _ = env_val.reset()
        ep_r = 0
        for t in range(args.max_length_of_trajectory):
            action = self.select_action(state)
            state, reward, done, truncated, _ = env_val.step(np.float32(action))
            ep_r += reward
            if done or truncated or t >= args.max_length_of_trajectory:
                log.info(f'human test. steps:{(t+1):>3} reward:{ep_r:>7.2f}')
                ep_r = 0
                break

    def get_w_file_list(self):
        w_file_list = glob.glob(os.path.join(W_DIR, args.best_w_file_name))
        sorted(w_file_list, key = os.path.getctime)
        return w_file_list

    def print_test_results(self, results):
        # results is lists of tuple (gt200_cnt, test_cnt, avg_reward, avg_steps, w_file)
        # check the tuple order by the return value of test_mode_one_ep
        log.info('sorted results:')
        for r in results:
            log.info(f'sorted results. gt200_cnt/test_cnt:({r[0]:>3}/{r[1]}) avg_reward:{r[2]:>7.2f} avg_steps:{r[3]:>3} w_file{r[4]}')

    def test_mode(self):
        if 'all' == args.best_w_file:
            w_files = self.get_w_file_list()
            w_files.sort()
            test_results = []
            for w_file in tqdm(w_files):
                test_results.append(self.test_mode_one_ep(w_file, print_log = False))
            test_results.sort()
            self.print_test_results(test_results)
        else:
            self.test_mode_one_ep(args.best_w_file)
            
    def test_mode_one_ep(self, w_file, print_log = True):
        self.actor.load_state_dict(torch.load(w_file))
        avg_reward = 0
        avg_steps = 0
        gt200_cnt = 0
        test_cnt = args.test_iteration
        for i in range(test_cnt):
            state, _ = env.reset()
            ep_r = 0
            for t in range(args.max_length_of_trajectory):
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = env.step(np.float32(action))
                ep_r += reward
                if done or truncated:
                    avg_reward += ep_r
                    avg_steps += (t + 1)
                    if ep_r >= 200:
                        gt200_cnt += 1
                    if print_log: log.info(f'test mode. ep{i:>3} steps:{(t+1):>3} reward:{ep_r:>7.2f}')
                    break
                state = next_state
        if test_cnt <= 0:
            return
        avg_reward /= test_cnt
        avg_steps /= test_cnt
        log.info(f'test mode. test_cnt:{test_cnt} gt200_cnt:({gt200_cnt}/{test_cnt}) avg_steps:{avg_steps:>3} avg_reward:{avg_reward:>7.2f} w_file{w_file}')
        return (gt200_cnt, test_cnt, avg_reward, avg_steps, w_file)
    
def main():
    agent = DDPG(state_dim, action_dim, max_action)
    # ou_process = OrnsteinUhlenbeckProcess(size=1, sigma=0.2)
    # noise = ou_process.sample()
    ep_r = 0
    if args.mode == 'test':
        agent.test_mode()
    elif args.mode == 'train':
        if args.load:
            agent.load()
        total_step = 0
        for i in range(args.max_episode):
            total_reward = 0
            step =0
            state, _ = env.reset()
            for t in range(args.max_length_of_trajectory):
                action = agent.select_action(state)
                action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)
                next_state, reward, done, truncated, _ = env.step(action)
                if truncated:
                    reward = -100
                agent.replay_buffer.push((state, next_state, action, reward, float(done)))
                state = next_state

                if done or truncated:
                    break
                step += 1
                total_reward += reward
            total_step += step+1
            log.info(f'training steps:{total_step:>5} ep:{i:>4} landStep:{step+1:>3} reward:{total_reward:>8.2f}')
            agent.learn()

            if (i % args.log_interval == 0 and i != 0) or (total_reward > 200):
                agent.save()
                agent.val_test(i)
                agent.human_test()
    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main()
