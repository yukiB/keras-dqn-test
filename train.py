import numpy as np

import argparse
from catch_ball import CatchBall
from dqn_agent import DQNAgent
from collections import deque
import copy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-l", "--load", dest="load", action="store_true", default=False)
    parser.add_argument("-e", "--epoch-num", dest="n_epochs", default=200, type=int)
    args = parser.parse_args()

    # parameters
    n_epochs = args.n_epochs
    state_num = 3

    # environment, agent
    env = CatchBall()
    agent = DQNAgent(env.enable_actions, env.name)
    if args.load:
        agent.load_model(args.model_path)
    else:
        agent.init_model()
    # variables
    win = 0
    total_frame = 0
    e = 0
    
    while e < n_epochs:
        # reset
        frame = 0
        loss = 0.0
        Q_max = 0.0
        do_replay_count = 0
        env.reset()
        state_t_1, reward_t, terminal = env.observe()
        S = deque(maxlen=state_num)
        win = 0
        while not terminal:
            state_t = state_t_1

            if len(S) == 0:
                [S.append(state_t) for i in range(state_num)]
            else:
                S.append(state_t)

            # execute action in environment
            #action_t = agent.select_action(S, agent.exploration)
            action_t = agent.select_action([state_t], agent.exploration)
            env.execute_action(action_t)

            # observe environment
            state_t_1, reward_t, terminal = env.observe()

            # store experience
            start_replay = False
            new_S = copy.copy(S)
            new_S.append(state_t_1)
            #start_replay = agent.store_experience(S, action_t, reward_t, new_S, terminal)
            start_replay = agent.store_experience([state_t], action_t, reward_t, [state_t_1], terminal)

            # experience replay
            if start_replay:
                do_replay_count += 1
                agent.update_exploration(e)
                if do_replay_count > 2:
                    agent.experience_replay()
                    do_replay_count = 0

            # update target network
            if total_frame % 100 == 0 and start_replay:
                agent.update_target_model()

            # for log
            frame += 1
            total_frame += 1
            loss += agent.current_loss
            #Q_max += np.max(agent.Q_values(S))
            Q_max += np.max(agent.Q_values([state_t]))
            if reward_t == 1:
                win += 1

        if start_replay:
            print("EPOCH: {:03d}/{:03d} | WIN: {:03d} | LOSS: {:.4f} | Q_MAX: {:.4f}".format(
                e, n_epochs - 1, win, loss / frame, Q_max / frame))
            win = 0
        if e > 0 and e % 100 == 0:
            agent.save_model(e)
            agent.save_model()
        if start_replay:
            e += 1

    # save model
    agent.save_model()
