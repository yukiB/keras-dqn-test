import numpy as np

import argparse
from catch_ball import CatchBall
from dqn_agent import DQNAgent
#from collections import deque


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", help='Path ot the model files')
    parser.add_argument("-l", "--load", dest="load", action="store_true",
                        default=False, help='Load trained model (default: off)')
    parser.add_argument("-e", "--epoch-num", dest="n_epochs", default=1000,
                        type=int, help='Numpber of training epochs (default: 1000)')
    parser.add_argument("--simple", dest="is_simple", action="store_true", default=False,
                        help='Train simple model without cnn (8 x 8) (default: off)')
    parser.add_argument("-g", "--graves", dest="graves", action="store_true",
                        default=False, help='Use RmpropGraves (default: off)')
    parser.add_argument("-d", "--ddqn", dest="ddqn", action="store_true",
                        default=False, help='Use Double DQN (default: off)')
    parser.add_argument("-s", "--save-interval", dest="save_interval", default=1000, type=int)
    args = parser.parse_args()

    # parameters
    n_epochs = args.n_epochs

    # environment, agent
    env = CatchBall(simple=args.is_simple)
    agent = DQNAgent(env.enable_actions, env.name, graves=args.graves, ddqn=args.ddqn)
    if args.load:
        agent.load_model(args.model_path, simple=args.is_simple)
    else:
        if not args.is_simple:
            agent.init_model()
        else:
            agent.init_simple_model()
    # variables
    win = 0
    total_frame = 0
    e = 0

    do_replay_count = 0
    
    while e < n_epochs:
        # reset
        frame = 0
        loss = 0.0
        Q_max = 0.0
        env.reset()
        state_t_1, reward_t, terminal = env.observe()
        win = 0
        while not terminal:
            state_t = state_t_1

            # execute action in environment
            action_t = agent.select_action([state_t], agent.exploration)
            env.execute_action(action_t)

            # observe environment
            state_t_1, reward_t, terminal = env.observe()

            # store experience
            start_replay = False
            start_replay = agent.store_experience([state_t], action_t, reward_t, [state_t_1], terminal)

            # experience replay
            if start_replay:
                do_replay_count += 1
                agent.update_exploration(e)
                if do_replay_count > 2:
                    agent.experience_replay(e)
                    do_replay_count = 0

            # update target network
            if total_frame % 500 == 0 and start_replay:
                agent.update_target_model()

            # for log
            frame += 1
            total_frame += 1
            loss += agent.current_loss
            Q_max += np.max(agent.Q_values([state_t]))
            if reward_t == 1:
                win += 1

        if start_replay:
            agent.experience_replay(e, win)
            print("EPOCH: {:03d}/{:03d} | WIN: {:03d} | LOSS: {:.4f} | Q_MAX: {:.4f}".format(
                e, n_epochs - 1, win, loss / frame, Q_max / frame))
            win = 0
        if e > 0 and e % args.save_interval == 0:
            agent.save_model(e, simple=args.is_simple)
            agent.save_model(simple=args.is_simple)
        if start_replay:
            e += 1

    # save model
    agent.save_model(simple=args.is_simple)
