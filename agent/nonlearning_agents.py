import json
from collections import defaultdict
import os.path as osp
from socket import IP_ADD_MEMBERSHIP
import numpy as np
from habitat import Env, logger
from habitat.config.default import Config
from habitat.core.agent import Agent
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from tqdm import tqdm, trange
import json 
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.tasks.nav.nav import DistanceToGoal, Success

from utils import get_surrounding_point_rel_pos_with_radius3d
from grid2sim_utils import grid2sim

def evaluate_agent(config: Config) -> None:
    split = config.EVAL.SPLIT
    data_path = config.EVAL.DATA_PATH
    config.defrost()
    # turn off RGBD rendering as neither RandomAgent nor HandcraftedAgent use it.
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = []
    config.TASK_CONFIG.TASK.SENSORS = []
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.DATASET.DATA_PATH = data_path
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    config.TASK_CONFIG.TASK.SDTW.SPLIT = split
    config.freeze()

    num_nan = 0
    path_data = {}

    env = Env(config=config.TASK_CONFIG)

    if config.EVAL.NONLEARNING.AGENT == 'GridToSimAgent':
        agent = GridToSimAgent(config.EVAL.NONLEARNING, env)
    elif config.EVAL.NONLEARNING.AGENT == 'ActionSimAgent':
        agent = ActionSimAgent(config.EVAL.NONLEARNING, env)
    else:
        raise NotImplementedError()

    stats = defaultdict(float)
    no_reachable_eps = []
    num_episodes = min(config.EVAL.EPISODE_COUNT, len(env.episodes))
    
    eval_dict = {} # record is_success of each episode
    for _ in trange(num_episodes):
        obs = env.reset()
        agent.reset()
    
        while not env.episode_over:
            action ,is_nan, ep_id = agent.act(obs)
            if is_nan:
                no_reachable_eps.append(ep_id)
            
            obs = env.step(action)

        is_success = env.task.measurements.measures[Success.cls_uuid].get_metric()
        try:
            eval_dict[agent.ep_results['episode_id']] = {
                'episode': agent.ep_results['episode_id'], 
                'traj_id': agent.ep_results['ep_id'],
                'is_success': is_success>0.5}
        except:
            pass
 
        if is_nan:
            num_nan += 1
        for m, v in env.get_metrics().items():
            stats[m] += v

    print("NAN", num_nan)
    print("NO reachable eps ", set(no_reachable_eps))
    print("Num ", len(set(no_reachable_eps)))
    stats = {k: v / num_episodes for k, v in stats.items()}

    logger.info(f"Averaged benchmark for {config.EVAL.NONLEARNING.AGENT}:")
    for stat_key in stats.keys():
        logger.info("{}: {:.3f}".format(stat_key, stats[stat_key]))

    with open(osp.join(config.EVAL.NONLEARNING.DUMP_DIR, f"stats_{split}.json"), "w") as f:
        json.dump(stats, f, indent=4)

    with open(osp.join(config.EVAL.NONLEARNING.DUMP_DIR, f"{split}_sim_results.json"),"w") as f:
        json.dump(eval_dict,f)
    print("DONE !!")
    
class GridToSimAgent(Agent):

    def __init__(self, config, env):
      
        with open(config.RESULT_PATH, "r") as f:
            self.data = {line['episode_id']:line for line in json.load(f)}
            
        self.map_root = config.MAP_ROOT

        self.actions = [
            HabitatSimActions.STOP,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ]
        
        self.shortest_path_follower = ShortestPathFollower(
            env._sim, goal_radius=0.2, return_one_hot=False, 
            stop_on_error=True
        )
        self.env = env
        self.is_nan  = False

    def reset(self):
        self.tmp_goal_index = -1 
        self.current_ep_id = None

    def act(self, observations):
        episode_id = self.env.current_episode.episode_id
        if self.current_ep_id is None or episode_id != self.current_ep_id:
            self.current_ep_id =  episode_id
            try:
                ep_results  = self.data[episode_id]
                self.ep_results = ep_results
            except:
                print(f"[WARNING] epid {episode_id} not predicted")
                return {"action": self.actions[0]}, self.is_nan, episode_id

            grid2sim(ep_results, map_root=self.map_root, dump_dir=None) # process prediction results

            self.sim_path = ep_results["sim_path"]

            # self.dis_score = ep_results['dist_score']
            start_position = self.env._sim.get_agent_state().position
            self.mid_value = start_position[1]
            
            self.tmp_goal_index = -1
            self.tmp_goal = (self.sim_path[self.tmp_goal_index][0], self.mid_value, self.sim_path[self.tmp_goal_index][2])

            flag=False
            if self.env._sim.is_navigable(np.array(self.tmp_goal)):
                flag=True

            # modify not navigable on simulator
            while not self.env._sim.is_navigable(np.array(self.tmp_goal)):
                if self.tmp_goal_index == -1:
                    # last point only change height to search navigable point
                    coordx = get_surrounding_point_rel_pos_with_radius3d(0,1)
                else:
                    # other point search a region inside sphere
                    # coordx = get_surrounding_point_rel_pos_with_radius3d(0.2, 11) # unseen
                    coordx = get_surrounding_point_rel_pos_with_radius3d(0.1, 11) # seen
 
                try:
                    self.sim_path[self.tmp_goal_index]
                except:
                    # print(self.current_ep_id, self.tmp_goal_index)
                    break

                for (xs, ys, zs) in coordx:
                    self.tmp_goal = (
                        self.sim_path[self.tmp_goal_index][0] + xs,
                        self.mid_value + zs,
                        self.sim_path[self.tmp_goal_index][2] + ys
                    )
                    if self.env._sim.is_navigable(np.array(self.tmp_goal)):
                        flag=True
                        break
                if flag:
                    break
                self.tmp_goal_index -= 1
                if self.tmp_goal_index == -10:
                    print(episode_id)
            
            if not flag:
                print(episode_id, self.sim_path, "\n")
                self.goal_point = (self.sim_path[-1][0], self.mid_value, self.sim_path[-1][2])
            else:
                self.goal_point = self.tmp_goal
            
              
        act_index = self.shortest_path_follower.get_next_action(np.array(self.goal_point))
        if act_index is None:
            return {"action": self.actions[0]}, self.is_nan, episode_id
            

        return {"action": self.actions[act_index]}, self.is_nan, episode_id


    
class ActionSimAgent(Agent):

    def __init__(self, config, env):
      
        with open(config.RESULT_PATH, "r") as f:
            self.data= {v['episode_id']:v['action_seq'] for v in json.load(f)}
            
        self.map_root = config.MAP_ROOT

        self.actions = [
            HabitatSimActions.STOP,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ]
        
        self.shortest_path_follower = ShortestPathFollower(
            env._sim, goal_radius=0.2, return_one_hot=False, 
            stop_on_error=True
        )
        self.env = env
        self.is_nan  = False

    def reset(self):
        self.action_index = -1 
        self.current_ep_id = None

    def act(self, observations):
        episode_id = self.env.current_episode.episode_id
        if self.current_ep_id is None or episode_id != self.current_ep_id:
            self.current_ep_id =  episode_id
            try:
                action_seq  = self.data[episode_id]
                self.action_seq = action_seq
                self.action_index = 0
            except:
                print(f"[WARNING] epid {episode_id} not predicted")
                return {"action": self.actions[0]}, self.is_nan, episode_id

        action = {"action": self.actions[self.action_seq[self.action_index]]}
        self.action_index+=1
        return action, self.is_nan, episode_id