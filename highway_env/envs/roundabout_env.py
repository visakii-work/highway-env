from typing import Tuple

from gym.envs.registration import register
import numpy as np

from highway_env.envs.common.action import action_factory, Action, DiscreteMetaAction, ActionType
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle

Observation = np.ndarray



class RoundaboutEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "LidarObservation",
                #"absolute": True,
                #"features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-15, 15], "vy": [-15, 15]},
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": [0,3,8,12]
            },
            "incoming_vehicle_destination": None,
            "collision_reward": -2.0,
            "high_speed_reward": 0.5,
            "right_lane_reward": 0,
            "lane_change_reward": -0.5,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "duration": 25,
        })
        return config

    def _reward(self, action: int, obs: np.ndarray) -> float:
        lane_change = action == 0 or action == 2

        # compute the distance here 
        col = 0
        sort_obs = obs[np.argsort(obs[:,col])].copy()

        #print(sort_obs)
        #input()
        #pick two closest obstacles

        obstacle_distance = sort_obs[:,col]

        obstacle_distances_grouped = []
        group = [obstacle_distance[0]]
        for i in range(1,len(obstacle_distance)):
            if obstacle_distance[i] != 1.:
                if obstacle_distance[i] - obstacle_distance[i-1] < 0.03:
                    group.append(obstacle_distance[i-1])
                else:
                    obstacle_distances_grouped.append(min(group))
                    group = [obstacle_distance[i]]

        #1/(1 + e^(-2*dist)) -- 
        
        #print(obstacle_distances_grouped)

        #input()
        distance_metric = 0.
        self.obstacle_distance = sort_obs[:,col]
        self.distance_per_bin = [0]*len(obstacle_distances_grouped)
        for i in range(len(obstacle_distances_grouped)):
            if obstacle_distances_grouped[i] <= 0.40:
                penalty = -np.exp(-6.5*obstacle_distances_grouped[i])
                self.distance_per_bin[i] = penalty
                distance_metric += penalty

        #print(distance_metric)
        #input()
        acc = (self.vehicle.speed - self.prev_speed)
        #reward = self.config["collision_reward"] * self.vehicle.crashed \
        #    + 1*self.vehicle.speed / 12 \
        #    + 0.5*self.config["lane_change_reward"] * lane_change + 1.5*distance_metric - 5e-6*acc**2
        
        self.collision_rew = self.config["collision_reward"] * self.vehicle.crashed 
        self.speed_rew =   np.exp(-0.05*(self.vehicle.speed-12)**2) #1*self.vehicle.speed / 12  #np.exp(-2*(8-self.vehicle.speed)**2)    #
        self.lane_change_rew = 0.5*self.config["lane_change_reward"] * lane_change
        self.obstacle_dist_rew = 1.0*distance_metric
        self.acc_rew = -2e-4*(acc)**2
        
        self.reward_terms = {'collision':self.collision_rew,
                            'speed':self.speed_rew,
                            'lane':self.lane_change_rew,
                            'obstacle':self.obstacle_dist_rew,
                            'acc':self.acc_rew
                            }

        reward = self.collision_rew + self.speed_rew + self.lane_change_rew + self.obstacle_dist_rew + self.acc_rew


        self.prev_speed = self.vehicle.speed
        rew_max = 1 
        rew_min = -6
        reward = np.clip(reward,rew_min,rew_max)
        self.norm_reward = reward#(reward - rew_min)/(rew_max - rew_min)#utils.lmap(reward,[rew_min,rew_max],[0,1])#
        return self.norm_reward

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.steps >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()
        self.prev_speed = 8

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.steps += 1
        self._simulate(action)

        obs = self.observation_type.observe()
        #print(obs)
        #print(direction)
        #input()
        reward = self._reward(action,obs)
        terminal = self._is_terminal()
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
            "observation": obs,
            "reward": self.norm_reward,
            "reward_terms":self.reward_terms,
            "distance_per_bin":self.distance_per_bin,
            "obstacle_distance":self.obstacle_distance,
        }
        
        try:
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass


        return obs, reward, terminal, info

    def _make_road(self) -> None:
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [0, 0]  # [m]
        radius = 20  # [m]
        alpha = 24  # [deg]

        net = RoadNetwork()
        radii = [radius, radius+4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane("se", "ex",
                         CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ex", "ee",
                         CircularLane(center, radii[lane], np.deg2rad(alpha), np.deg2rad(-alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ee", "nx",
                         CircularLane(center, radii[lane], np.deg2rad(-alpha), np.deg2rad(-90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("nx", "ne",
                         CircularLane(center, radii[lane], np.deg2rad(-90 + alpha), np.deg2rad(-90 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ne", "wx",
                         CircularLane(center, radii[lane], np.deg2rad(-90 - alpha), np.deg2rad(-180 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("wx", "we",
                         CircularLane(center, radii[lane], np.deg2rad(-180 + alpha), np.deg2rad(-180 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("we", "sx",
                         CircularLane(center, radii[lane], np.deg2rad(180 - alpha), np.deg2rad(90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("sx", "se",
                         CircularLane(center, radii[lane], np.deg2rad(90 + alpha), np.deg2rad(90 - alpha),
                                      clockwise=False, line_types=line[lane]))

        # Access lanes: (r)oad/(s)ine
        access = 170  # [m]
        dev = 85  # [m]
        a = 5  # [m]
        delta_st = 0.2*dev  # [m]

        delta_en = dev-delta_st
        w = 2*np.pi/dev
        net.add_lane("ser", "ses", StraightLane([2, access], [2, dev/2], line_types=(s, c)))
        net.add_lane("ses", "se", SineLane([2+a, dev/2], [2+a, dev/2-delta_st], a, w, -np.pi/2, line_types=(c, c)))
        net.add_lane("sx", "sxs", SineLane([-2-a, -dev/2+delta_en], [-2-a, dev/2], a, w, -np.pi/2+w*delta_en, line_types=(c, c)))
        net.add_lane("sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c)))

        net.add_lane("eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c)))
        net.add_lane("ees", "ee", SineLane([dev / 2, -2-a], [dev / 2 - delta_st, -2-a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("ex", "exs", SineLane([-dev / 2 + delta_en, 2+a], [dev / 2, 2+a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c)))

        net.add_lane("ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c)))
        net.add_lane("nes", "ne", SineLane([-2 - a, -dev / 2], [-2 - a, -dev / 2 + delta_st], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("nx", "nxs", SineLane([2 + a, dev / 2 - delta_en], [2 + a, -dev / 2], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c)))

        net.add_lane("wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c)))
        net.add_lane("wes", "we", SineLane([-dev / 2, 2+a], [-dev / 2 + delta_st, 2+a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("wx", "wxs", SineLane([dev / 2 - delta_en, -2-a], [-dev / 2, -2-a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c)))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        position_deviation = 2
        speed_deviation = 2

        # Ego-vehicle
        ego_start = ["n","e","w","s"]
        ego_start_loc = self.np_random.choice(ego_start)

        ego_lane = self.road.network.get_lane((ego_start_loc+"er", ego_start_loc+"es", 0))
        ego_vehicle = self.action_type.vehicle_class(self.road,
                                                     ego_lane.position(125, 0),
                                                     speed=8,
                                                     heading=ego_lane.heading_at(140))
        ego_destinations = ["exr", "nxr", "wxr","sxr"]
        try:
            ego_vehicle.plan_route_to(self.np_random.choice(ego_destinations))
        except AttributeError:
            pass
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # Incoming vehicle
        destinations = ["exr", "sxr", "nxr"]
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = other_vehicles_type.make_on_lane(self.road,
                                                   ("we", "sx", 1),
                                                   longitudinal=5 + self.np_random.randn()*position_deviation,
                                                   speed=10 + self.np_random.randn() * speed_deviation)

        if self.config["incoming_vehicle_destination"] is not None:
            destination = destinations[self.config["incoming_vehicle_destination"]]
        else:
            destination = self.np_random.choice(destinations)
        vehicle.plan_route_to(destination)
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in list(range(1, 4)) + list(range(-1, 0)):
            vehicle = other_vehicles_type.make_on_lane(self.road,
                                                       ("we", "sx", 0),
                                                       longitudinal=20*i + self.np_random.randn()*position_deviation,
                                                       speed=10 + self.np_random.randn() * speed_deviation)
            vehicle.plan_route_to(self.np_random.choice(destinations))
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        # Entering vehicle
        vehicle = other_vehicles_type.make_on_lane(self.road,
                                                   ("eer", "ees", 0),
                                                   longitudinal=50 + self.np_random.randn() * position_deviation,
                                                   speed=10 + self.np_random.randn() * speed_deviation)
        vehicle.plan_route_to(self.np_random.choice(destinations))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)





class RoundaboutEnv_SP(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "LidarObservation",
                #"absolute": True,
                #"features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-15, 15], "vy": [-15, 15]},
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": [0,3,8,12]
            },
            "incoming_vehicle_destination": None,
            "collision_reward": -0.1,
            "high_speed_reward": 0.5,
            "right_lane_reward": 0,
            "lane_change_reward": -0.05,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "duration": 20
        })
        return config

    def _reward(self, action: int, obs: np.ndarray) -> float:
        lane_change = action == 0 or action == 2

        # compute the distance here 
        col = 0
        sort_obs = obs[np.argsort(obs[:,col])].copy()

        #print(sort_obs)
        #pick two closest obstacles

        obstacle_distance = sort_obs[:,col]
        #1/(1 + e^(-2*dist)) -- 
        
        #print(obstacle_distance)
        
        distance_metric = 0.
        for i in range(len(obstacle_distance)):
            if obstacle_distance[i] <= 0.25:
                distance_metric += -np.exp(-7.5*obstacle_distance[i])

        #print(distance_metric)
        #input()
        acc = (self.vehicle.speed - self.prev_speed)
        reward = self.config["collision_reward"] * self.vehicle.crashed \
            + 1*self.vehicle.speed / 12 \
            + 0*self.config["lane_change_reward"] * lane_change + 1.25*distance_metric - 5e-6*acc
        
        
        self.prev_speed = self.vehicle.speed
        #print("distance",0.25*distance_metric)
        #print("high speed reward",self.config["high_speed_reward"] * \
        #         MDPVehicle.get_speed_index(self.vehicle) / (MDPVehicle.DEFAULT_TARGET_SPEEDS.size - 1))
        #print("vehicle speed",0.75*self.vehicle.speed/16)
        #print("reward",reward)
        #input()
        rew_max = 1 
        rew_min = -3

        return (reward - rew_min)/(rew_max - rew_min)

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.steps >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()
        self.prev_speed = 8

    def step(self, action):
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.steps += 1
        #self._simulate(action)
        dt = 1 / self.config["simulation_frequency"]
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        
        for frame in range(frames):
            # Forward action to the vehicle
            if action is not None \
                    and not self.config["manual_control"] \
                    and self.time % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                #print(action[index])
                self.action_type.act(action[0]) # sets the ego actions
                    

            index = 0
            for vehicle in self.road.vehicles:
                #print("action :",action)
                if len(action) == 1:
                    vehicle.act(action[0])
                else:
                    vehicle.act(action[index])
                    index+=1
            
    
            
            for vehicle in self.road.vehicles:
                vehicle.step(dt)
            for i, vehicle in enumerate(self.road.vehicles):
                for other in self.road.vehicles[i+1:]:
                    vehicle.handle_collisions(other, dt)
                for other in self.road.objects:
                    vehicle.handle_collisions(other, dt)
            
            self.time += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False

        obs = self.observation_type.observe()

        obs_agents = self.get_lidar_obs_agent()

        obs_list = [obs]

        obs_list.extend(obs_agents)
        
        reward = self._reward(action,obs)
        terminal = self._is_terminal()
        info = self._info(obs, action)

        return obs_list, reward, terminal, info

    def get_lidar_obs_agent(self,):
        lidar_obs_list = []
        maximum_range = 60
        self.cells = 8
        self.angle = 2 * np.pi / self.cells
        self.DISTANCE = 0
        for item in self.road.vehicles[1:]:
            observer_vehicle = item
            origin = item.position
            grid = np.ones((self.cells, 5)) * maximum_range
            origin_velocity = item.velocity
            direction_vectors = []
            for obstacle in self.road.vehicles + self.road.objects:
                if obstacle is observer_vehicle or not obstacle.solid:
                    continue
                center_distance = np.linalg.norm(obstacle.position - origin)
                direction_vectors.append(obstacle.position - origin)
                if center_distance > maximum_range:
                    continue
                center_angle = self.position_to_angle(obstacle.position, origin)
                center_index = self.angle_to_index(center_angle)
                distance = center_distance - obstacle.WIDTH / 2
                if distance <= grid[center_index, self.DISTANCE]:
                    direction = self.index_to_direction(center_index)
                    velocity = (obstacle.velocity - origin_velocity)#.dot(direction)
                    #self.direction[center_index,:] = [direction[0],direction[1]]
                    grid[center_index, :] = [distance,direction[0], direction[1],velocity[0],velocity[1]]

                # Angular sector covered by the obstacle
                corners = utils.rect_corners(obstacle.position, obstacle.LENGTH, obstacle.WIDTH, obstacle.heading)
                angles = [self.position_to_angle(corner, origin) for corner in corners]
                min_angle, max_angle = min(angles), max(angles)
                start, end = self.angle_to_index(min_angle), self.angle_to_index(max_angle)
                if start < end:
                    indexes = np.arange(start, end+1)
                else:
                    indexes = np.hstack([np.arange(start, self.cells), np.arange(0, end + 1)])

                # Actual distance computation for these sections
            
                for index in indexes:
                    direction = self.index_to_direction(index)
                    ray = [origin, origin + maximum_range * direction]
                    distance = utils.distance_to_rect(ray, corners)
                    if distance <= grid[index, self.DISTANCE]:
                        velocity = (obstacle.velocity - origin_velocity)#.dot(direction)
                        #self.direction[index,:] = [direction[0],direction[1]]
                        grid[index, :] = [distance,direction[0], direction[1],velocity[0],velocity[1]]

            grid /= maximum_range
            lidar_obs_list.append(grid)

        return lidar_obs_list

    def position_to_angle(self, position: np.ndarray, origin: np.ndarray) -> float:
        return np.arctan2(position[1] - origin[1], position[0] - origin[0]) + self.angle/2

    def position_to_index(self, position: np.ndarray, origin: np.ndarray) -> int:
        return self.angle_to_index(self.position_to_angle(position, origin))

    def angle_to_index(self, angle: float) -> int:
        return int(np.floor(angle / self.angle)) % self.cells

    def index_to_direction(self, index: int) -> np.ndarray:
        return np.array([np.cos(index * self.angle), np.sin(index * self.angle)])
   
    def _make_road(self) -> None:
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [0, 0]  # [m]
        radius = 20  # [m]
        alpha = 24  # [deg]

        net = RoadNetwork()
        radii = [radius, radius+4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane("se", "ex",
                         CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ex", "ee",
                         CircularLane(center, radii[lane], np.deg2rad(alpha), np.deg2rad(-alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ee", "nx",
                         CircularLane(center, radii[lane], np.deg2rad(-alpha), np.deg2rad(-90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("nx", "ne",
                         CircularLane(center, radii[lane], np.deg2rad(-90 + alpha), np.deg2rad(-90 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ne", "wx",
                         CircularLane(center, radii[lane], np.deg2rad(-90 - alpha), np.deg2rad(-180 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("wx", "we",
                         CircularLane(center, radii[lane], np.deg2rad(-180 + alpha), np.deg2rad(-180 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("we", "sx",
                         CircularLane(center, radii[lane], np.deg2rad(180 - alpha), np.deg2rad(90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("sx", "se",
                         CircularLane(center, radii[lane], np.deg2rad(90 + alpha), np.deg2rad(90 - alpha),
                                      clockwise=False, line_types=line[lane]))

        # Access lanes: (r)oad/(s)ine
        access = 170  # [m]
        dev = 85  # [m]
        a = 5  # [m]
        delta_st = 0.2*dev  # [m]

        delta_en = dev-delta_st
        w = 2*np.pi/dev
        net.add_lane("ser", "ses", StraightLane([2, access], [2, dev/2], line_types=(s, c)))
        net.add_lane("ses", "se", SineLane([2+a, dev/2], [2+a, dev/2-delta_st], a, w, -np.pi/2, line_types=(c, c)))
        net.add_lane("sx", "sxs", SineLane([-2-a, -dev/2+delta_en], [-2-a, dev/2], a, w, -np.pi/2+w*delta_en, line_types=(c, c)))
        net.add_lane("sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c)))

        net.add_lane("eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c)))
        net.add_lane("ees", "ee", SineLane([dev / 2, -2-a], [dev / 2 - delta_st, -2-a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("ex", "exs", SineLane([-dev / 2 + delta_en, 2+a], [dev / 2, 2+a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c)))

        net.add_lane("ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c)))
        net.add_lane("nes", "ne", SineLane([-2 - a, -dev / 2], [-2 - a, -dev / 2 + delta_st], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("nx", "nxs", SineLane([2 + a, dev / 2 - delta_en], [2 + a, -dev / 2], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c)))

        net.add_lane("wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c)))
        net.add_lane("wes", "we", SineLane([-dev / 2, 2+a], [-dev / 2 + delta_st, 2+a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("wx", "wxs", SineLane([dev / 2 - delta_en, -2-a], [-dev / 2, -2-a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c)))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        position_deviation = 2
        speed_deviation = 2

        # Ego-vehicle
        ego_lane = self.road.network.get_lane(("ser", "ses", 0))
        ego_vehicle = self.action_type.vehicle_class(self.road,
                                                     ego_lane.position(125, 0),
                                                     speed=8,
                                                     heading=ego_lane.heading_at(140))
        ego_destinations = ["exr", "nxr","wxr"]
        try:
            ego_vehicle.plan_route_to(self.np_random.choice(ego_destinations))
        except AttributeError:
            pass
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # Incoming vehicle
        destinations = ["exr", "sxr", "nxr"]
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = self.action_type.vehicle_class(self.road,
                                                   [-19.74606815,13.64158322],
                                                   #longitudinal=5 + self.np_random.randn()*position_deviation,
                                                   speed=8 + self.np_random.randn() * speed_deviation,
                                                   target_speeds=[0,3,8,12])

        if self.config["incoming_vehicle_destination"] is not None:
            destination = destinations[self.config["incoming_vehicle_destination"]]
        else:
            destination = self.np_random.choice(destinations)
        vehicle.plan_route_to(destination)
        #vehicle.randomize_behavior()
        
        self.road.vehicles.append(vehicle)
        #[ 2. 45.]
        #[-17.32966062  16.60370027]
        #[-0.86433742 19.98131429]
        #[16.62757063 11.11413041]
        #[-15.61261731 -12.49984722]
        #[115.35446052  -2.        ]
    
        vehicle = self.action_type.vehicle_class(self.road,
                                                   [-0.86433742, 19.98131429],
                                                   #longitudinal=5 + self.np_random.randn()*position_deviation,
                                                   speed=8 + self.np_random.randn() * speed_deviation,
                                                   target_speeds=[0,3,8,12])

        if self.config["incoming_vehicle_destination"] is not None:
            destination = destinations[self.config["incoming_vehicle_destination"]]
        else:
            destination = self.np_random.choice(destinations)
        vehicle.plan_route_to(destination)
        #vehicle.randomize_behavior()
        
        self.road.vehicles.append(vehicle)

        
        vehicle = self.action_type.vehicle_class(self.road,
                                                   [16.62757063, 11.11413041],
                                                   #longitudinal=5 + self.np_random.randn()*position_deviation,
                                                   speed=8 + self.np_random.randn() * speed_deviation,
                                                   target_speeds=[0,3,8,12])

        if self.config["incoming_vehicle_destination"] is not None:
            destination = destinations[self.config["incoming_vehicle_destination"]]
        else:
            destination = self.np_random.choice(destinations)
        vehicle.plan_route_to(destination)
        #vehicle.randomize_behavior()
        
        self.road.vehicles.append(vehicle)

        '''
        vehicle = self.action_type.vehicle_class(self.road,
                                                   [115.35446052, -2.],
                                                   #longitudinal=5 + self.np_random.randn()*position_deviation,
                                                   speed=8 + self.np_random.randn() * speed_deviation,
                                                   target_speeds=[0,3,6,9,12])

        if self.config["incoming_vehicle_destination"] is not None:
            destination = destinations[self.config["incoming_vehicle_destination"]]
        else:
            destination = self.np_random.choice(destinations)
        vehicle.plan_route_to(destination)
        #vehicle.randomize_behavior()
        
        self.road.vehicles.append(vehicle)
        # Other vehicles
        '''
        for i in list(range(-1, 0)):# + list(range(1, 2)):
            vehicle = other_vehicles_type.make_on_lane(self.road,
                                                       ("we", "sx", 0),
                                                       longitudinal=20*i + self.np_random.randn()*position_deviation,
                                                       speed=8 + self.np_random.randn() * speed_deviation)
            vehicle.plan_route_to(self.np_random.choice(destinations))
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)
        
        # Entering vehicle
        vehicle = other_vehicles_type.make_on_lane(self.road,
                                                   ("eer", "ees", 0),
                                                   longitudinal=50 + self.np_random.randn() * position_deviation,
                                                   speed=8 + self.np_random.randn() * speed_deviation)
        vehicle.plan_route_to(self.np_random.choice(destinations))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        
        #for vehicle in self.road.vehicles:
        #    print("position",vehicle.position)

        #input()


register(
    id='roundabout-v0',
    entry_point='highway_env.envs:RoundaboutEnv',
)

register(
    id='roundabout-v1',
    entry_point='highway_env.envs:RoundaboutEnv_SP',
)

