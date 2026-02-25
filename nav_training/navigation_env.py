from typing import Optional
import numpy as np
import gymnasium as gym


class NavWorldEnv(gym.Env):

    def __init__(self, size: int = 10):
        # The size of the square grid (20x20 by default)
        self.size = size

        # Initialize positions - will be set randomly in reset()
        # Using 0.0, 0.0, 0.0 as "uninitialized" state
        self._agent_location = np.array([0.0, 0.0, 0.0], dtype=float)
        self._target_location = np.array([0.0, 0.0], dtype=float)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(-size, size, shape=(3,), dtype=np.float64),   # [x, y, orient] coordinates
                "target": gym.spaces.Box(-size, size, shape=(2,), dtype=np.float64),  # [x, y] coordinates
                "distance": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "relative_distance_body": gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64), # [dx, dy]
                "relative_yaw_error": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64), # radians
            }
        )

        # Define what actions are available (6 directions)
        self.action_space = gym.spaces.Discrete(7)

        self.distance = 0.0
        self._last_distance = 0.0
        self._step_count = 0

        self.action_to_vel = {
            0: np.array([0.0, 0.0, 0.0]),   # Stay in place
            1: np.array([1.0, 0.0, 0.0]),   # Move forward (x)
            2: np.array([-1.0, 0.0, 0.0]),  # Move backward
            3: np.array([0.0, 1.0, 0.0]),   # Move left (y)
            4: np.array([0.0, -1.0, 0.0]),  # Move right
            5: np.array([0.0, 0.0, 1.0]),   # Rotate counterclockwise
            6: np.array([0.0, 0.0, -1.0])   # Rotate clockwise
        }

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """

        rel_distance_world = self._target_location - self._agent_location[:2]
        self.distance = np.linalg.norm(rel_distance_world)
        abs_distance = np.array([self.distance], dtype=np.float64)

        # relative distance in body frame
        yaw = self._agent_location[2]
        rel_dx = rel_distance_world[0] * np.cos(yaw) + rel_distance_world[1] * np.sin(yaw)
        rel_dy = -rel_distance_world[0] * np.sin(yaw) + rel_distance_world[1] * np.cos(yaw)
        rel_distance_body = np.array([rel_dx, rel_dy], dtype=np.float64)

        # relative yaw error
        desired_yaw = np.arctan2(rel_distance_world[1], rel_distance_world[0])
        yaw_error = desired_yaw - yaw
        yaw_error = (yaw_error + np.pi) % (2*np.pi) - np.pi
        
        rel_yaw_error = np.array([yaw_error], dtype=np.float64)

        return {"agent": self._agent_location, 
                "target": self._target_location, 
                "distance": abs_distance,
                "relative_distance_body": rel_distance_body,
                "relative_yaw_error": rel_yaw_error
                }
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": self.distance
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self._agent_location = np.random.uniform(low=-self.size, high=self.size, size=3)
        self._agent_location[2] = np.random.uniform(low=0.0, high=2*np.pi, size=1)

        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location[:2]
        while np.array_equal(self._target_location, self._agent_location[:2]):
            self._target_location = np.random.uniform(low=-self.size, high=self.size, size=2)

        observation = self._get_obs()
        info = self._get_info()

        self._step_count = 0
        self.distance = np.linalg.norm(self._agent_location[:2] - self._target_location)
        self._last_distance = self.distance

        return observation, info
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-6 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        self._step_count += 1

        # Update agent position, ensuring it stays within grid bounds
        self._agent_location = self._update_location(self._agent_location, action)

        truncated = self._step_count >= 1e3
        #truncated = False

        observation = self._get_obs()
        info = self._get_info()
        
        # Reward
        reward = self._progress_reward() + self._yaw_reward()

        # Update parameters
        self._last_distance = self.distance
        # Check if agent reached the target
        terminated = self.distance <= 0.3


        return observation, reward, terminated, truncated, info
    
    def _update_location(self, pos, action):

        dt = 0.2

        # Update motion
        vel = self.action_to_vel[action]

        dx = pos[0] + (vel[0] * np.cos(pos[2]) - vel[1] * np.sin(pos[2])) * dt
        dy = pos[1] + (vel[0] * np.sin(pos[2]) + vel[1] * np.cos(pos[2])) * dt
        dz = (pos[2] + vel[2] * dt) % (2*np.pi)
        new_pos = np.array([dx, dy, dz], dtype=np.float64)

        new_pos[:2] = np.clip(new_pos[:2], -self.size, self.size)

        return new_pos
    
    def _progress_reward(self):
        progress = self._last_distance - self.distance

        return progress
    
    def _yaw_reward(self):
        curr_yaw = self._agent_location[2]
        rel_distance_world = self._target_location - self._agent_location[:2]
        desired_yaw = np.arctan2(rel_distance_world[1], rel_distance_world[0])

        if desired_yaw < 0.0:
            desired_yaw += 2 * np.pi

        error = desired_yaw - curr_yaw
        error = (error + np.pi) % (2*np.pi) - np.pi # between -pi and pi

        return 0.1 * np.cos(error)