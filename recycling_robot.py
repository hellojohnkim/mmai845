import gym
import numpy as np

# Define our actions
SEARCH=0
WAIT=1
RECHARGE=2
ACTIONS=[SEARCH,WAIT,RECHARGE]

# Define our states
LOW=0
HIGH=1

# We define a mapping of actions and states for readability later
ACTION_TO_TEXT={SEARCH:'SEARCH', WAIT:'WAIT', RECHARGE:'RECHARGE'}
STATE_TO_TEXT={LOW:'LOW', HIGH: 'HIGH'}

# An OpenAI Gym implementation of the recycling robot MDP described in
# Reinforcement Learning: An Introduction by Andrew Barto and Richard Sutton
# The example is on page 52 of the 2nd edition
# Developed for MMAI-845
class recyclingRobot(gym.Env):
    def __init__(self,
            alpha=0.5, # Probability of not transitioning to a low battery state after searching in the high reward state
            beta=0.9, # Probability of not depleting the battery if searching in a low reward state
            r_search=8, # Reward for searching for waste to recycle
            r_wait=3, # Reward for waiting and collecting surrounding refuse
            ):

        # We set the number of states internally
        self.num_states = 2 

        # We must set the size of our observations and actions so an agent
        # can be created for the environment
        self.observation_space = gym.spaces.Discrete(self.num_states)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))

        # We set an initial state arbitrarily for our reset function
        # This can be a distribution if we prefer as well
        self.init_state = HIGH

        # Create an empty table to hold our transition probabilities
        self.transition_table = {}
  
        # We enter the transition probabilities from the text for the low
        # energy state
        # Each action has a probability of transitioning to either of the two states
        self.transition_table[LOW] = {
                SEARCH: {LOW: beta, HIGH: (1 - beta)},
                WAIT: {LOW: 1, HIGH: 0},
                RECHARGE: {LOW: 0, HIGH: 1}, 
                }

        # We enter the transition probabilities from the text for the high
        # energy state
        # Since we can't define variable action spaces in gym, we assume a RECHARGE high-energy
        # Keeps the environment in a high-energy state and collects surrounding waste
        # Each action has a probability of transitioning to either of the two states
        self.transition_table[HIGH] = {
                SEARCH: {LOW: (1 - alpha), HIGH: alpha},
                WAIT: {LOW: 0, HIGH: 1},
                RECHARGE: {LOW: 0, HIGH: 1}, 
                }

        # We define a fixed reward table based on the possible transitions
        # It is possible to calculate this purely on the transition without
        # predefining this table as well
        # Some of these are not possible, but we include them in case we would
        # like to change the transitions for some reason later
        self.reward_table = {
                (LOW, WAIT, LOW): r_wait,
                (LOW, WAIT, HIGH): None,
                (LOW, SEARCH, LOW): r_search,
                (LOW, SEARCH, HIGH): -3,
                (LOW, RECHARGE, LOW): None,
                (LOW, RECHARGE, HIGH): 0,
                (HIGH, WAIT, LOW): None,
                (HIGH, WAIT, HIGH): r_wait,
                (HIGH, SEARCH, LOW): r_search,
                (HIGH, SEARCH, HIGH): r_search,
                (HIGH, RECHARGE, LOW): None,
                (HIGH, RECHARGE, HIGH): r_wait,
                }

    # Place us in the initial state
    # This does not need to be deterministic
    # Returns:
    #   obs: an observation of our current state after the reset
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)
        self.state = self.init_state
        return self._get_obs(), {}

    # Get the observation based on our current state
    # This function is simple here, but may be more complex depending on
    # the task
    # Returns:
    #   obs: the observation of the current state
    def _get_obs(self):
        return self.state

    # This function progresses the environment one timestep given the current
    # state and the action. This is where the dynamics are applied.
    # Inputs:
    #   action: The desired action to apply to the environment
    # Returns:
    #   obs: observation of the new state
    #   reward: the reward received for the transition
    #   done: a variable indicating whether we have terminated or not
    #   info: a dictionary data structure containing additional information
    #         about the environment we may want to track
    def step(self, action):
        # Get our current state so we can calculate the reward later
        state = self._get_obs()

        # This task is continuing, so we never terminate, and these variables are
        # always False
        terminated = False
        truncated = False
        

        # We have no additional information to pass back now
        info = {}

        # Get the entry for this state-action pair in our transition table
        transition_entry = self.transition_table[state][action]

        # Since we have a low number of fixed states, we can process the entry
        # into states and probabilities easily directly. With a more complex
        # table, we can iterate over the transitions
        possible_states = [LOW, HIGH]
        state_probabilities = [transition_entry[LOW], transition_entry[HIGH]]

        # We use the numpy library to select the next state according to our
        # probability distribution
        next_state = np.random.choice(possible_states, p=state_probabilities)

        # Now that we have the distribution, we can calculate the reward
        reward = self._get_reward(state, action, next_state)

        # We make sure to update our current state
        self.state = next_state
        return self._get_obs(), reward, terminated, truncated, info 

    # This function calculates the reward for a given_transition
    # Inputs:
    #   state: The current state
    #   action: The action applied
    #   next_state: The next state we enter
    # Returns:
    #   reward: The given reward for the transitions
    def _get_reward(self, state, action, next_state):
        index = (state, action, next_state)
        return self.reward_table[index]

if __name__=='__main__':
    
    # Option 1: We directly create our environment and use it
    print("--------------- Running option 1 ---------------")
    env = recyclingRobot()
    # We need to reset the environment to initialize it
    state, info = env.reset()
    
    # We run for 500 steps to observe the output
    for i in range(25):
        # Select a random action
        # We can implement arbitrary algorithms or strategies in place of this
        action = np.random.choice(ACTIONS)
        # We apply our action and observe the outcome
        next_state, reward, terminated, truncated, info  = env.step(action)
        # We print the transition and reward for visualization
        print("State: {}, Action: {}, State': {}, Reward: {}".format(\
                STATE_TO_TEXT[state], ACTION_TO_TEXT[action], STATE_TO_TEXT[next_state], reward))
        state = next_state

    # Option 2: Register our environment with gym and use this to create the environment
    #           This ensures we are in full compliance with the gym interface and automatically implements some other features,
    #           such as adding episode length limits
    print("--------------- Running option 2 ---------------")
    gym.envs.register(id='RecyclingRobot-v0',
            entry_point='recycling_robot:recyclingRobot',
            max_episode_steps=500)

    # Create the gym environment using the id
    env = gym.make('RecyclingRobot-v0')
        
    state, info = env.reset()

    # We run until termination to observe the output
    for i in range(25):
        # Select a random action
        # We can implement arbitrary algorithms or strategies in place of this
        action = np.random.choice(ACTIONS)
        # We apply our action and observe the outcome
        next_state, reward, terminated, truncated, info = env.step(action)
        # We print the transition and reward for visualization
        print("State: {}, Action: {}, State': {}, Reward: {}".format(\
                STATE_TO_TEXT[state], ACTION_TO_TEXT[action], STATE_TO_TEXT[next_state], reward))
        state = next_state
