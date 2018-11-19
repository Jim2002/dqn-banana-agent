## Deep Reinforcement Learning - Bananas

### Model Architecture
The Udacity provided Deep Q-Network code in PyTorch was used and adapted for this environment. 

I used a simple, two hidden layer, 128-64 node, nn with ReLU activation functions. 


### Hyperparameters
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network


#### Exploitation vs. Exploration

I found in this environment that, relative to other DRL agents I have trained, the agent does not need to try nearly as many random actions to gather the necessary experience to act efficiently. Therefore, I have weighted exploitation heavily from an early stage while keeping a minimal amount of exploration. I used 1.0 starting epsilon and 0.01 minimum in the default code. Decay at 0.995

#### Experience Memory

In optimizing this agent, I found that the agent learns optimal actions fairly quickly, as described above, and therefore, I had the agent do a soft update of the network after every agent action. This approach, as opposed to only updating the network after every 4 actions in the default code, was far superior, cutting the training time by more than half. This change, when combined with the epsilon changes above, have a dramatic effect on speed of learning.


## Results and Future Work

The above approach produced an average reward of 13+ in 396 episodes. 

Much of the improvement and variation in results centers around how much, how fast and how often to use the agent's past experiences to train the network and improve the action policy. Future work focused on this area could yield improved results, namely Prioritized Experience Replay or other methods of selecting/optimizing which experiences to utilize for training purposes.
