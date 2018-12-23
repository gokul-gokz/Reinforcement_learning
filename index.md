# Reinforcement learning:
	Learn what action to take in a state(policy) through reward function.
## Main Charactersistics:
1.Trial and error search
2.Delayed rewards

## How is it different from other ML algorithms?

Key difference:

	1. Explore and exploitation.<br/>
	2. Learn at the same time maximize rewards
	
Supervised Learning:

	1.Learning from trained examples provided by a supervisor. Classify/extraoplate.  
	2.But in learning from interaction(RL), impossible to know all possible situation/representation.  

Unsupervised Learning:<br/>
	
	1.Finds patterns in the problem whereas RL learns howto maximize rewards.

Evolutionary vs learning from value function:
	  
	1.Evolutionary: Rewards are given only at the end of the step.  
	2.Value functions: Learns from each step.

## Components of RL:
**Policy**: What action to take in state.<br/>
**Rewards**: Response which the system gets for performing a action on a state(intrinsic desirablity of a state)<br/>
**Value function**: Long term desirablity of states (total reward accumulated over future from start to goal) actions are chosen which maximizes the value and not the immediate rewards.<br/>
**Model**:Based on the action performed on the current state, resultant states will be predicted.<br/>

## Multi arm bandit problem:
__ Given a slot machine with n arms and each arm has its own probablity distribution of success. Pulling any one of it will give a stochastic reward. The objective is to collect maximum reward in the long run. __

![alt text](http://path/to/img.jpg "Multi_arm_bandit")