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
*Given a slot machine with n arms and each arm has its own probablity distribution of success. Pulling any one of it will give a stochastic reward. The objective is to collect maximum reward in the long run.*

![Bandit](Images/Multi-armed-bandit.jpg "Multi_arm_bandit")

<ins>For Example</ins>:  
Consider a 3 arm bandit problem. When you are starting you dont know which one gives the maximum reward. So, you need to pull all of them atleast once to know the reward of it. 
But the rewards are stochastic, so from one trial it's not possible to estimate the rewards.  
 
<ins>**Traditional method of solving**</ins>:
Pull all the arms to a fixed number of times and get an statistical inference.  
<ins>**Problems**</ins>:  
How to determine the number of times to pull?  
Keeping the n-high will result in less rewards in long run because we would have pulled the arms with less rewards and high rewards equally for most of the times. Not the best strategy.  
Keeping the n-low , estimation might go wrong.

<ins>**RL Strategies**</ins>:  
**Epsilon greedy**:  

* In this strategy , we will choose the best arm(exploit) based on the (mean of rewards) for most of the times, but at the same times we also randomly choose other arms to (explore) the possibilities.  
* When to explore and when to exploit is decided by the epsilon value. 
Example:
Number of trials = 100 epsilon = 10  
Then it means that , choose the best arm(exploit) 90% of times and randomly choose the arm(10%) of times.