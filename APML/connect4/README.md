# Connect4

Implemented a game player using RL with Deep-Q-Learning

![Connect4%20119b202c5e504ce3b5a3d02f01a81a67/Untitled.png](Connect4%20119b202c5e504ce3b5a3d02f01a81a67/Untitled.png)

We had a long discussion on what our model of learning should be. After we went over the lectures
and the exam limitations, we came to conclusion that we won’t be able to use an implementation that might explode with the number of states, take too much time to act or learn. Guided by those, we decided to implement the deep-Q learning model that use a neural network to approximate Q and showed relatively good results in pong with a little amount of time. 

The Deep-Q-Learning model is
based on using gradient descent in order to minimize δst ,at which is composed of :

- The Q function at the previous state and action.
- The reward for the previous action plus the optimal Q value for the new state with some
action times the discounted reward constant.

![Answers](Answers.pdf)