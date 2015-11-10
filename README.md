# ml
For my final project for Peter Stone's Autonomous Robots class, I'm going to be investigating developing a velocity-less goalie behavior (i.e. one that operates solely on global positioning of the ball). To accomplish this, I'll be treating the task of detecting shots on goal as a classification problem (i.e. either the ball is being shot on goal or not).

Because it is impossible to determine whether the ball is being shot on goal or it is just really close with just the most recent location of the ball (i.e. a markov assumption), I'm going to be using a recurrent neural network for this task. The hope is that a RNN will learn to fire when the ball is close AND it is getting closer.

Extensions to this idea include using progressively less and less information to detect a shot on goal. One could imagine backing off to the ball-relative coordinates of the ball, as well as the size of the ball and the pan of the head, with the ultimate goal being to go straight from images to a decision.

Because I'm as interested in the models as I am in the end application, I'm going to be implementing all of the discriminative models. Here are the ones I have planned:

- Linear Regression
- Softmax
- Fully-Connected Neural Network
- Recurrent Neural Network

For each classifier, in order to gain intution, I'll implement a 1d version of it (no vectors nor matrics) and then extend it to handle arbitrarily-sized input. I was heavily inspired to take this approach by [Andrev Karpathy's Backpropagation tutorial](http://cs231n.github.io/optimization-2/). Thanks Andrev!
