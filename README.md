# A Better Goalie Behavior
For my final project for Peter Stone's Autonomous Robots class, I'm going to be developing an improved goalie behavior. To accomplish this, instead of hand-tuning velocity thresholds, we treat shot detection as a classification problem. View the wiki on the right hand side for more details.

As a side note, I will be implementing all of the models I use from scratch. Here are the ones I have planned:

- Linear Regression
- Softmax
- Fully-Connected Neural Network
- Recurrent Neural Network

I'm implementing linear regression and a fully-connected neural network because they're building blocks to implement softmax and RNNs, respectively.

For each classifier, in order to gain intution, I'll implement a 1d version of it (no vectors nor matrics) and then extend it to handle arbitrarily-sized input. I was heavily inspired to take this approach by [Andrev Karpathy's Backpropagation tutorial](http://cs231n.github.io/optimization-2/). Thanks Andrev!
