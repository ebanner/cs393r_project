# ml
For my final project for Peter Stone's Autonomous Robots class, I'm going to be developing an improved goalie behavior. To accomplish this, instead of hand-tuning velocity thresholds, we treat shot detection as a classification problem. Here are the steps I will proceed in:

1. Generate training data
2. Train a softmax classifier with position and velocity
3. Train a recurrent neural network (RNN) with only position
4. Train a RNN with only frame space features (e.g. ball location and size) and head tilt
5. Train a RNN (and convolutional neural network) with only images

The first step will be accomplished by writing a simple simulator that models ball movement. The training examples are gold-standard ball locations and velocities and the labels are whether the ball is going to go into the goal or not. Additionally, we can treat this as a multi-class classification problem where the labels are the following:

- Shot on goal center
- Shot on goal left
- Shot on goal right
- No shot on goal

In 2, I will train a softmax classifier on this dataset. After validating this classifier on a validation set, I will implement it on the nao. The success of this model will depend on how accurate our Kalman filter is at reliably estimating position and velocity of the ball. Additionally, how accurate the simulated data reflects the field conditions will also make a difference.

In 3, I will train a RNN on position data alone. This has the benefit of eliminating the noisy velocity estimates from the shot detection pipeline. The use of a RNN is key because the RNN is able to condition its prediction on previous inputs. No model that makes a markov assumption and gets only position data will be able to predict shots on goal with reasonable accuracy.

In 4, we will forgoe the Kalman filter estimates all together and detect shots on goal by visual data and head angles alone. This has the benefit of removing the innaccuracies of the Kalman filter all together. However, this step will require a data collection process, which could be time consuming. I do not expect to reach this state in time for the final project deadline.

In 5, I will have reached the holy grail. By throwing away all intermediate noisy calculations (e.g. ball location, size, and head angles), I will be making predictions solely from visual data. This stage is the most difficult of all. A CNN will most likely have to be trained to extract good features (possibly unsupervised on past robocup games) and the training data from 4 will be used.

As a side note, I will be implementing all of the models I use from scratch. Here are the ones I have planned:

- Linear Regression
- Softmax
- Fully-Connected Neural Network
- Recurrent Neural Network

I'm implementing linear regression because it's a good exercise.

For each classifier, in order to gain intution, I'll implement a 1d version of it (no vectors nor matrics) and then extend it to handle arbitrarily-sized input. I was heavily inspired to take this approach by [Andrev Karpathy's Backpropagation tutorial](http://cs231n.github.io/optimization-2/). Thanks Andrev!
