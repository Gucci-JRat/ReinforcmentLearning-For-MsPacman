# ReinforcmentLearning-For-MsPacman

Hello world, this repository implements reinforcment learning for Ms. Pacman. 
We use Open AI's gym to get Ms. Pacman's environment. 
And the latest version of Tensorflow to train our agent using DeepQNetwork to update QTable.

I have made a youtube video for a detailed explantion of the code : https://www.youtube.com/watch?v=Qqmfqq553RI&t=1s 

You can also read my blog on medium to gain more understading about this repository: 
https://medium.com/@jashr2312/how-to-make-self-learning-games-using-reinforcement-learning-dee2f8904b71

### Packages Required 

- gym
- gym[atari]
- tensorflow
- matplotlib
- collections
- random
- numpy

You can install the package by typing the following command in the terminal. Just replace "package_name" with the name of the package.
```sh
$ pip install "package_name"
```
### How to execute ?
After cloning/downloading the repository. In the terminal, navigate to the directory where you cloned/downloaded the repository and type. 

```sh
$ python Pacman.py
```

### Note
In order to train the model instead of testing it you need to edit the Pacman.py code a bit. For training the model load_model needs to be false.
```sh
load_model=False
```

The major differnce between training and testing is that during training, the model saves the weights after every 50 episodes. However, during testing it is not the case. Moreover, the hyperparameters for the agent class are completely different for training and testing. 

The model needs a good amount of time and episodes to train. I have just trained the model for 1500 games, to give you'll an idea. 1500 episodes is clearly not enough for the network architecture we are using. Fell free to change stuff around to see if you can improve the agent's performance.  
