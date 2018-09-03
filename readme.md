# Active learning - lessons from a day of hacking

On August 24, we set out to play with active learning. Jetze, Ruben and Rob met at PyData Amsterdam 2018 conference and planned a day of hacking together. Being in the office at Greenhouse Group, Nico, Charlotte, Stijn, Tim, Esmee and Charlotte joined us.

# Motivation for active learning
For this days work, we set out to find a motivating problem. And settled on Active Learning, because of its practical implication. In general, active learning plays the following game: Say you have a training data set of inputs and labels <img alt="$\{(X_i, y_i)\}_{i=0}^{N_t}$" src="https://rawgit.com/RobRomijnders/hacking_180824/master/svgs/efb990abf905b992f6b893d871a2831a.svg?invert_in_darkmode" align=middle width="90.16161pt" height="30.279149999999998pt"/>. A model trained on such data set will achieve (a presumably) low performance. You have access to another pool of only inputs <img alt="$\{(X_i)_{i=0}\}^{N_p}$" src="https://rawgit.com/RobRomijnders/hacking_180824/master/svgs/09655fbd607c3a7a48f14a917697ca38.svg?invert_in_darkmode" align=middle width="86.70238499999999pt" height="27.598230000000008pt"/>. An oracle could label the data points in the pool, but this requires effort/money/time. Then how would you query a data points, such that you ask for the most informative observations for your model?

### Intuition
A good query follows some intuiton. We'll discuss intuition for the classification case in a problem with ten classes.

  * Let's say a current model achieves good performance in classs 5. Then querying a data point from the pool that looks like inputs from class 5 will not bring much information.
  * Let's say a current model achieves bad performance in class 4. Then querying a data point from the pool that looks like inputs from class 4 will bring some information.

Keep in mind that we wouldn't have labels (yet) for the data points in the pool. So we frame our intuition as:

  * Let's say for a data point, our current model predicts high scores for label 7. Then we wouldn't gain much information from the label. We assume our model is already correct.
  * Let's say for a data point, our current model predicts equal scores for any class. Then we would gain much information by querying that label. We assume our model would improve its prediction after it got the label.


# Life cycle of active learning
As we were wrapping our head around active learning, we drew the following cycle:

This cycle has four steps and four arrows:

  * Train model
  	* Run model on labeled data
  * Make predictions
   * Run model on unlabeled data
  * Query policy
  	* Rank observations based on metric (max margin, entropy, least confidence)
  * Retrain model
  	* Run model on labeled data (including newly obtained labeled observations)

# Data set
We use the MNIST data set to experiment with active learning. The MNIST has ten classes, and a total of 60.000 images and labels. As we want to simulate an active learning scenario, we presume a small initial data set. Getting more data incurs a cost. Therefore, the initial training set contains 60 images and labels. Each cycle of Active Learning, the learner may query 20 labels. (One could change this via the `n_instances variable`)

# Code structure
  * `play_data.py` serves as our main script. This script loads the data and sets up the active learner.
    * It writes results to a log. This log will be parsed by the plotting functionality
  * `plotting/main_plot.py` contains most of the plotting functionality.
  * Directory
  	* We assume a `log/` directory to save the logs
  	* We assume a 'data/mnist' directory to find the binaries for the MNIST data set. Download them [here](http://yann.lecun.com/exdb/mnist/)

# Results
In MNIST, balanced training data, we did not see much performance between entropy sampler and random sampler. We expect this comes from the fact that balanced training set in MNIST already creates a decent classifier and every new observation will add roughly the same information.

Next we sampled training observations according to a bias, such that we create a unbalanced training set. Our hypothesis was that the queried observations would turn out to be of the underrepresented classes. It turns out this holds partially. Yes, the smaller classes are queried more, but only until the classifier did a decent job at predicting these classes. This point occured sooner than we expected. This result may be explained by the feature of MNIST that some pairs of classes are hard to distinguish.

In general for MNIST we found better results for max margin policies, than entropy or least confidence. The intuition behind max margin is that uncertainty between 2 classes is penalized as much as uncertainty between k classes. This is likely to result in better queries because in MNIST it is difficult to distinguish between certain sets of classes.

## Comparing performance over lifetime
![comparison](https://github.com/RobRomijnders/hacking_180824/blob/master/hacking_180824/im/performance_comparison_random_margin.png?raw=true)

## Plotting cumulative counts and performance per class
![margin_sampling](https://github.com/RobRomijnders/hacking_180824/blob/master/hacking_180824/im/margin_sampling_cumcounts_performance.png?raw=true)
![random_sampling](https://github.com/RobRomijnders/hacking_180824/blob/master/hacking_180824/im/random_sampling_cumcounts_performance.png?raw=true)

# Further reading

  * [Nice slide deck on active learning](https://www.cs.cmu.edu/~tom/10701_sp11/recitations/Recitation_13.pdf)
  * We got starter code from the [modAL package](https://github.com/cosmic-cortex/modAL)
  * Two papers on metrics to define confusion/uncertainty/information [paper1](http://www.mdpi.com/1099-4300/15/4/1464/pdf) [paper2](https://authors.library.caltech.edu/13795/1/MACnc92c.pdf)
