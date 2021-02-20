# AA_experiment
 
The code in this repository presents an initial attempt to make a simple CNN model (LeNet) more
robust against adversarial attacks. Specifically, we consider the performance of our method to
guard the model against a Fast Gradient Sign Attack (FGSM) in application to MNIST.

This is done in quite a simple manner: after the convolutional layers of the LeNet, the image
is flattened into a 320-dimensional vector. During training of the LeNet with "clean" examples,
we use this vector to also train another MNIST classifier that directly takes it as an input.

The advantage of this second classifier is that it can help update images' encoded representations
during application of the LeNet. This is done by subtracting (a multiple of) the gradient of the
LeNet-derived vector with respect to the loss of the second classifier, before classifying the
vector using the rest of LeNet.

Since adversarial attacks like FGSM often perturb inputs to maximize classifier loss, this vector
update should counteract these perturbations somewhat. Initial experiments show modest improvements
in accuracy of 10-20%.

There is potential for more improvement, however: we can consider adding more intermediate update
layers, instead of just one (earlier updates should make our model more robust). We can also
consider changing the manner in which we update - currently we only subtract the update network's
gradient from the input once.

Some of the code in this repository is built off of the [Pytorch tutorial on FGSMs](https://github.com/pytorch/tutorials/blob/master/beginner_source/fgsm_tutorial.py).
