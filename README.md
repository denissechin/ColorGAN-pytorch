# ColorGAN
***
 GAN with ResnetUnet backbone for image colorizing

# Overview
***
- Dataset: Flickr30k
- Development environment:

Python 3.7.9

Torch 1.7.0

# Result
***
Model was trained for 15 epochs

Left image is a predicted one, right - ground truth
- Good results

![alt](./img/example3.png)

![alt](./img/example2.png)

- Bad results

![alt](./img/example1.png)

![alt](./img/example6.png)

- Performance on old gray images

![alt](./img/example4.png)

![alt](./img/example5.png)


# TODO

***
- Try augmentations
- Train more epochs
- Train with higher resolution
- Implement comfortable prediction code
