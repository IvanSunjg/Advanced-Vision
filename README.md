# Image Classification on Limited Data

For the University of Edinburgh's Advanced Vision coursework, we were tasked with performing image classification on a limited dataset. More specifically, we performed image classification on a dataset that contained 1000 classes with 50 images per class.

For image classification, we were restricted to using ResNet50 as an immutable backbone and to try to improve classification accuracy from the baseline 31%. To do so, we experimented with various hyperparameter combinations and data augmentation techniques, going so far as to write our own data augmentation library.

## Repository Structure

* avgmentations
  * Our custom data augmentation library. Builds on existing pytorch and torchvision classes to implement more advanced transformations and to also allow the modification of transformations across training epochs
* colab
  * Our Google Colab and Jupyter Notebook workflow. Most of this work can be found in `colab/examples.ipynb`
* mlp-cluster
  * Our MLP Cluster workflow, using the school's cluster instead of Google Colab as the free version of Google Colab was very restricting. This is where most of work can be found.
