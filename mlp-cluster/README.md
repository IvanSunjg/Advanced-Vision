# Experiment Workflow

We have built a workflow that allows us to very easily share and run experiments on the MLP cluster. The workflow is built with a focus on being able to submit experiments from the command line without having to edit any scripts multiple times. This has proved to be very helpful in running multiple experiments in parallel across teammates when hyperparameter tuning for our ablation study.

## Setup

1. Set up the MLP cluster with the relevant conda environment and pytorch packages following this [guide](mlp_cluster_guide.md).

## Running an Experiment

1. Make sure that your `avgmentations` library is updated:

    ```sh
    pip install 'git+https://github.com/IvanSunjg/Advanced-Vision.git#egg=avgmentations&subdirectory=avgmentations'
    ```

2. Make sure that you are in the `Advanced-Vision/mlp-cluster/` directory.
3. Make sure that the image training folder, `train/`, and the ResNet50 pretrained model, `resnet50_fconv_model_best.pth.tar`, are located in the `Advanced-Vision/mlp-cluster/` directory.
4. Define your experiment in [`experiment.py`](experiment.py)
    * To define an experiment, you have to add a function to [`experiment.py`](experiment.py) with the name of your experiment, its hyperparameters as the function parameters, and have the return value be an "update-transforms" Python dictionary.
        1. **Naming Convention**: For experiment names, we would like to use the following convention to make it clear what augmentations the experiment performs and how they are linked.
            * Non-basic data augmentations are written in lowercase and joined using underscores in the order they are applied, i.e. if an experiment applies `AugMix` and then `CutOut` on all images, then its name would be `augmix_cutout`.
            * Different augmentations that are used during different epochs are joined using `then` and underscores, i.e. if an experiment applies `MixUp` for the first few epochs and then applies no advanced augmentations (aka `default`) afterwards, then its name would be `mixup_then_default`.
        2. **"update-transforms" Python dictionary**: The "update-transforms" Python dictionary is a dictionary that defines what augmentations to apply at what epoch. The keys are integers that define the epoch at which a transform is applied, and the value for each epoch key is the transform to apply at that epoch. The transform can be any augmentation(s) from `torchvision.transforms`, `albumentations` (untested), or `avgmentations`.
        3. **Examples**

            For the following examples, assume that these imports and code have been run:

            ```python
            from avgmentations import augmentations as A
            from avgmentations.resnet_dataset import RESNET_MEAN, RESNET_STD
            from torchvision import transforms
            from torchvision.datasets import ImageFolder

            default_dataset = ImageFolder(
              root = root,
              transform = A.Compose(
                transforms.ToTensor(),
                transforms.Normalize(RESNET_MEAN, RESNET_STD),
                transforms.Resize(256),
                transforms.CenterCrop(224),
              ]),
              target_transform = A.OneHot(n_classes)
            )
            ```

            1. In this example, we will define an experiment that applies just `CutOut` on all images at all epochs.

                ```python
                def cutout(self, n_holes=3, length=50):
                  exp = {
                    0: A.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(RESNET_MEAN, RESNET_STD),
                      transforms.Resize(256),
                      transforms.RandomCrop(224),

                      A.CutOut(n_holes=n_holes, length=length)
                    ])
                  }
                  return exp
                ```

            2. In this example, we will define an experiment that applies `AugMix` then `CutOut` on all images at all epochs.

                ```python
                def augmix_cutout(self, k=3, w1=0.2, w2=0.3, w3=0.5, m=0.2, n_holes=3, length=50):
                  exp = {
                    0: A.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(RESNET_MEAN, RESNET_STD),
                      transforms.Resize(256),
                      transforms.RandomCrop(224),

                      A.AugMix(k=k, w1=w1, w2=w2, m=m)
                      A.CutOut(n_holes=n_holes, length=length)
                    ])
                  }
                  return exp
                ```

            3. In this example, we will define an experiment that applies `MixUp` until epoch `n1`, and then applies basic augmentations followed by `GridMask` until epoch `n2`, and finally applying no augmentations (aka `default`) for the rest of the epochs.

                ```python
                def mixup_then_basic_gridmask_then_default(self, n1, n2, alpha=1.0, min_lam=0, max_lam=1, sharpnesss_factor=2):
                  exp = {
                    0: A.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(RESNET_MEAN, RESNET_STD),
                      transforms.Resize(256),
                      transforms.RandomCrop(224),

                      A.MixUp(default_dataset, alpha=alpha, min_lam=min_lam, max_lam=max_lam)
                    ]),
                    n1: A.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(RESNET_MEAN, RESNET_STD),
                      transforms.Resize(256),
                      transforms.RandomCrop(224),

                      transforms.RandomHorizontalFlip(),
                      transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor),

                      A.GridMask()
                    ])
                    n2: A.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(RESNET_MEAN, RESNET_STD),
                      transforms.Resize(256),
                      transforms.RandomCrop(224),
                    ])
                  }
                  return exp
                ```

5. Run Your Experiment
    * To submit and run experiment on the cluster, you have to run `bash send_model_to_job.sh` with flags that specify exactly what experiment you would like to run and what training hyperparameters you would like to use.
        1. **Experiment Specifications**: To specify the type of experiment you would like to run, you must use the `--exp_type` and `--exp_kwargs` flags
            1. `--exp_type`: Short for "experiment type", this flag defines the type of experiment you would like to run and the supplied argument should be the name of an experiment defined in [`experiment.py`](experiment.py). For example, if we wanted to run the `AugMix` and `CutOut` experiment defined in the second example above, we supply *augmix_cutout* for the `--exp_type` flag. Notice that the supplied argument matches the exact name used for the experiment's function definition.
            2. `--exp_kwargs`: Short for "experiment keyword arguments", this flag defines the parameters you would like to pass to the experiment as defined in the experiment's function definition. The passed argument needs to be a string containing a Python dictionary of the parameters you would like to manipulate, and their corresponding value. For example, if we wanted to use only 2 augmentations for `AugMix` and 2 holes of length 80 for `CutOut` in the `CutOut` and `AugMix` experiment defined in the second example above, we would supply *"{'k': 2, 'n_holes': 2, 'length': 80}"* for the `--exp_kwargs` flag. Notice that the supplied argument is wrapped in quotes, the keys of dictionary are strings, and that those keys match the specificed parameter names in the function definition.

            Together, you can run specific experiments with the default hyperparameters defined in [`arg_extractor.py`](arg_extractor.py) by only specifying the `--exp_type` flag and optionally the `--exp_kwargs` flag depending on if default values have been specified for the experiment's function parameters. Following the examples just discussed you can run the customized `AugMix` and `CutOut` experiment using:

            ```bash
            bash send_model_to_job.sh --exp_type augmix_cutout --exp_kwargs "{'k': 2, 'n_holes': 2, 'length': 80}"
            ```

        2. **Training Hyperparameters**: To specify the model training hyperparameters, you can use the flags defined in [`arg_extractor.py`](arg_extractor.py). These can be used to specify the learning rate, step size, gamma, weight decay, loss function, etc.

        All together, you can run specified experiments using supplied training hyperparameters with `bash send_model_to_job.sh` and the relevant flags as explained above. This will start send the experiment to the Slurm cluster and start it automatically. For example, to send the `AugMix` and `CutOut` example experiment with a learning rate of 0.1 to Slurm, you can run the following:

        ```bash
        bash send_model_to_job.sh --exp_type augmix_cutout --exp_kwargs "{'k': 2, 'n_holes': 2, 'length': 80}" --lr 0.1
        ```

6. Interpret Output and Errors
    * When you have successfully submitted an experiment job, you will be supplied with `job_id` which you can also find using `squeue -u $USER` to view your job's info. The output of the job will be directed to `slurm-{job_id}.out` and the errors of the job will be directed to `slurm-{job_id}.err`.
    * The first line of the output file will show which experiment type and experiment keyword arguments were used. The second line should print all of the hyperparameter and flag values as designated by our [argument extractor](arg_extractor.py).
7. Share Experiments for Other to Run
    * Some experiments, such as `MixUp` and `CutMix`, will take longer than others to run and so having multiple people testing the same base experiment but with different hyperparameters in parallel will be very beneficial. Using this framework, this can be done by just sending a colleague the `--exp_type` and `--exp_kwargs` flags and syncing on what hyperparameters you guys would like to test in parallel.

Hopefully this helps provide insight into how to submit experiments using our new workflow :)
