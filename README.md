# pixelz-pytorch-tools
Some tools that we use at Pixelz for training with pytorch and a new dataset class with support for each iteration having multiple images. They combine features from Caffe and Pytorch. Simply copy the file `train_pixelz.py` in your working directory (or anywhere else approriate) and use `
import train_pixelz`

## Training utilities
### Description
The main function is:
```python
train_pixelz.train_model(model, optimizer, scheduler, dataset, batch_size, shuffle=False, \
                         snaphot_prefix = None, num_iters=None, \
                         num_epochs=None, iter_size=1, display_iter = 20, snaphost_interval=None, \
                         load_snapshot_path=None, restore_dataloader=True, display_gpu=False, schedule_on_iter=False)
```

It allows the user to easily train a model with a dataset without the hassle of coding himself the training loop. It contains the possibility to spread the batch size over several iterations iter_size, the possibility to schedule per epoch or per iteration schedule_on_iter, the possibility ot train for a set number of iterations or epochs and the possibility to save and reload snapshots containing the model, scheduler, and optimizer states including a few other parameters to properly restart from the snapshot (except for the dataloader since it does not have a state dict).
There is a requirement that the model contains its own loss function.
It does only training, not validation since usually do not want to run inference load on a large training server (waste of computing time)

And the learning rate finder is:
```
train_pixelz.find_lr(model, optimizer, dataset, batch_size, shuffle=False, iter_size=1, load_snapshot_path=None, restore_dataloader=True,
                     min_lr=1e-10, max_lr=1, iter_per_run=50, num_runs=30, result_file="results.npz")
```
It will perform an exponential sampling (e.g. 1e-6, 1e-5, 1e-4,...) of `num_runs` samples between `min_lr` and `max_lr`. For each sample run, the dataset will be reset and `iter_per_run` iterations will be performed.

Both also available for multi node, multi gpu training with automated support for sync batch norm (with simple distributed configuration). It is using DistributedDataParallel with one process per GPU. It will allocate up to 6 cores per GPU.
See the line `num_workers = min(max(nprocs // ngpus_per_node, 1), 6)  # no need to have more than 6 workers per gpu` if you need to change this.

And a utility to easily configure learning rates for each parameter: make_param_groups.

We tried contributing this to Pytorch but this kind of tool is not supposed to be in it. See [this discussion](https://github.com/pytorch/pytorch/issues/25986 "Training and learning rate finder utilities").

### Example usage:
Configure optimizer and learning rates and train
```python
custom_lr = { \
    'conv12.weight' : 1e-3, \
    'conv12.bias' : 1e-3  \
}
optimizer = optim.SGD(train_pixelz.make_param_groups(model, custom_lr), lr=1e-2)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
model = train_pixelz.train_model(model, optimizer, scheduler, train_dataset, \
                                 batch_size=6, shuffle=False, num_epochs=5, iter_size=2, schedule_on_iter=False, \
                                 snaphost_interval=0.5, snaphot_prefix='/data/pixelz_train_6/snapshot')
```

Configure multi gpu settings on a single node and train
```python
os.environ["RANK"] = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "35003"
os.environ["WORLD_SIZE"] = "1"

model = train_pixelz.train_model_multigpu(model, optimizer, scheduler, train_dataset, \
                                          batch_size=6, shuffle=False, num_epochs=5, iter_size=2, schedule_on_iter=False, \
                                          snaphost_interval=0.5, snaphot_prefix='/data/pixelz_train_6/snapshot')
```
The learning rate finder version:
```python
optimizer = optim.SGD(train_pixelz.make_param_groups(model, custom_lr), lr=1e-6)
model = train_pixelz.find_lr(model, optimizer, train_dataset, \
                             batch_size=6, shuffle=False, num_runs=15, min_lr=1e-6, max_lr=1e-1, iter_size=3)
```
With multi-gpu support
```python
os.environ["RANK"] = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "35003"
os.environ["WORLD_SIZE"] = "1"
model = train_pixelz.find_lr_multigpu(model, optimizer, train_dataset, \
                                      batch_size=6, shuffle=False, num_runs=15, min_lr=1e-6, max_lr=1e-1, iter_size=3)
```
The optimizer MUST have a learning rate equal to `min_lr` for this to work properly.

So far we used it for single or multi gpu on a single node, with several computer vision architectures (VGG, resnet and variants, etc) without problem. Feel free to open an issue if there is one.


### Limitations
- the dataset needs to be of constant type for torch.utils.data.distributed.DistributedSampler
- since in the distributed case, we need a specific sampler (and we wanted to handle that automatically), and we wanted to keep the same function signature, the user cannot provide his/her own dataloader and sampler. We could enable the user to provide the dataloader instead of dataset, batch_size, shuffle=False either for only train_model or for both APIs. In the multi-gpu case, then it would be the user's responsibility to provide the appropriate sampler and he would need to change it between single and multi gpu cases.
- we use the following trick to pass an arbitrary dataset to the neural network so there should not be any other limitation to the dataset
```python
for inputs in data_loader:
    loss = model(*inputs) / iter_size
```
- the model forward function must only take as inputs the object(s) returned by a dataset iteration in the same order. If the model needs other variables that are constant throughout the training, they can be passed at model initialization. If the other variables depend on factors unrelated to the model, they (or the factors) can be in the dataset and if they vary based on the model, then the model can compute them internally.
- as already said, the loss computation must be done inside the model forward function
- we only tested the case where the model returns a single one valued tensor (e.g. a single number, the loss) but when going over the code, there should not be a problem as long as the model returns a single tensor (not necessarily single valued) since this value is only used to compute and display the loss.
- the API only supports a single optimizer (cannot use two optimizers at the same time) but you can always combine several optimizers into a custom one

### Some extra stuff that it cannot do as well:
- criterion to stop when loss is too low (easy to add as yet another param)
- optimizer closure (could be added as a parameter? can we define the closure outside of the training loop?)
- gradient clipping (not trivial to add as the generic functionality is adding the possibility to have some extra processing function at the end of each step/epoch)
- dataset reset for torchtext after each epoch (same issue as grad clipping)
- hogwild and other custom distributed settings (not even trying to add this)
- gan (that's a case on its own, not trying it)
- reinforcement learning would be possible but ugly because of the need to include part of the training loop in the network



## Dataset utility
The class `MultiImgsDataset` enables having a dataset with several input and output images for each iteration. The image files should be in separate folders (one folder per input or output) and all the files for a given iteration are assumed to have the same name (discarding the file extension). It should still work as long as the sorting algorithm of python keeps them in the same order.
The directories must be arranged in this way:
```
        dir1/123.png
        dir1/456.png
        dir1/789.png
        dir1/999.png

        dir2/abc.png
        dir2/def.png
        dir2/ghi.png
        dir2/jkl.png
```
Although it is safer to name the files
```
        dir1/0001_123.png
        dir1/0002_456.png
        dir1/0003_789.png
        dir1/0004_999.png

        dir2/0001_abc.png
        dir2/0002_def.png
        dir2/0003_ghi.png
        dir2/0004_jkl.png
```

It also contains a fast jpeg image loader `train_pixelz.fast_img_loader` based on libturbo jpeg and [jpeg4py](https://github.com/ajkxyz/jpeg4py) but it is not mandatory to use it.

We tried contributing this to pytorch but it was not generic enough to be added in it. See [this pull request](https://github.com/pytorch/vision/pull/1345 "Add MultiImageFolder dataset") and [this discussion](https://github.com/pytorch/vision/issues/1406 "[RFC] Abstractions for segmentation / detection transforms").

### Example usage
```python
train_transforms = [
    transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [1, 1, 1])]),  # input 1 color image
    transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize([0.424], [1])]),  # intput 2 graylevel image
    transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize([0.424], [1])]),  # intput 3 graylevel image
    transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]), # groundtruth 1
    transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])  # groundtruth 2
]

train_data_dirs = ['path/to/input1',
                    'path/to/input2',
                    'path/to/input3',
                    'path/to/groundtruth1',
                    'path/to/groundtruth2'
                    ]
train_dataset = train_pixelz.MultiImgsDataset(train_data_dirs, train_transforms, train_pixelz.fast_img_loader)
```
Without the custom loader:
```
train_dataset = train_pixelz.MultiImgsDataset(train_data_dirs, train_transforms)
```
