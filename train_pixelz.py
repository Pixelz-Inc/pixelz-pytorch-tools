# -*- coding: utf-8 -*-
"""
Created on Tue June 11 14:44:15 2019

@author: sebastien eskenazi

This is a set of utility functions to tain a pytorch network
Notably it contains

- the dataset class MultiImgsDataset to load several
    images in each iteration for networks with multiple image inputs and/or
    images to compute the loss function
- the function train_model to train a model
"""

import os
import os.path
import torch
import torch.utils.data as data
from torch.optim import lr_scheduler
import numpy as np
from torchvision import get_image_backend
import time

from PIL import Image
import jpeg4py as jpeg

import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed


def fast_img_loader(path):
    # Use (wrapper of) libjpeg-turbo for faster JPEG decode
    try:
        img = jpeg.JPEG(path).decode()  # cv2.imread(path)[:,:,::-1]
        img = Image.fromarray(img)
        return img.convert('RGB')
    except Exception:
        # print('Failed to decode image {}: {}'.format(path, e))
        # Fall back to PIL image loader in case of exception
        with open(path.encode('utf-8'), 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


def is_image_file(filename, extensions):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return filename.lower().endswith(extensions)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_dataset(dir, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return is_image_file(x, extensions)
    if not os.path.isdir(dir):
        raise ValueError(dir + " is not a directory")
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                images.append(path)
    return images


_repr_indent = 4
_IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class MultiImageFolder(data.Dataset):
    """A dataset to load multiple images from separate directories at each iteration.
    The directories must be arranged in this way: ::

        dir1/123.png
        dir1/456.png
        dir1/789.png
        dir1/999.png

        dir2/abc.png
        dir2/def.png
        dir2/ghi.png
        dir2/jkl.png

    Args:
        directories (list): List of directories where the images are. They must all contain the same number of images.
        transforms (list of callable, optional): A list of function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        samples (list): List of image path lists [image path 1, image path 2, ...] len[samples][x] = len(directories)
    """

    def __init__(self, directories, transforms=None, loader=default_loader, is_valid_file=None):
        self.directories = [os.path.expanduser(a) for a in directories]
        self.transforms = transforms
        if self.transforms is not None:
            if len(self.directories) != len(self.transforms):
                raise ValueError("There must be exactly one transform per directory or no transform at all.")
        self.loader = loader
        self.extensions = _IMG_EXTENSIONS if is_valid_file is None else None

        sampleslist = [make_dataset(self.directories[i], self.extensions, is_valid_file) for i in range(len(self.directories))]
        for list1 in sampleslist:
            if len(list1) != len(sampleslist[0]):
                raise ValueError("All directories must contain the same number of images.")
            if len(list1) == 0:
                raise (RuntimeError("At least one of the directories does not contain any valid file.\n"
                                    "Supported extensions are: " + ",".join(_IMG_EXTENSIONS)))
        self.samples = list(zip(*sampleslist))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.

        """
        paths = self.samples[index]
        sample = [self.loader(paths[i]) for i in range(len(paths))]
        if self.transforms is not None:
            sample = [self.transforms[i](sample[i]) for i in range(len(sample))]
        return sample

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.directories is not None:
            for directory in self.directories:
                body.append("Directory location: {}".format(directory))
                body += self.extra_repr().splitlines()
        if self.transforms is not None:
            for transform in self.transforms:
                body += [repr(transform)]
                body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self):
        return ""


# wrapper on print to make only one gpu print its output
def print_multigpu(can_display, *args, **kwargs):
    if can_display:
        print(*args, **kwargs)


# restore_dataloader: only useful if restoring from load_snapshot_path, dataloader does not have a state dict that can be saved
#                           hence we compute the number of dataloader iterations already done in the current epoch
#                           and reply them if restore_dataloader is true
# adam_wd: weight decay for adam optimizer (should be set to zero either here or in the optimizer),
#               based on https://www.fast.ai/2018/07/02/adam-weight-decay/
# schedule_on_iter: by default the scheduler does one step per epoch, when schedule_on_iter is true it does one step per iter
# iter_size: specifies after how many dataloader iterations the solver should be run, this basically enables
#               a larger batch size by splitting the batch into several succesive smaller batch and only updating the model
#               after iter_size batches, similar to the same feature in caffe
# gpu_id: id of gpu to use, set to None to use cpu
# mp_rank: rank/id of node in case of distributing training accross several nodes, use 0 if only one node
# mp_nodecount: total number of nodes involved in distributed training
# batch_size is for one gpu
def train_model(model, optimizer, scheduler, dataset, batch_size, shuffle=False,
                snaphot_prefix=None, num_iters=None,
                num_epochs=None, iter_size=1, display_iter=20, snaphost_interval=None,
                load_snapshot_path=None, restore_dataloader=True, display_gpu=False, schedule_on_iter=False,
                gpu_id=None, mp_backend='nccl', mp_init_method="env://", mp_rank=0, mp_nodecount=1, ngpus_per_node=1):
    since = time.time()
    torch.manual_seed(0)
    assert(iter_size > 0)
    assert(round(iter_size) == iter_size)
    if gpu_id is None:
        gpu_id = -1

    # initialize stuff
    model.train(True)  # Set model to training mode

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        if gpu_id >= 0:
            print("Use GPU: {} for training".format(gpu_id))
            torch.cuda.set_device(gpu_id)
            model.cuda(gpu_id)
        else:
            model.cuda()
    else:
        model = model.cpu()

    world_size = mp_nodecount * ngpus_per_node
    can_display_save = not(ngpus_per_node > 1 or mp_nodecount > 1) or (gpu_id == 0)

    nprocs = torch.multiprocessing.cpu_count()
    num_workers = min(max(nprocs // ngpus_per_node, 1), 6)  # no need to have more than 6 workers per gpu
    print_multigpu(can_display_save, "using {} dataloader workers".format(num_workers), flush=True)

    if ngpus_per_node > 1 or mp_nodecount > 1:
        mp_rank = mp_rank * ngpus_per_node + gpu_id
        dist.init_process_group(backend=mp_backend, init_method=mp_init_method,
                                world_size=mp_nodecount * ngpus_per_node, rank=mp_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id])
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=mp_nodecount * ngpus_per_node, rank=mp_rank, shuffle=shuffle)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=sampler)
    else:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    # zero the parameter gradients
    optimizer.zero_grad()
    dataset_size = data_loader.dataset.__len__()
    iterid = 0
    display_since = time.time()
    running_loss_display = 0.0
    running_loss_epoch = 0.0
    avg_running_loss_display = 0.0
    avg_running_loss_epoch = 0.0
    iter_size_counter = 0
    start_epoch = 0
    batch_size = data_loader.batch_size
    drop_last = data_loader.drop_last
    shuffle = isinstance(data_loader.sampler, torch.utils.data.sampler.RandomSampler)
    if drop_last:
        num_dataset_iter = int(np.floor(float(dataset_size) / (batch_size * world_size)))
    else:
        num_dataset_iter = int(np.ceil(float(dataset_size) / (batch_size * world_size)))

    # routine to resume from a snapshot
    if load_snapshot_path is not None:
        checkpoint = torch.load(load_snapshot_path)
        print_multigpu(can_display_save, "resuming training from iteration " + str(checkpoint['iterid']), flush=True)
        # restore inputs
        if num_iters is None:
            num_iters = checkpoint['num_iters']
        # no need to handle num epochs, it will be handled below
        if iter_size is None:
            iter_size = checkpoint['iter_size']
        if display_iter is None:
            display_iter = checkpoint['display_iter']
        if snaphost_interval is None:
            snaphost_interval = checkpoint['snaphost_interval']
        if snaphot_prefix is None:
            snaphot_prefix = checkpoint['snaphot_prefix']
        # restore status
        iterid = checkpoint['iterid']
        start_epoch = (iterid * checkpoint['iter_size']) // num_dataset_iter
        iter_size_counter = checkpoint['iter_size_counter']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        running_loss_display = checkpoint['running_loss_display']
        running_loss_epoch = checkpoint['running_loss_epoch']
        # basic check to verify that the dataset is the same
        # restore dataloader (if it does not have shuffle=True -> check not implemented -> not needed since we have torch.manual_seed(0))
        if checkpoint['dataset_size'] == dataset_size and checkpoint['batch_size'] == batch_size \
                and checkpoint['shuffle'] == shuffle and checkpoint['drop_last'] == drop_last \
                and restore_dataloader is True:
            print_multigpu(can_display_save, "restoring data loader state")
            print_multigpu(can_display_save, "set restore_dataloader=False to disable it", flush=True)
            resume_counter = 0
            max_count = (iterid * checkpoint['iter_size']) % num_dataset_iter
            if resume_counter < max_count:
                for _ in data_loader:
                    resume_counter = resume_counter + 1
                    if resume_counter == max_count:
                        break
        time_elapsed = time.time() - since
        print_multigpu(can_display_save, 'State restore completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), flush=True)
        since = time.time()

    if num_iters is not None and num_epochs is not None:
        raise ValueError("num_iters and num_epochs cannot both have a value")
    elif num_iters is not None:
        num_epochs = int(np.ceil(float(num_iters * iter_size) / num_dataset_iter))
    elif num_epochs is not None:
        num_iters = int(np.ceil(float(num_epochs * num_dataset_iter) / iter_size))
    else:
        raise ValueError("one of num_iters and num_epochs must have a value")
    # convert snapshot interval in iterations if needed
    # and perform snapshot checks
    if snaphot_prefix is not None:
        if not os.path.isdir(os.path.split(snaphot_prefix)[0]):
            os.makedirs(os.path.split(snaphot_prefix)[0])
        if os.path.isfile(snaphot_prefix + '_iter_' + str(num_iters) + '.pth'):
            raise ValueError("Snapshot files already exist, aborting")
        if snaphost_interval is not None:
            if snaphost_interval <= num_epochs:
                snaphost_interval = int(round(snaphost_interval * (num_dataset_iter // iter_size) + 1))
            else:
                snaphost_interval = int(round(snaphost_interval))
            for i in range(iterid // snaphost_interval, num_iters, snaphost_interval):
                if os.path.isfile(snaphot_prefix + '_iter_' + str(i) + '.pth'):
                    raise ValueError("Snapshot files already exist, aborting")
        else:
            print("WARNING: only the final model will be saved, no intermediate snapshot")
    else:
        if snaphost_interval is not None:
            raise ValueError("Missing snapshot prefix, aborting")
        else:
            print_multigpu(can_display_save, "WARNING: no snapshot will be saved", flush=True)

    for epoch in range(start_epoch, num_epochs):
        print_multigpu(can_display_save, 'Epoch {}/{}'.format(epoch + 1, num_epochs))
        print_multigpu(can_display_save, '-' * 10, flush=True)
        if ngpus_per_node > 1 or mp_nodecount > 1:
            sampler.set_epoch(epoch)
        # Iterate over data
        for inputs in data_loader:
            iter_size_counter = iter_size_counter + 1
            if use_gpu:
                inputs = [inputsi.cuda(non_blocking=True) for inputsi in inputs]

            # forward, put the loss in the network model
            # to reduce the number of ops and to avoid splitting the intputs between network and loss inputs
            loss = model(*inputs) / iter_size
            loss.backward()

            # statistics
            running_loss_display = running_loss_display + float(loss.data)
            running_loss_epoch = running_loss_epoch + float(loss.data)

            # backward + optimize only every iter_size
            # update iteration number only every iter_size
            if iter_size_counter == iter_size:  # iter_size_counter starts at 1 as this is after the iteration update
                optimizer.step()
                # zero the parameter gradients
                optimizer.zero_grad()
                # update scheduler
                if schedule_on_iter:
                    scheduler.step()
                iterid = iterid + 1
                iter_size_counter = 0
                # snapshot model
                if snaphost_interval is not None and snaphot_prefix is not None:
                    if iterid % snaphost_interval == 0:
                        if can_display_save:
                            torch.save(model.state_dict(), snaphot_prefix + '_iter_' + str(iterid) + '.pth')
                            torch.save({
                                # input params
                                'num_iters': num_iters,
                                'num_epochs': num_epochs,
                                'iter_size': iter_size,
                                'display_iter': display_iter,
                                'snaphost_interval': snaphost_interval,
                                'snaphot_prefix': snaphot_prefix,
                                'dataset_size': dataset_size,
                                'batch_size': batch_size,
                                'drop_last': drop_last,
                                'shuffle': shuffle,
                                # status
                                'epoch': epoch,
                                'iterid': iterid,
                                'iter_size_counter': iter_size_counter,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'running_loss_display': running_loss_display,
                                'running_loss_epoch': running_loss_epoch
                            }, snaphot_prefix + '_iter_' + str(iterid) + '.train_pth')

                if iterid % display_iter == 0:
                    avg_running_loss_display = running_loss_display / display_iter
                    print_multigpu(can_display_save, 'Epoch {} Iter {} Loss: {:.4f} {} iter in {:.2f}s. Learning rate {}'.format(epoch + 1, iterid,
                                   avg_running_loss_display, display_iter, time.time() - display_since,
                                   ' '.join(list(map(lambda x: '{:.2e}'.format(x), scheduler.get_lr())))), flush=True)
                    if display_gpu:
                        print_multigpu(can_display_save, "used a maximum of {:.3f} GB of GPU memory".format(torch.cuda.max_memory_allocated() / 1000000000), flush=True)
                    running_loss_display = 0.0
                    display_since = time.time()

                if iterid == num_iters:
                    break

        # update scheduler
        if not schedule_on_iter:
            scheduler.step()

        avg_running_loss_epoch = running_loss_epoch / num_dataset_iter * iter_size
        print_multigpu(can_display_save, 'Epoch {} Loss: {:.4f} '.format(epoch + 1, avg_running_loss_epoch), flush=True)
        running_loss_epoch = 0.0

        if iterid == num_iters:
            break
    if snaphot_prefix is not None:
        if can_display_save:
            torch.save(model.state_dict(), snaphot_prefix + '_iter_' + str(iterid) + '.pth')
            torch.save({
                # input params
                'num_iters': num_iters,
                'num_epochs': num_epochs,
                'iter_size': iter_size,
                'display_iter': display_iter,
                'snaphost_interval': snaphost_interval,
                'snaphot_prefix': snaphot_prefix,
                'dataset_size': dataset_size,
                'batch_size': batch_size,
                'drop_last': drop_last,
                'shuffle': shuffle,
                # status
                'epoch': epoch,
                'iterid': iterid,
                'iter_size_counter': iter_size_counter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'running_loss_display': running_loss_display,
                'running_loss_epoch': running_loss_epoch
            }, snaphot_prefix + '_iter_' + str(iterid) + '.train_pth')

    time_elapsed = time.time() - since
    print_multigpu(can_display_save, 'Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, avg_running_loss_epoch, avg_running_loss_display


# wrapper on train_model that has gpu_id as first argument, to be used for spawning
def train_model_spawnee(gpu_id, model, optimizer, scheduler, dataset, batch_size, shuffle=False,
                        snaphot_prefix=None, num_iters=None,
                        num_epochs=None, iter_size=1, display_iter=20, snaphost_interval=None,
                        load_snapshot_path=None, restore_dataloader=True, display_gpu=False, schedule_on_iter=False,
                        mp_backend='nccl', mp_init_method="env://", mp_rank=0, mp_nodecount=1, ngpus_per_node=1):

    train_model(model, optimizer, scheduler, dataset, batch_size, shuffle, snaphot_prefix, num_iters,
                num_epochs, iter_size, display_iter, snaphost_interval,
                load_snapshot_path, restore_dataloader, display_gpu, schedule_on_iter,
                gpu_id, mp_backend, mp_init_method, mp_rank, mp_nodecount, ngpus_per_node)


# batch_size is for one gpu
# drop in replacement for train_model when using multiple gpus, only difference is gpu_id and ngpus_per_node which are removed
def train_model_multigpu(model, optimizer, scheduler, dataset, batch_size, shuffle=False,
                         snaphot_prefix=None, num_iters=None,
                         num_epochs=None, iter_size=1, display_iter=20, snaphost_interval=None,
                         load_snapshot_path=None, restore_dataloader=True, display_gpu=False, schedule_on_iter=False,
                         mp_backend='nccl', mp_init_method="env://", mp_rank=0, mp_nodecount=1):
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 or mp_nodecount > 1:
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(train_model_spawnee, nprocs=ngpus_per_node,
                 args=(model, optimizer, scheduler, dataset, batch_size, shuffle,
                       snaphot_prefix, num_iters,
                       num_epochs, iter_size, display_iter, snaphost_interval,
                       load_snapshot_path, restore_dataloader, display_gpu, schedule_on_iter,
                       mp_backend, mp_init_method, mp_rank, mp_nodecount, ngpus_per_node))
    else:
        # Simply call main_worker function, discard multi gpu arguments
        train_model(model, optimizer, scheduler, dataset, batch_size, shuffle, snaphot_prefix, num_iters,
                    num_epochs, iter_size, display_iter, snaphost_interval,
                    load_snapshot_path, restore_dataloader, display_gpu, schedule_on_iter)


# custom_lr must be a dict binding a parameter name to a learning rate
# custom_lr = {"param1" : 0.01,
#              "param2" : 0.03}
def make_param_groups(model, custom_lr):
    param_groups = []
    default_group = {"params": []}
    lr_list = np.unique([custom_lr[n] for n in custom_lr.keys()])
    for lridx in range(len(lr_list)):
        param_groups.append({"params": [], "lr": lr_list[lridx]})
    if len(list(model.parameters())) == len(list(model.named_parameters())):
        # check that all keys exist in the model, do nothing
        for n in custom_lr.keys():
            found = False
            for m in model.named_parameters():
                if m[0] == n:
                    found = True
                    break
            if not found:
                raise ValueError("Parameter " + n + " could not be found in the model.  Cannot make parameter groups")
        # create parameter list
        for m in model.named_parameters():
            found = False
            for n in custom_lr.keys():
                if m[0] == n:
                    param_groups[int(np.nonzero(lr_list == custom_lr[n])[0][0])]["params"].append(m[1])
                    found = True
                    break
            if not found:
                default_group["params"].append(m[1])
        param_groups.append(default_group)
        return param_groups
    else:
        raise ValueError("Some parameters are not named. Cannot make parameter groups")


# training function to find the best learning rate
# the learning rate will be increased exponentially at every run
# the loss will be displayed at every run and at the end
# it is up to the user to collect and analyze it from the console output
# all runs are started from the same state except for the dataloader
def find_lr(model, optimizer, dataset, batch_size, shuffle=False, iter_size=1, load_snapshot_path=None, restore_dataloader=True,
            min_lr=1e-10, max_lr=1, iter_per_run=50, num_runs=30, result_file="results.npz"):
    print("Start training to find best learning rate. You should make sure that your optimizer has a learning rate equal to min_lr")
    print("The value of min_lr is {:.2e}".format(min_lr))
    loss_list = np.zeros(num_runs)
    assert(max_lr > min_lr > 0)
    # would be nice not to need to save to disk
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, result_file)
    for i in range(num_runs):
        checkpoint = torch.load(result_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda x: (max_lr / min_lr)**(i / (num_runs - 1)))
        _, _, loss_list[i] = train_model(model=model, optimizer=optimizer, scheduler=scheduler,
                                         dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                         num_iters=iter_per_run,
                                         iter_size=iter_size, display_iter=iter_per_run,
                                         load_snapshot_path=load_snapshot_path,
                                         restore_dataloader=restore_dataloader, schedule_on_iter=True)
    print("lr\tloss")
    for i in range(num_runs):
        print("{:.2e}\t{:.4f}".format(min_lr * (max_lr / min_lr)**(i / (num_runs - 1)), loss_list[i]))
    np.savez(result_file, loss_list=loss_list, lr_list=[min_lr * (max_lr / min_lr)**(i / (num_runs - 1)) for i in range(num_runs)])


# training function to find the best learning rate
# the learning rate will be increased exponentially at every run
# the loss will be displayed at every run and at the end
# it is up to the user to collect and analyze it from the console output
# all runs are started from the same state except for the dataloader
# keep result_file to enable drop in replacement of find_lr
# batch_size is for one gpu
# will only use one node
def find_lr_multigpu(model, optimizer, dataset, batch_size, shuffle=False, iter_size=1, load_snapshot_path=None, restore_dataloader=True,
                     min_lr=1e-10, max_lr=1, iter_per_run=50, num_runs=30, result_file="results.npz"):
    print("Start training to find best learning rate. You should make sure that your optimizer has a learning rate equal to min_lr")
    print("The value of min_lr is {:.2e}".format(min_lr))
    assert(max_lr > min_lr > 0)
    # fix this!!!! would be nice not to need to save to disk
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, result_file)
    for i in range(num_runs):
        checkpoint = torch.load(result_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler = lr_scheduler.LambdaLR(optimizer, make_scheduler(i, min_lr, max_lr, num_runs))
        # make a constant scheduler
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[0, iter_per_run + 2], gamma=(max_lr / min_lr)**(i / (num_runs - 1)))
        train_model_multigpu(model=model, optimizer=optimizer, scheduler=scheduler,
                             dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                             num_iters=iter_per_run,
                             iter_size=iter_size, display_iter=iter_per_run,
                             load_snapshot_path=load_snapshot_path,
                             restore_dataloader=restore_dataloader, schedule_on_iter=True)
