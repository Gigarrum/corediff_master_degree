import torch
import os.path as osp
import os
import time
from torchvision.utils import save_image
import torch.distributed as dist
import inspect
from utils.ops import reduce_tensor, load_network
import csv


def get_varname(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


class LoggerX(object):

    def __init__(self, save_root):
        self.models_save_dir = osp.join(save_root, 'save_models')
        self.images_save_dir = osp.join(save_root, 'save_images')
        self.log_save_dir = osp.join(save_root, 'save_logs')
        os.makedirs(self.models_save_dir, exist_ok=True)
        os.makedirs(self.images_save_dir, exist_ok=True)
        self._modules = []
        self._module_names = []
        self.world_size = 1
        self.local_rank = 0

        ####### CODE ADD #######
        # Create CSV files to store training and testing logs
        os.makedirs(self.log_save_dir, exist_ok=True)
        self._initialize_csv(self.train_log_file, ['epoch', 'train_loss'])
        self._initialize_csv(self.test_log_file, ['epoch', 'psnr', 'ssim'])
        ####### CODE ADD #######

    @property
    def modules(self):
        return self._modules

    @property
    def module_names(self):
        return self._module_names

    @modules.setter
    def modules(self, modules):
        for i in range(len(modules)):
            self._modules.append(modules[i])
            self._module_names.append(get_varname(modules[i]))

    ####### CODE ADD #######
    def _initialize_csv(self, file_path, headers):
        if not osp.exists(file_path):
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)

    def log_train_loss(self, epoch, train_loss):
        if self.local_rank != 0:
            return
        with open(self.train_log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss])

    def log_test_metrics(self, epoch, psnr, ssim):
        if self.local_rank != 0:
            return
        with open(self.test_log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, psnr, ssim])
####### CODE ADD #######

    def checkpoints(self, epoch):
        if self.local_rank != 0:
            return
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            torch.save(module.state_dict(), osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch)))

    def load_checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            module.load_state_dict(load_network(osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch))))

    def load_test_checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            if module_name == 'ema_model':
                module = self.modules[i]
                module.load_state_dict(load_network(osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch))))

    def msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)

        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                var = reduce_tensor(var)
                var = var.item()
            output_str += '{} {:2.5f}, '.format(var_name, var)

        if self.local_rank == 0:
            print(output_str)

    def save_image(self, grid_img, n_iter, sample_type):
        save_image(grid_img, osp.join(self.images_save_dir,
                                      '{}_{}_{}.png'.format(n_iter, self.local_rank, sample_type)),
                   nrow=1)
        
    
