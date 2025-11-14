import os
import os.path as osp
from glob import glob
from torch.utils.data import Dataset
import numpy as np
import torch
from functools import partial
import torch.nn.functional as F

################# CODE ADD ################
def mock_first_and_last_frames_context(slices_list, context_mock_strategy):
    "The goal of this method is to Mock a context for the 1st and last frame"

    if context_mock_strategy == 'copy_frame':
      slices_list.insert(0, slices_list[0])
      # DO NOT USE -1 in the insertion index. It works differently than [-1]
      slices_list.insert(len(slices_list), slices_list[-1]) 
    elif context_mock_strategy == 'copy_neighbor':
      slices_list.insert(0, slices_list[1])
      # DO NOT USE -1 in the insertion index. It works differently than [-1]
      slices_list.insert(len(slices_list), slices_list[-2])
    else:
      print("No Context mock strategy was chosen for 1st and last frame! They will be ignored during denoise!")
      return slices_list

    # DEBUG prints
    print("path [0]: ", slices_list[0])
    print("path [1]: ",slices_list[1])
    print("path [2]: ",slices_list[2])
    print("path [3]: ",slices_list[3])
    print("path [4]: ",slices_list[4])
    print("path [-5]: ",slices_list[-5])
    print("path [-4]: ",slices_list[-4])
    print("path [-3]: ",slices_list[-3])
    print("path [-2]: ",slices_list[-2])
    print("path [-1]: ",slices_list[-1])
    
    return slices_list
################# CODE ADD ################


class CTDataset(Dataset):
    def __init__(self, dataset, mode, test_id=9, dose=5, context=True, context_mock_strategy_for_1st_and_last_frames=None):
        self.mode = mode
        self.context = context
        
        ################# CODE ADD ################
        # The parameter was also ADD to __init__() params
        self.context_mock_strategy_for_1st_and_last_frames = context_mock_strategy_for_1st_and_last_frames
        self.dataset = dataset
        ################# CODE ADD ################
        print(dataset)

        if dataset in ['mayo_2016_sim', 'mayo_2016']:
            if dataset == 'mayo_2016_sim':
                data_root = './data_preprocess/gen_data/mayo_2016_sim_npy'
            elif dataset == 'mayo_2016':
                data_root = './data_preprocess/gen_data/mayo_2016_npy'
                
            patient_ids = [67, 96, 109, 143, 192, 286, 291, 310, 333, 506]
            if mode == 'train':
                patient_ids.pop(test_id)
            elif mode == 'test':
                patient_ids = patient_ids[test_id:test_id + 1]

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('L{:03d}_target_'.format(id) + '*_img.npy'))))

                ################# CODE ADD ################
                if context:
                    patient_list = mock_first_and_last_frames_context(patient_list, context_mock_strategy_for_1st_and_last_frames)
                ################# CODE ADD ################
                
                patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            base_target = patient_lists

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('L{:03d}_{}_'.format(id, dose) + '*_img.npy'))))
                if context:
                    ################# CODE ADD ################
                    patient_list = mock_first_and_last_frames_context(patient_list, context_mock_strategy_for_1st_and_last_frames)
                    ################# CODE ADD ################

                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
            base_input = patient_lists

        elif dataset == 'mayo_2020':
            data_root = './data_preprocess/gen_data/mayo_2020_npy'
            if dose == 10:
                patient_ids = ['C052', 'C232', 'C016', 'C120', 'C050']
            elif dose == 25:
                patient_ids = ['L077', 'L056', 'L186', 'L006', 'L148']

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, (id + '_target_' + '*_img.npy'))))
                patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            base_target = patient_lists

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, (id + '_{}_'.format(dose) + '*_img.npy'))))
                if context:
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
                base_input = patient_lists


        elif dataset == 'piglet':
            data_root = './data_preprocess/gen_data/piglet_npy'

            patient_list = sorted(glob(osp.join(data_root, 'piglet_target_' + '*_img.npy')))
            base_target = patient_list[1:len(patient_list) - 1]

            patient_list = sorted(glob(osp.join(data_root, 'piglet_{}_'.format(dose) + '*_img.npy')))
            if context:
                cat_patient_list = []
                for i in range(1, len(patient_list) - 1):
                    patient_path = ''
                    for j in range(-1, 2):
                        patient_path = patient_path + '~' + patient_list[i + j]
                    cat_patient_list.append(patient_path)
                    base_input = cat_patient_list
            else:
                patient_list = patient_list[1:len(patient_list) - 1]
                base_input = patient_list


        elif dataset == 'phantom':
            data_root = './data_preprocess/gen_data/xnat_npy'

            patient_list = sorted(glob(osp.join(data_root, 'xnat_target' + '*_img.npy')))[9:21]
            base_target = patient_list[1:len(patient_list) - 1]

            patient_list = sorted(glob(osp.join(data_root, 'xnat_{:0>3d}_'.format(dose) + '*_img.npy')))[9:21]
            if context:
                cat_patient_list = []
                for i in range(1, len(patient_list) - 1):
                    patient_path = ''
                    for j in range(-1, 2):
                        patient_path = patient_path + '~' + patient_list[i + j]
                    cat_patient_list.append(patient_path)
                    base_input = cat_patient_list
            else:
                patient_list = patient_list[1:len(patient_list) - 1]
                base_input = patient_list

        ################# CODE ADD ################

        if dataset == '2detect':
            data_root = '../2DeteCT'
            
            # Range of indexes related each subsample from 2DETECT
            # MIX_2 was currently chosen for train and all other for tests
            sample_mapping = {
                'MIX_1': range(1, 1800 + 1),
                'MIX_2': range(1801, 3720 + 1),
                'MIX_3': range(3721, 5000 + 1),
                'FIG_OOD_PURE': range(5521, 5570 + 1),
                'ALMOND_OOD_PURE': range(5571, 5620 + 1),
                'BANANA_OOD_PURE': range(5621, 5670 + 1),
                'RAISIN_OOD_PURE': range(5671, 5720 + 1),
                'WALNUT_OOD_PURE': range(5721, 5770 + 1),
                'COFFEE_BEANS_OOD_PURE': range(5771, 5820 + 1),
                'LAVA_STONE_OOD_PURE': range(5821, 5870 + 1),
                'MIX_3_OOD_NOISE': range(5871, 5920 + 1),
                'TITANIUM_PROSTHESES_SCREWS_OOD_MIX_3': range(5971, 6070 + 1),
                'PEANUT_OOD_MIX_3': range(6121, 6170 + 1),
                'PISTACHIO_OOD_MIX_3': range(6171, 6220 + 1),
                'HAZELNUT_OOD_MIX_3': range(6221, 6270 + 1),
                'GRAPE_OOD_MIX_3': range(6271, 6320 + 1),
                'FLESH_FIG_OOD_MIX_3': range(6321, 6370 + 1),
            }

            train_samples = ['MIX_2']

            if mode == 'train':
                sample_ids = train_samples
            elif mode == 'test':
                sample_ids = [sample_name for sample_name in sample_mapping.keys() if sample_name not in [train_samples]]

            samples_slices_paths_lists = []
            for sample_id in sample_ids:
                sample_slices_paths = []
                for slice_idx in sample_mapping[sample_id]:
                    slice_dir_name = "slice" + str(slice_idx).zfill(5)
                    slice_path = os.path.join(data_root, slice_dir_name, "mode2", "reconstruction.tif")
                    sample_slices_paths.append(slice_path)

                if context:
                    sample_slices_paths = mock_first_and_last_frames_context(sample_slices_paths, context_mock_strategy_for_1st_and_last_frames)
                    
                    samples_slices_paths_lists = samples_slices_paths_lists + sample_slices_paths[1:len(sample_slices_paths) - 1]
            base_target = samples_slices_paths_lists

            samples_slices_paths_lists = []
            for sample_id in sample_ids:
                sample_slices_paths = []
                for slice_idx in sample_mapping[sample_id]:
                    slice_dir_name = "slice" + str(slice_idx).zfill(5)
                    slice_path = os.path.join(data_root, slice_dir_name, "mode1", "reconstruction.tif")
                    sample_slices_paths.append(slice_path)

                if context:

                    sample_slices_paths = mock_first_and_last_frames_context(sample_slices_paths, context_mock_strategy_for_1st_and_last_frames)

                    cat_sample_paths_list = []
                    for i in range(1, len(sample_slices_paths) - 1):
                        path = ''
                        for j in range(-1, 2):
                            path = path + '~' + sample_slices_paths[i + j]
                        cat_sample_paths_list.append(path)
                    samples_slices_paths_lists = samples_slices_paths_lists + cat_sample_paths_list
                else:
                    sample_slices_paths = sample_slices_paths[1:len(sample_slices_paths) - 1]
                    samples_slices_paths_lists = samples_slices_paths_lists + sample_slices_paths

            base_input = samples_slices_paths_lists

        ################# CODE ADD ################

        self.input = base_input
        self.target = base_target
        print(len(self.input))
        print(len(self.target))


    def __getitem__(self, index):
        input, target = self.input[index], self.target[index]
        ################# CODE ADD ################
        if self.dataset == "2detect":
            import imageio
            data_load_method = imageio.imread
            translation=0
            MIN_B = 0
            MAX_B = 1000
        else:
            data_load_method = np.load
            # This parameters just replicate original parameters set by author to normalize_ method
            translation=-1024
            MIN_B=-1024
            MAX_B=3072
            
        ################# CODE ADD ################

        ################# CODE CHANGED ################
        if self.context:
            input = input.split('~')
            inputs = []
            for i in range(1, len(input)):
                inputs.append(data_load_method(input[i])[np.newaxis, ...].astype(np.float32))
            input = np.concatenate(inputs, axis=0)  #(3, 512, 512)
        else:
            input = data_load_method(input)[np.newaxis, ...].astype(np.float32) #(1, 512, 512)
        target = data_load_method(target)[np.newaxis,...].astype(np.float32) #(1, 512, 512)
        
        input = self.normalize_(input, translation, MIN_B, MAX_B)
        target = self.normalize_(target, translation, MIN_B, MAX_B)
        ################ CODE CHANGED ################

        return input, target

    def __len__(self):
        return len(self.target)

    def normalize_(self, img, translation, MIN_B, MAX_B):
        img = img + translation
        img[img < MIN_B] = MIN_B
        img[img > MAX_B] = MAX_B
        img = (img - MIN_B) / (MAX_B - MIN_B)
        return img


dataset_dict = {
    'train': partial(CTDataset, dataset='2detect', mode='train', test_id=None, dose=None, context=True),
    'mayo_2016_sim': partial(CTDataset, dataset='mayo_2016_sim', mode='test', test_id=9, dose=5, context=True),
    'mayo_2016': partial(CTDataset, dataset='mayo_2016', mode='test', test_id=9, dose=25, context=True),
    'mayo_2020': partial(CTDataset, dataset='mayo_2020', mode='test', test_id=None, dose=None, context=True),
    'piglet': partial(CTDataset, dataset='piglet', mode='test', test_id=None, dose=None, context=True),
    'phantom': partial(CTDataset, dataset='phantom', mode='test', test_id=None, dose=108, context=True),
    ################ CODE ADD ################
    '2detect': partial(CTDataset, dataset='2detect', mode='train', test_id=None, dose=None, context=True)  
    ################ CODE ADD ################
}
