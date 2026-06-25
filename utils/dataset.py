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

def crop_pair(img1, img2, crop_size, crop_strategy, random_generator=None):
    """
    Apply the same crop to two images.

    Args:
        img1, img2: numpy arrays with shape (C, H, W)
        crop_size: int
        random: bool

    Returns:
        cropped_img1, cropped_img2
    """
    assert img1.shape[-2:] == img2.shape[-2:], "Images must have same spatial size"

    _, H, W = img1.shape
    if H == crop_size and W == crop_size:
        return img1, img2
    if crop_strategy == 'random':
        top = random_generator.integers(0, H - crop_size + 1)
        left = random_generator.integers(0, W - crop_size + 1)
    elif crop_strategy == 'center':
        top = (H - crop_size) // 2
        left = (W - crop_size) // 2
    else:
        raise Exception(f"Crop strategy {crop_strategy} not supported!")
    return img1[:, top:top + crop_size, left:left + crop_size], img2[:, top:top + crop_size, left:left + crop_size]


class CTDataset(Dataset):
    def __init__(self, dataset, mode, test_id=9, dose=5, context=True, crop_strategy=None, context_mock_strategy_for_1st_and_last_frames='copy_neighbor', normalization_strategy='mean_std', rng_seed=42):
        self.mode = mode
        self.context = context
        # Create a single generator withy fixed seed so it allow complete reproducibility when extracting random. This can't be done every time the random_crop is called
        # otherwise it will always reset the random generator and apply the same crop 
        self.random_generator = np.random.default_rng(rng_seed)
        
        ################# CODE ADD ################
        # The parameter was also ADD to __init__() params
        self.context_mock_strategy_for_1st_and_last_frames = context_mock_strategy_for_1st_and_last_frames
        self.crop_strategy = crop_strategy
        self.dataset = dataset
        self.normalization_strategy = normalization_strategy
        print(crop_strategy, normalization_strategy, context_mock_strategy_for_1st_and_last_frames)
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
            data_root = '/ibira/lnls/labs/tepui/home/paulo.mausbach/master_degree_storage/data/2DeteCT'
            
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

            sample_ndct_mean_std_mapping = {
                'MIX_1': {
                    'mean': 0.0006538797169923782,
                    'std': 0.0009740228415466845
                    },
                'MIX_2': {
                    'mean': 0.0006392895593307912,
                    'std': 0.0009253055322915316
                    },
                'MIX_3': {
                    'mean': 0.0006852527731098235,
                    'std': 0.0009844953892752528
                    },
                'FIG_OOD_PURE': {
                    'mean': 0.0006105180364102125,
                    'std': 0.0007530362927354872
                    },
                'ALMOND_OOD_PURE': {
                    'mean': 0.0004920786595903337,
                    'std': 0.000580836262088269
                    },
                'BANANA_OOD_PURE': {
                    'mean': 0.000435380672570318,
                    'std': 0.0005090352497063577
                    },
                'RAISIN_OOD_PURE': {
                    'mean': 0.00045652143307961524,
                    'std': 0.0006079224986024201
                    },
                'WALNUT_OOD_PURE': {
                    'mean': 0.0004319391446188092,
                    'std': 0.0005231823888607323
                    },
                'COFFEE_BEANS_OOD_PURE': {
                    'mean': 0.00038193189539015293,
                    'std': 0.0003958650049753487
                    },
                'LAVA_STONE_OOD_PURE': {
                    'mean': 0.0006710302550345659,
                    'std': 0.0013717758702114224
                    },
                'MIX_3_OOD_NOISE': {
                    'mean': 0.0007378465961664915,
                    'std': 0.0012105517089366913
                    },
                'TITANIUM_PROSTHESES_SCREWS_OOD_MIX_3': {
                    'mean': 0.0007924546371214092,
                    'std': 0.001340771559625864
                    },
                'PEANUT_OOD_MIX_3': {
                    'mean': 0.0007109223515726626,
                    'std': 0.00100797472987324
                    },
                'PISTACHIO_OOD_MIX_3': {
                    'mean': 0.000727855833247304,
                    'std': 0.0009698904468677938
                    },
                'HAZELNUT_OOD_MIX_3': {
                    'mean': 0.0007273655501194298,
                    'std': 0.0009698904468677938
                    },
                'GRAPE_OOD_MIX_3': {
                    'mean': 0.0007468361291103065,
                    'std': 0.0009705426055006683
                    },
                'FLESH_FIG_OOD_MIX_3': {
                    'mean': 0.0006862918962724507,
                    'std': 0.000911360839381814
                    }
            }
            
            sample_ldct_mean_std_mapping = {
                'MIX_1': {
                    'mean': 0.0007928369450382888,
                    'std': 0.0014398089842870831
                    },
                'MIX_2': {
                    'mean': 0.0007173863705247641,
                    'std': 0.0012580337934195995
                    },
                'MIX_3': {
                    'mean': 0.0007028057589195669,
                    'std': 0.0011917755473405123
                    },
                'FIG_OOD_PURE': {
                    'mean': 0.0007410991238430142,
                    'std': 0.0010316030820831656
                    },
                'ALMOND_OOD_PURE': {
                    'mean': 0.000590855663176626,
                    'std': 0.00078888691496104
                    },
                'BANANA_OOD_PURE': {
                    'mean': 0.0005239111487753689,
                    'std': 0.0007047480321489275
                    },
                'RAISIN_OOD_PURE': {
                    'mean': 0.0005526128807105124,
                    'std': 0.0008370787836611271
                    },
                'WALNUT_OOD_PURE': {
                    'mean': 0.0005231364048086107,
                    'std': 0.0007290175999514759
                    },
                'COFFEE_BEANS_OOD_PURE': {
                    'mean': 0.0004596579528879374,
                    'std': 0.0005764670786447823
                    },
                'LAVA_STONE_OOD_PURE': {
                    'mean': 0.0007996205822564662,
                    'std': 0.001938883913680911
                    },
                'MIX_3_OOD_NOISE': {
                    'mean': 0.0007307335617952049,
                    'std': 0.001552431844174862
                    },
                'TITANIUM_PROSTHESES_SCREWS_OOD_MIX_3': {
                    'mean': 0.0008188736974261701,
                    'std': 0.0015750402817502618
                    },
                'PEANUT_OOD_MIX_3': {
                    'mean': 0.0007324786274693906,
                    'std': 0.001219474826939404
                    },
                'PISTACHIO_OOD_MIX_3': {
                    'mean': 0.0007548062712885439,
                    'std': 0.0012305786367505789
                    },
                'HAZELNUT_OOD_MIX_3': {
                    'mean': 0.0007491590222343802,
                    'std': 0.0011984084267169237
                    },
                'GRAPE_OOD_MIX_3': {
                    'mean': 0.0007696138345636427,
                    'std': 0.0012076576240360737
                    },
                'FLESH_FIG_OOD_MIX_3': {
                    'mean': 0.0007145455456338823,
                    'std': 0.001149201299995184
                    }
            }
            

            train_samples = ['MIX_2']
            val_samples = ['MIX_1']

            # WARNING!! IF USING MORE THAN 1 SUBSAMPLE AS TRAIN SET, THIS VALUES MUST BE ADJUSTED. AVERAGING THE MEAN AND STD
            # FROM BOTH OF THEM IS NOT CORRECT!
            self.standardization_mean = sample_ldct_mean_std_mapping['MIX_2']['mean']
            self.standardization_std = sample_ldct_mean_std_mapping['MIX_2']['std']

            if mode == 'train':
                sample_ids = train_samples
            elif mode == 'test':
                sample_ids = val_samples
            
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
        else:
            data_load_method = np.load

            
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
        
        print("\n==================\n",input.shape, target.shape,"\n==================\n")
        ################ CODE CHANGED ################

        ################# CODE ADD ################
        if self.dataset == "2detect": 
            if self.normalization_strategy == "min_max":
                translation = 0
                MIN_INPUT_VAL = input.min()
                MAX_INPUT_VAL = input.max()

                # Apply min/max normalization using slice local min/max
                # The same scale must be applied to both images to guarante both standardized in the
                # same scale
                input = self.normalize_(input, translation, MIN_INPUT_VAL, MAX_INPUT_VAL)
                target = self.normalize_(target, translation, MIN_INPUT_VAL, MAX_INPUT_VAL)
            
            elif self.normalization_strategy == "mean_std":
                # Apply mean/std normalization using subsample mean/std values
                input = (input -  self.standardization_mean) / self.standardization_std
                target = (target -  self.standardization_mean) / self.standardization_std
            else:
                raise Exception("Normalization strategy invalid!")
        else:
            # This parameters just replicate original parameters set by author to normalize_ method
            translation = -1024
            MIN_B_INPUT = -1024
            MAX_B_INPUT = 3072
            MIN_B_TARGET = -1024
            MAX_B_TARGET = 3072

            # Apply min/max normalization using HU window used on CoreDiff original paper
            input = self.normalize_(input, translation, MIN_B_INPUT, MAX_B_INPUT)
            target = self.normalize_(target, translation, MIN_B_TARGET, MAX_B_TARGET)

        # Check if image need to be croped
        #if self.mode == 'train' or self.mode == 'train_osl_framework':
        #    self.crop_strategy = 'random'
        #else:
        #    self.crop_strategy = 'center'
        if self.crop_strategy is not None:
            input, target = crop_pair(input, target, crop_size=512, crop_strategy=self.crop_strategy, random_generator=self.random_generator)
        ################# CODE ADD ################

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
    #'train': partial(CTDataset, dataset='2detect', mode='train', test_id=None, dose=None, context=True, crop_strategy="center", context_mock_strategy_for_1st_and_last_frames="copy_neighbor"), # THIS IS THE DATASET USED FOR TRAINING, NO MATTER THE PARAM PASSED!!!!
    'train': partial(CTDataset, dataset='2detect', mode='train', test_id=None, dose=None, context=True),
    'mayo_2016_sim': partial(CTDataset, dataset='mayo_2016_sim', mode='test', test_id=9, dose=5, context=True),
    'mayo_2016': partial(CTDataset, dataset='mayo_2016', mode='test', test_id=9, dose=25, context=True),
    'mayo_2020': partial(CTDataset, dataset='mayo_2020', mode='test', test_id=None, dose=None, context=True),
    'piglet': partial(CTDataset, dataset='piglet', mode='test', test_id=None, dose=None, context=True),
    'phantom': partial(CTDataset, dataset='phantom', mode='test', test_id=None, dose=108, context=True),
    ################ CODE ADD ################
    #'2detect': partial(CTDataset, dataset='2detect', mode='test', test_id=None, dose=None, context=True, crop_strategy="center", context_mock_strategy_for_1st_and_last_frames="copy_neighbor")
    '2detect': partial(CTDataset, dataset='2detect', mode='test', test_id=None, dose=None, context=True)
    ################ CODE ADD ################
}
