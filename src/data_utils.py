import matplotlib.pyplot as plt
import numpy as np

# import os
# datasets = filter(lambda x: os.path.isdir(x), os.listdir('.'))
# missing data sets ['Industrial Multivariate']

all_datasets = ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Car',
                'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z',
                'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                'ECG200', 'ECG5000', 'ECGFiveDays', 'Earthquakes', 'ElectricDevices', 'FISH', 'FaceAll', 'FaceFour',
                'FacesUCR', 'FordA', 'FordB', 'Gun_Point', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'InlineSkate',
                'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT',
                'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
                'MiddlePhalanxTW', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2',
                'OSULeaf', 'OliveOil', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
                'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType',
                'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII',
                'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols', 'ToeSegmentation1', 'ToeSegmentation2',
                'Trace', 'TwoLeadECG', 'Two_Patterns', 'UWaveGestureLibraryAll', 'Wine', 'WordsSynonyms', 'Worms',
                'WormsTwoClass', 'synthetic_control', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
                'uWaveGestureLibrary_Z', 'wafer', 'yoga']

timenet_train_datasets = ['ArrowHead', 'ItalyPowerDemand', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'TwoLeadECG',
                          'FacesUCR', 'Plane', 'Gun_Point', 'WordsSynonyms', 'ToeSegmentation1', 'ToeSegmentation2',
                          'Lighting7', 'DiatomSizeReduction', 'OSULeaf', 'Ham', 'FISH', 'ShapeletSim', 'ShapesAll']

timenet_val_datasets = ['MoteStrain', 'CBF', 'Trace', 'Symbols', 'Herring', 'Earthquakes']

train_datasets = ['50words', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Car', 'CinC_ECG_torso', 'Coffee',
                  'Computers', 'DiatomSizeReduction', 'ECG200', 'Earthquakes', 'FISH', 'FaceAll', 'FaceFour',
                  'FacesUCR', 'Gun_Point', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'InlineSkate',
                  'InsectWingbeatSound',
                  'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT', 'Meat',
                  'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'OSULeaf', 'OliveOil',
                  'Phoneme',
                  'Plane', 'RefrigerationDevices', 'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances',
                  'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves',
                  'Symbols', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'UWaveGestureLibraryAll',
                  'Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass']

test_datasets = ['Adiac', 'ChlorineConcentration', 'Cricket_X', 'Cricket_Y', 'Cricket_Z',
                 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'ECG5000',
                 'ECGFiveDays', 'ElectricDevices', 'FordA', 'FordB', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup',
                 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'PhalangesOutlinesCorrect',
                 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'Strawberry',
                 'SwedishLeaf', 'Two_Patterns', 'wafer', 'yoga', 'synthetic_control', 'uWaveGestureLibrary_X',
                 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z']

my_train_datasets = ['DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                     'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
                     'PhalangesOutlinesCorrect']
my_test_datasets = ['ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW']

my_dataset = ['ChlorineConcentration', 'Cricket_X']

Timenet_Results = {'Adiac': 0.322, 'ChlorineConcentration': 0.269, 'Cricket_X': 0.300, 'Cricket_Y': 0.338,
                   'Cricket_Z': 0.308, 'DistalPhalanxOutlineAgeGroup': 0.223, 'DistalPhalanxOutlineCorrect': 0.188,
                   'DistalPhalanxTW': 0.208, 'ECG5000': 0.069, 'ECGFiveDays': 0.074, 'ElectricDevices': 0.267,
                   'FordA': 0.219, 'FordB': 0.263, 'MedicalImages': 0.250, 'MiddlePhalanxOutlineAgeGroup': 0.210,
                   'MiddlePhalanxOutlineCorrect': 0.270, 'MiddlePhalanxTW': 0.363, 'PhalangesOutlinesCorrect': 0.207,
                   'ProximalPhalanxOutlineAgeGroup': 0.146, 'ProximalPhalanxOutlineCorrect': 0.175,
                   'ProximalPhalanxTW': 0.195, 'Strawberry': 0.062, 'SwedishLeaf': 0.102, 'Two_Patterns': 0.0,
                   'wafer': 0.005, 'yoga': 0.160, 'synthetic_control': 0.013, 'uWaveGestureLibrary_X': 0.214,
                   'uWaveGestureLibrary_Y': 0.311, 'uWaveGestureLibrary_Z': 0.281}

Best_Results = {'Adiac': 0.322, 'ChlorineConcentration': 0.269, 'Cricket_X': 0.236, 'Cricket_Y': 0.197,
                'Cricket_Z': 0.180, 'DistalPhalanxOutlineAgeGroup': 0.160, 'DistalPhalanxOutlineCorrect': 0.187,
                'DistalPhalanxTW': 0.208, 'ECG5000': 0.066, 'ECGFiveDays': 0.063, 'ElectricDevices': 0.267,
                'FordA': 0.219, 'FordB': 0.263, 'MedicalImages': 0.247, 'MiddlePhalanxOutlineAgeGroup': 0.210,
                'MiddlePhalanxOutlineCorrect': 0.270, 'MiddlePhalanxTW': 0.363, 'PhalangesOutlinesCorrect': 0.207,
                'ProximalPhalanxOutlineAgeGroup': 0.137, 'ProximalPhalanxOutlineCorrect': 0.175,
                'ProximalPhalanxTW': 0.188, 'Strawberry': 0.062, 'SwedishLeaf': 0.099, 'Two_Patterns': 0.0,
                'wafer': 0.005, 'yoga': 0.155, 'synthetic_control': 0.013, 'uWaveGestureLibrary_X': 0.211,
                'uWaveGestureLibrary_Y': 0.291, 'uWaveGestureLibrary_Z': 0.280}


class MultipleDatasets(object):
    def __init__(self, directory, datasets=[], batch_size=1, merge_train_test=False, length_constraint=512, data_length=0, val_percentage=0):
        self.data = {}
        self.directory = directory
        self.datasets = datasets
        self.batch_size = batch_size
        self.merge_train_test = merge_train_test

        self.length_constraint = length_constraint

        self.force_length = False
        self.desired_length = data_length
        if self.desired_length > 0:
            self.force_length = True

        self.total_length = 35973

        self.need_validation_data = False
        self.val_percentage = val_percentage
        if self.val_percentage > 0:
            self.need_validation_data = True

    def load_data(self, verbose=False):
        """Input:
        dir: location of the UCR archive
        ratio: ratio to split training and testset
        dataset: name of the dataset in the UCR archive"""
        data = {}
        remove_datasets = []
        for dataset_name in self.datasets:
            datadir = self.directory + '/' + dataset_name + '/' + dataset_name
            train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
            test = np.loadtxt(datadir + '_TEST', delimiter=',')

            if train.shape[1] >= self.length_constraint:
                remove_datasets.append(dataset_name)
                continue

            dataset_data = {}
            dataset_data['X_train'] = train[:, 1:]
            dataset_data['X_test'] = test[:, 1:]

            dataset_data['Y_train'] = train[:, 0]
            dataset_data['Y_test'] = test[:, 0]

            mean_var_data = np.vstack((dataset_data['X_train'], dataset_data['X_test']))
            mean_data = np.mean(mean_var_data, axis=0)
            var_data = np.std(mean_var_data, axis=0)

            dataset_data['X_train'] -= mean_data
            dataset_data['X_test'] -= mean_data

            dataset_data['X_train'] /= var_data
            dataset_data['X_test'] /= var_data

            data[dataset_name] = dataset_data
            if verbose:
                print dataset_name, ', N = ', train.shape[0], ', L = ', train.shape[1], ', min_label = ', np.min(
                    train[:, 0]), ', max_label = ', np.max(train[:, 0])

        for x in remove_datasets:   #remove incompatible datasets
            self.datasets.remove(x)

        self.data = data

        if self.force_length:
            self.make_same_size(verbose)

        if self.merge_train_test:
            self.combine_train_test(verbose)

        if self.need_validation_data:
            self.make_validation_set(verbose)

    def make_validation_set(self, verbose):
        for dataset_name in self.datasets:
            dataset = self.data[dataset_name]
            num_train = dataset['X_train'].shape[0]
            mask = np.random.choice([0, 1], size=num_train, p=[self.val_percentage , 1-self.val_percentage])
            dataset['X_val'] = dataset['X_train'][mask == 0]
            dataset['X_train'] = dataset['X_train'][mask == 1]

            dataset['Y_val'] = dataset['Y_train'][mask == 0]
            dataset['Y_train'] = dataset['Y_train'][mask == 1]

            assert dataset['X_val'].shape[0] + dataset['X_train'].shape[0] == num_train, \
                "error in splitting validation set"

    def combine_train_test(self, verbose):
        """
        Used to train auto-encoder and those datasets are used only for training purposes,
        hence collating train and test data.
        :return:
        """
        for dataset_name in self.data:
            dataset = self.data[dataset_name]
            dataset['X_train'] = np.vstack((dataset['X_train'], dataset['X_test']))
            dataset['Y_train'] = np.concatenate((dataset['Y_train'], dataset['Y_test']), axis=0)
            dataset['X_test'] = {}
            dataset['Y_test'] = {}

    def make_same_size(self, verbose=False):
        for dataset_name in self.data:
            dataset = self.data[dataset_name]
            for key in ['X_train', 'X_test']:
                curr_len = dataset[key].shape[1]
                rep = self.desired_length / curr_len + 1
                dataset[key] = np.tile(dataset[key], (1, rep))
                dataset[key] = np.delete(dataset[key], range(self.desired_length, dataset[key].shape[1]), axis=1)

                assert dataset[key].shape[1] == self.desired_length, 'error in make_same_size'
                if verbose:
                    print dataset[key].shape

    def fix_batch_len(self, data, divby_len, zeros=False, verbose=False):

        curr_len = data.shape[1]
        add_len = divby_len - curr_len%divby_len
        make_len = curr_len + add_len
        if not zeros:
            rep = make_len / curr_len + 1
            data = np.tile(data, (1, rep))
            data = np.delete(data, range(make_len, data.shape[1]), axis=1)
        else:
            data = np.hstack((data, np.zeros(shape=(data.shape[0], add_len))))
        assert data.shape[1] == make_len, 'error in fix_batch_len'
        return data

    def get_dataset(self, dataset):
        return self.data[dataset]

    def collate(self):
        assert self.force_length == True, 'Length not equal'
        ret = np.zeros(shape=(1, self.desired_length))
        for dataset in self.data:
            ret = np.append(ret, self.data[dataset]['X_train'], axis=0)
        return ret[1:, ]

    def get_length(self):
        return self.total_length

    def get_batch(self):
        # Make a minibatch of training data
        dataset_name = np.random.choice(self.datasets)
        # print dataset_name, self.data[dataset_name]['X_train'].shape
        num_train = self.data[dataset_name]['X_train'].shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.data[dataset_name]['X_train'][batch_mask]
        y_batch = self.data[dataset_name]['Y_train'][batch_mask]
        return X_batch, y_batch


def my_plot(data):
    plt.figure()
    plt.imshow(data)
    plt.show()


def ts_plot(data):
    x = data['X_train']
    if isinstance(x, dict):
        for key, value in x.iteritems():
            my_plot(x[key])
    else:
        my_plot(x)


if __name__ == "__main__":
    ucr = MultipleDatasets(directory="/home/gauravanand25/Dropbox/umass/682-nn/Project/UCR_TS_Archive_2015",
                           datasets=my_train_datasets, data_length=512, merge_train_test=True, val_percentage=0)
    ucr.load_data()
    data = ucr.collate()
    for dataset in my_train_datasets:
        ts_plot(ucr.get_dataset(dataset))
