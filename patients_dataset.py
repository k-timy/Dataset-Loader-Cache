import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
import pickle



class PatientsDataset(Dataset):
    def __init__(self, path,force_recache):
        self.path = path
        self.dataset = []
        this_script_path = os.path.dirname(os.path.realpath(__file__))
        self.cache_file_name = os.path.join(this_script_path,'patients_data.cache')
        self.load_from_cache(force_recache)

    def loaddata(self):
        """
        This method is an arbitrary method. Depending on how you store your dataset, you can rewrite it.
        For querying from database, or reading from multiple files, or possibly performing any data
        pre-processing tasks,etc.

        :return: does not return anything
        """
        files = os.listdir(self.path)

        for f in files:
            # skip irrelevant files
            if 'patient' not in f:
                continue
            pid = f.split('.')[0]
            df = pd.read_csv(os.path.join(self.path, f))
            df.drop('Unnamed: 0', axis=1, inplace=True)

            vals = df.values

            # Adding the data of patients to a list, in format of tuples:
            # first the data, then patient ids. Other data such as labels,
            # etc can be added to this tuple based on the problem and model of the experiment.
            self.dataset.append((vals,pid))


    def cache_dataset(self,force_recache=False):
        """
        # Store the preprocessed data in a single file
        # if a cache file is not already available
        :return: does not return anything.
        """
        if os.path.exists(self.cache_file_name) and not force_recache:
            return
        else:
            with open(self.cache_file_name, 'wb') as f:
                pickle.dump(self.dataset, f)
                
    # Load data from cache file
    def load_from_cache(self,force_recache=False):
        """
        Load the data from the cache, if there is any cache files available. Otherwise, load it directly from its
         source. Either a database or multiple files.
        :param force_recache: Force the function to load the data from original files and store in cache again. Without
        considering if there is any cache file available or not.
        :return:
        """
        if os.path.exists(self.cache_file_name):
            if force_recache == True:
                self.loaddata()
                self.cache_dataset(force_recache)
                print('Loaded dataset and stored in cache.')
            else:
                with open(self.cache_file_name, 'rb') as f:
                    self.dataset = pickle.load(f)
                print('Loaded dataset from cache.')
        else:
            self.loaddata()
            self.cache_dataset()
            print('Loaded dataset and stored in cache.')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
