from fake_data_generator import generate_fake_data
from patients_dataset import PatientsDataset
import os
import time
import numpy as np

np.random.seed(0)

def main():
    print('Generating fake patients data...')
    generate_fake_data(10000)
    print('Patients data are generated in separate files' + os.linesep)

    print('Loading data without cache.')
    before = time.time()
    dataset = PatientsDataset('data',True)
    after = time.time()
    print('Loaded data without using cache in: {0:.2f} seconds.{1}'.format(after - before,os.linesep))

    print('Loading data with cache:')
    before = time.time()
    dataset = PatientsDataset('data', False)
    after = time.time()
    print('Loaded data with using cache in: {0:.2f} seconds.{1}'.format(after - before,os.linesep))


if __name__ == '__main__':
    main()