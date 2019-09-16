import numpy as np
import pandas as pd

np.random.seed(0)


def generate_fake_data(data_count = 10000,col_size=50, min_row_size=24,max_row_size=80):
    """
    Generates a fake dataset of patients with length equals to data_count in format of multivariate time series.
    Vectors have col_size dimensions And the lengths of time series differ between min_row_size and max_row_size.
    """
    for i in range(data_count):
        rows = np.random.randint(min_row_size,max_row_size)
        patient_data = np.random.rand(rows,col_size)
        # Each DataFrame represents the multivariate timeseries of a single patient.
        df = pd.DataFrame(patient_data)
        df.to_csv('data/{}.patient'.format(i))