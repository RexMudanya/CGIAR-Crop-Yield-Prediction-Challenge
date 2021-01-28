# todo: load data method
# todo: data prep
# todo: create models
# todo: score models
# todo: create submission csv for each model
# todo: add hyperparameter optimization to encoder and models
# todo: pick best method based on RMSE

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from constants import band_names

"""
data prep:
1. scale data
2. encode data (auto encoder)
"""

"""
create models:
1. add hyperparameter optimization
2. output score, RMSE
3. create submission script for each model
"""

"""
create submission:
1. add bools for :
- scaled only
- scale + encoded
"""


def process_im(fid, folder='data/image_arrays_train'):
  fn = f'{folder}/{fid}.npy'
  arr = np.load(fn)
  bands_of_interest = ['S2_B5', 'S2_B4', 'S2_B3', 'S2_B2', 'CLIM_pr', 'CLIM_soil']
  values = {}
  for month in range(12):
    bns = [str(month) + '_' + b for b in bands_of_interest] # Bands of interest for this month
    idxs = np.where(np.isin(band_names, bns)) # Index of these bands
    vs = arr[idxs, 20, 20] # Sample the im at the center point
    for bn, v in zip(bns, vs[0]):
      values[bn] = v
  return values


def make_submission(model, filename=None, encoded=False):
    '''

    :param model: object fit to the data
    :param filename: string for .csv to save the predictions
    :param encoded: if the model was fitted with encoder data
    :return: None
    '''

    if filename is None:
        filename = "Sub.csv"

    sample_submission = pd.read_csv("data/SampleSubmission.csv")

    # Prep the data, using the same method we did for train
    test_sampled = pd.DataFrame(
        [
            process_im(fid, folder="data/image_arrays_test")
            for fid in sample_submission["Field_ID"].values
        ]
    )

    # scale the data
    t = MinMaxScaler()
    t.fit(test_sampled)
    test_sampled = t.transform(test_sampled)

    if encoded:
        # load encoder
        encoder = load_model("encoder.h5", compile=False)
        test_sampled = encoder.predict(test_sampled)

    # Get model predictions
    predictions = model.predict(test_sampled)

    # Store them in the submission dataframe and save
    sample_submission["Yield"] = predictions
    sample_submission.to_csv(filename, index=False)
