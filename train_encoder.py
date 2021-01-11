# train autoencoder for classification with no compression in the bottleneck layer
# TODO:  add hyperparameter optimization
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
import pandas as pd
import numpy as np

# Train.csv has the Field_IDs needed to find the npy files
train = pd.read_csv("data/Train.csv")

# Look at a sample:
fid = train["Field_ID"].sample().values[0]
fn = f"data/image_arrays_train/{fid}.npy"  # File name based on Field_ID
print(f"Loading {fn} as an array")
arr = np.load(fn)  # Loading the data with numpy
print("Array shape:", arr.shape)  # 360 bands, images 40 or 41px a side
rgb_jan = np.stack([arr[4], arr[3], arr[2]], axis=-1)  # Combine three bands for viewing
rgb_jan = rgb_jan / np.max(
    rgb_jan
)  # Scale band values to (0, 1) for easy image display


# There are some hard-coded band indexes in the examples above that won't have made sense - how did we know which bands were which?
# There are 30 bands for each month. You can see the full list of bands with:

band_names = [
    "0_S2_B1",
    "0_S2_B2",
    "0_S2_B3",
    "0_S2_B4",
    "0_S2_B5",
    "0_S2_B6",
    "0_S2_B7",
    "0_S2_B8",
    "0_S2_B8A",
    "0_S2_B9",
    "0_S2_B10",
    "0_S2_B11",
    "0_S2_B12",
    "0_S2_QA10",
    "0_S2_QA20",
    "0_S2_QA60",
    "0_CLIM_aet",
    "0_CLIM_def",
    "0_CLIM_pdsi",
    "0_CLIM_pet",
    "0_CLIM_pr",
    "0_CLIM_ro",
    "0_CLIM_soil",
    "0_CLIM_srad",
    "0_CLIM_swe",
    "0_CLIM_tmmn",
    "0_CLIM_tmmx",
    "0_CLIM_vap",
    "0_CLIM_vpd",
    "0_CLIM_vs",
    "1_S2_B1",
    "1_S2_B2",
    "1_S2_B3",
    "1_S2_B4",
    "1_S2_B5",
    "1_S2_B6",
    "1_S2_B7",
    "1_S2_B8",
    "1_S2_B8A",
    "1_S2_B9",
    "1_S2_B10",
    "1_S2_B11",
    "1_S2_B12",
    "1_S2_QA10",
    "1_S2_QA20",
    "1_S2_QA60",
    "1_CLIM_aet",
    "1_CLIM_def",
    "1_CLIM_pdsi",
    "1_CLIM_pet",
    "1_CLIM_pr",
    "1_CLIM_ro",
    "1_CLIM_soil",
    "1_CLIM_srad",
    "1_CLIM_swe",
    "1_CLIM_tmmn",
    "1_CLIM_tmmx",
    "1_CLIM_vap",
    "1_CLIM_vpd",
    "1_CLIM_vs",
    "2_S2_B1",
    "2_S2_B2",
    "2_S2_B3",
    "2_S2_B4",
    "2_S2_B5",
    "2_S2_B6",
    "2_S2_B7",
    "2_S2_B8",
    "2_S2_B8A",
    "2_S2_B9",
    "2_S2_B10",
    "2_S2_B11",
    "2_S2_B12",
    "2_S2_QA10",
    "2_S2_QA20",
    "2_S2_QA60",
    "2_CLIM_aet",
    "2_CLIM_def",
    "2_CLIM_pdsi",
    "2_CLIM_pet",
    "2_CLIM_pr",
    "2_CLIM_ro",
    "2_CLIM_soil",
    "2_CLIM_srad",
    "2_CLIM_swe",
    "2_CLIM_tmmn",
    "2_CLIM_tmmx",
    "2_CLIM_vap",
    "2_CLIM_vpd",
    "2_CLIM_vs",
    "3_S2_B1",
    "3_S2_B2",
    "3_S2_B3",
    "3_S2_B4",
    "3_S2_B5",
    "3_S2_B6",
    "3_S2_B7",
    "3_S2_B8",
    "3_S2_B8A",
    "3_S2_B9",
    "3_S2_B10",
    "3_S2_B11",
    "3_S2_B12",
    "3_S2_QA10",
    "3_S2_QA20",
    "3_S2_QA60",
    "3_CLIM_aet",
    "3_CLIM_def",
    "3_CLIM_pdsi",
    "3_CLIM_pet",
    "3_CLIM_pr",
    "3_CLIM_ro",
    "3_CLIM_soil",
    "3_CLIM_srad",
    "3_CLIM_swe",
    "3_CLIM_tmmn",
    "3_CLIM_tmmx",
    "3_CLIM_vap",
    "3_CLIM_vpd",
    "3_CLIM_vs",
    "4_S2_B1",
    "4_S2_B2",
    "4_S2_B3",
    "4_S2_B4",
    "4_S2_B5",
    "4_S2_B6",
    "4_S2_B7",
    "4_S2_B8",
    "4_S2_B8A",
    "4_S2_B9",
    "4_S2_B10",
    "4_S2_B11",
    "4_S2_B12",
    "4_S2_QA10",
    "4_S2_QA20",
    "4_S2_QA60",
    "4_CLIM_aet",
    "4_CLIM_def",
    "4_CLIM_pdsi",
    "4_CLIM_pet",
    "4_CLIM_pr",
    "4_CLIM_ro",
    "4_CLIM_soil",
    "4_CLIM_srad",
    "4_CLIM_swe",
    "4_CLIM_tmmn",
    "4_CLIM_tmmx",
    "4_CLIM_vap",
    "4_CLIM_vpd",
    "4_CLIM_vs",
    "5_S2_B1",
    "5_S2_B2",
    "5_S2_B3",
    "5_S2_B4",
    "5_S2_B5",
    "5_S2_B6",
    "5_S2_B7",
    "5_S2_B8",
    "5_S2_B8A",
    "5_S2_B9",
    "5_S2_B10",
    "5_S2_B11",
    "5_S2_B12",
    "5_S2_QA10",
    "5_S2_QA20",
    "5_S2_QA60",
    "5_CLIM_aet",
    "5_CLIM_def",
    "5_CLIM_pdsi",
    "5_CLIM_pet",
    "5_CLIM_pr",
    "5_CLIM_ro",
    "5_CLIM_soil",
    "5_CLIM_srad",
    "5_CLIM_swe",
    "5_CLIM_tmmn",
    "5_CLIM_tmmx",
    "5_CLIM_vap",
    "5_CLIM_vpd",
    "5_CLIM_vs",
    "6_S2_B1",
    "6_S2_B2",
    "6_S2_B3",
    "6_S2_B4",
    "6_S2_B5",
    "6_S2_B6",
    "6_S2_B7",
    "6_S2_B8",
    "6_S2_B8A",
    "6_S2_B9",
    "6_S2_B10",
    "6_S2_B11",
    "6_S2_B12",
    "6_S2_QA10",
    "6_S2_QA20",
    "6_S2_QA60",
    "6_CLIM_aet",
    "6_CLIM_def",
    "6_CLIM_pdsi",
    "6_CLIM_pet",
    "6_CLIM_pr",
    "6_CLIM_ro",
    "6_CLIM_soil",
    "6_CLIM_srad",
    "6_CLIM_swe",
    "6_CLIM_tmmn",
    "6_CLIM_tmmx",
    "6_CLIM_vap",
    "6_CLIM_vpd",
    "6_CLIM_vs",
    "7_S2_B1",
    "7_S2_B2",
    "7_S2_B3",
    "7_S2_B4",
    "7_S2_B5",
    "7_S2_B6",
    "7_S2_B7",
    "7_S2_B8",
    "7_S2_B8A",
    "7_S2_B9",
    "7_S2_B10",
    "7_S2_B11",
    "7_S2_B12",
    "7_S2_QA10",
    "7_S2_QA20",
    "7_S2_QA60",
    "7_CLIM_aet",
    "7_CLIM_def",
    "7_CLIM_pdsi",
    "7_CLIM_pet",
    "7_CLIM_pr",
    "7_CLIM_ro",
    "7_CLIM_soil",
    "7_CLIM_srad",
    "7_CLIM_swe",
    "7_CLIM_tmmn",
    "7_CLIM_tmmx",
    "7_CLIM_vap",
    "7_CLIM_vpd",
    "7_CLIM_vs",
    "8_S2_B1",
    "8_S2_B2",
    "8_S2_B3",
    "8_S2_B4",
    "8_S2_B5",
    "8_S2_B6",
    "8_S2_B7",
    "8_S2_B8",
    "8_S2_B8A",
    "8_S2_B9",
    "8_S2_B10",
    "8_S2_B11",
    "8_S2_B12",
    "8_S2_QA10",
    "8_S2_QA20",
    "8_S2_QA60",
    "8_CLIM_aet",
    "8_CLIM_def",
    "8_CLIM_pdsi",
    "8_CLIM_pet",
    "8_CLIM_pr",
    "8_CLIM_ro",
    "8_CLIM_soil",
    "8_CLIM_srad",
    "8_CLIM_swe",
    "8_CLIM_tmmn",
    "8_CLIM_tmmx",
    "8_CLIM_vap",
    "8_CLIM_vpd",
    "8_CLIM_vs",
    "9_S2_B1",
    "9_S2_B2",
    "9_S2_B3",
    "9_S2_B4",
    "9_S2_B5",
    "9_S2_B6",
    "9_S2_B7",
    "9_S2_B8",
    "9_S2_B8A",
    "9_S2_B9",
    "9_S2_B10",
    "9_S2_B11",
    "9_S2_B12",
    "9_S2_QA10",
    "9_S2_QA20",
    "9_S2_QA60",
    "9_CLIM_aet",
    "9_CLIM_def",
    "9_CLIM_pdsi",
    "9_CLIM_pet",
    "9_CLIM_pr",
    "9_CLIM_ro",
    "9_CLIM_soil",
    "9_CLIM_srad",
    "9_CLIM_swe",
    "9_CLIM_tmmn",
    "9_CLIM_tmmx",
    "9_CLIM_vap",
    "9_CLIM_vpd",
    "9_CLIM_vs",
    "10_S2_B1",
    "10_S2_B2",
    "10_S2_B3",
    "10_S2_B4",
    "10_S2_B5",
    "10_S2_B6",
    "10_S2_B7",
    "10_S2_B8",
    "10_S2_B8A",
    "10_S2_B9",
    "10_S2_B10",
    "10_S2_B11",
    "10_S2_B12",
    "10_S2_QA10",
    "10_S2_QA20",
    "10_S2_QA60",
    "10_CLIM_aet",
    "10_CLIM_def",
    "10_CLIM_pdsi",
    "10_CLIM_pet",
    "10_CLIM_pr",
    "10_CLIM_ro",
    "10_CLIM_soil",
    "10_CLIM_srad",
    "10_CLIM_swe",
    "10_CLIM_tmmn",
    "10_CLIM_tmmx",
    "10_CLIM_vap",
    "10_CLIM_vpd",
    "10_CLIM_vs",
    "11_S2_B1",
    "11_S2_B2",
    "11_S2_B3",
    "11_S2_B4",
    "11_S2_B5",
    "11_S2_B6",
    "11_S2_B7",
    "11_S2_B8",
    "11_S2_B8A",
    "11_S2_B9",
    "11_S2_B10",
    "11_S2_B11",
    "11_S2_B12",
    "11_S2_QA10",
    "11_S2_QA20",
    "11_S2_QA60",
    "11_CLIM_aet",
    "11_CLIM_def",
    "11_CLIM_pdsi",
    "11_CLIM_pet",
    "11_CLIM_pr",
    "11_CLIM_ro",
    "11_CLIM_soil",
    "11_CLIM_srad",
    "11_CLIM_swe",
    "11_CLIM_tmmn",
    "11_CLIM_tmmx",
    "11_CLIM_vap",
    "11_CLIM_vpd",
    "11_CLIM_vs",
]


def process_im(fid, folder="data/image_arrays_train"):
    fn = f"{folder}/{fid}.npy"
    arr = np.load(fn)
    bands_of_interest = ["S2_B5", "S2_B4", "S2_B3", "S2_B2", "CLIM_pr", "CLIM_soil"]
    values = {}
    for month in range(12):
        bns = [
            str(month) + "_" + b for b in bands_of_interest
        ]  # Bands of interest for this month
        idxs = np.where(np.isin(band_names, bns))  # Index of these bands
        vs = arr[idxs, 20, 20]  # Sample the im at the center point
        for bn, v in zip(bns, vs[0]):
            values[bn] = v
    return values


# Example
# process_im('35AFSDD')

"""With this, we can sample the inputs for each field in train and use that to build a dataframe of input features:"""

# Make a new DF with the sampled values from each field
train_sampled = pd.DataFrame([process_im(fid) for fid in train["Field_ID"].values])

# Add in the field ID and yield
train_sampled["Field_ID"] = train["Field_ID"].values
train_sampled["Yield"] = train["Yield"].values

X, y = train_sampled[train_sampled.columns[:-2]], train_sampled["Yield"]

# number of input columns
n_inputs = X.shape[1]
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=1
)
# scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)
# define encoder
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs * 2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = round(float(n_inputs) / 2.0)
bottleneck = Dense(n_bottleneck)(e)
# define decoder, level 1
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(n_inputs * 2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation="linear")(d)
# define autoencoder model
model = Model(inputs=visible, outputs=output)
# compile autoencoder model
model.compile(optimizer="adam", loss="mse")
# plot the autoencoder
plot_model(model, "autoencoder_no_compress.png", show_shapes=True)
# fit the autoencoder model to reconstruct input
history = model.fit(
    X_train,
    X_train,
    epochs=200,
    batch_size=16,
    verbose=2,
    validation_data=(X_test, X_test),
)
# plot loss
pyplot.plot(history.history["loss"], label="train")
pyplot.plot(history.history["val_loss"], label="test")
pyplot.legend()
pyplot.show()
# define an encoder model (without the decoder)
encoder = Model(inputs=visible, outputs=bottleneck)
plot_model(encoder, "encoder_no_compress.png", show_shapes=True)
# save the encoder to file
encoder.save("encoder.h5")
