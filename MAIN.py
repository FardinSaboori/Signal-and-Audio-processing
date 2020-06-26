import wave
from scipy.io.wavfile import read, write
import io
import matplotlib.pyplot as plt
import numpy as np
import pywt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import pandas as pd

# READ THE FILE
audio_file = "./audio/ahem_x.wav"
# print(audio_file)
wf = wave.open("./audio/ahem_x.wav", 'rb')
# data, samplerate = sf.read(wf)

with open("./audio/ahem_x.wav", "rb") as wavfile:
    input_wav = wavfile.read()
data, rate = read(io.BytesIO(input_wav))
print(data)

# PLOTTING THE SIGNAL USING SCALOGRAM AND 3D PLOTTING
plt.plot(rate)
plt.show()
# Continuous Wavelet Transform (CWT) and Scalogram plotting
scale = np.arange(1, 101)
coef, freq = pywt.cwt(rate[:10000], scale, 'morl')
plt.imshow(coef, cmap='coolwarm', aspect='auto')
plt.show()
print(coef)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(1, 10001, 1) # points in the signal
y = np.arange(1, 101, 1) # scale of 1-100
x, y = np.meshgrid(x, y)

ax.plot_surface(x, y, coef, color='b')
plt.show()
# If your learning algorithm is too slow because the input dimension is too high,
# then using PCA to speed it up can be a reasonable choice.
print(coef.shape)
pca = PCA(n_components=1)
features = np.empty((0, 100))

features = np.vstack([features, pca.fit_transform(coef).flatten()])
newarr = features.reshape(-1)
print(newarr.shape)

# here i made a dataframe to perform a regression and prediction model on the data which obviously
# results in a low correlation score:
labels = np.array(range(100))
print(labels.shape)
df1 = pd.DataFrame(newarr, columns=['coef'])
df2 = pd.DataFrame(labels, columns=['number'])
print(df2)
df = pd.concat([df2, df1], sort=False, axis=1)
print(df[:70])
# here we can split the data into train and test and based on the xdata, make a prediction for our target data.
xtrain, xtest, ytran, ytest = train_test_split(df[['coef']], df['number'], test_size=0.2, random_state=0)
# Support vector machines (SVMs)
clf = SVC()
lr = LinearRegression()
clf.fit(xtrain, ytran)
lr.fit(xtrain, ytran)
t = lr.score(xtest, ytest)
print(t)
#ypredicted = clf.predict(xtest)
#print(accuracy_score(ytest, ypredicted)*100)