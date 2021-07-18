import segyio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def loadsgy(filename):
  with segyio.open(filename, 'r', iline=5, xline=21) as segyfile:
    segyfile.mmap()
    data = segyio.tools.cube(segyfile)
    ntraces = segyfile.tracecount
    sr = segyio.tools.dt(segyfile)
    nsamples = segyfile.samples.size
    twt = segyfile.samples
    size_mb= data.nbytes/1024**2
    inlines = segyfile.ilines
    crosslines = segyfile.xlines
    header = segyio.tools.wrap(segyfile.text[0])
    return data
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def load_well(filename):
    well = np.loadtxt(filename, skiprows=15)
    return well
# Load Data
# Relative Impedance
lfr = loadsgy('export_model_P-Impedance.sgy')
lfr = lfr.transpose(2, 0, 1).reshape(-1, lfr.shape[1])
# Seismic
seis = loadsgy('arbitrary_line.sgy')
seis = seis.transpose(2, 0, 1).reshape(-1, seis.shape[1])
# Amplitude Envelope
env = loadsgy('envelope.sgy')
env = env.transpose(2, 0, 1).reshape(-1, env.shape[1])
# Quadrature
quad = loadsgy('quadrature_al.sgy')
quad = quad.transpose(2, 0, 1).reshape(-1, quad.shape[1])
# Well
y_f21 = load_well('./data/01-F02-1_logs.txt')
y_f32 = load_well('./data/02-F03-2_logs.txt')
y_f34 = load_well('./data/03-F03-4_logs.txt')
y_f61 = load_well('./data/04-F06-1_logs.txt')
# Slicing Data
lfr_f21 = lfr[130:275, 0]
seis_f21 = seis[130:275, 0]
env_f21 = env[130:275, 0]
quad_f21 = quad[130:275, 0]

lfr_f32 = lfr[130:275, 1017]
seis_f32 = seis[130:275, 1017]
env_f32 = env[130:275, 1017]
quad_f32 = quad[130:275, 1017]

lfr_f34 = lfr[130:275, 738]
seis_f34 = seis[130:275, 738]
env_f34 = env[130:275, 738]
quad_f34 = quad[130:275, 738]

lfr_f61 = lfr[130:275, 117]
seis_f61 = seis[130:275, 117]
env_f61 = env[130:275, 117]
quad_f61 = quad[130:275, 117]

y_f21 = y_f21[130:275, 2].reshape(145,1)
y_f32 = y_f32[130:275, 2].reshape(145,1)
y_f34 = y_f34[130:275, 2].reshape(145,1)
y_f61 = y_f61[130:275, 2].reshape(145,1)
print(y_f21)
# filtering
y_f21 = y_f21.flatten()
y_f32 = y_f32.flatten()
y_f34 = y_f34.flatten()
y_f61 = y_f61.flatten()

y_f21_f = butter_lowpass_filter(y_f21, 70, 1000/4, order=5)
y_f32_f = butter_lowpass_filter(y_f32, 70, 1000/4, order=5)
y_f34_f = butter_lowpass_filter(y_f34, 70, 1000/4, order=5)
y_f61_f = butter_lowpass_filter(y_f61, 70, 1000/4, order=5)

# y_f21_f = butter_bandpass_filter(y_f21, 35, 70, 1000/4, order=5)
# y_f32_f = butter_bandpass_filter(y_f32, 35, 70, 1000/4, order=5)
# y_f34_f = butter_bandpass_filter(y_f34, 35, 70, 1000/4, order=5)
# y_f61_f = butter_bandpass_filter(y_f61, 35, 70, 1000/4, order=5)

# Selecting Data Training test split
x1 = np.vstack((seis_f21, env_f21, quad_f21, lfr_f21))
x2 = np.vstack((seis_f32, env_f32,quad_f32, lfr_f32))
x1 = x1.T
x2 = x2.T
xtrain = np.vstack((x1, x2))
ytrain = np.hstack((y_f21_f, y_f32_f)).T
ytrain = ytrain.reshape(-1, 1)

# Selecting data blind test
x_bt1 = np.vstack((seis_f34,  env_f34,quad_f34, lfr_f34)).T
x_bt2 = np.vstack((seis_f61,  env_f61,quad_f61, lfr_f61)).T
y_bt1 = y_f34_f.reshape(-1, 1)
y_bt2 = y_f61_f.reshape(-1, 1)
# Scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler_x = StandardScaler()
scaler_y = MinMaxScaler()
xscale=scaler_x.fit_transform(xtrain)
yscale=scaler_y.fit_transform(ytrain)

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras import initializers

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale, test_size=0.2)
model = Sequential()
model.add(Dense(1000, input_dim=xtrain.shape[1], kernel_initializer='random_normal',
                activation='relu', use_bias = True, bias_initializer = initializers.Constant(0.1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history = model.fit(X_train, y_train, epochs= 20,validation_data = (X_test,y_test))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

# Blind test Process F034
x_bt1 = scaler_x.transform(x_bt1)
ypred = model.predict(x_bt1)
ypred = scaler_y.inverse_transform(ypred)
plt.figure()
plt.title('Predict Impedansi Sumur F-034')
plt.plot(ypred, label = 'Predicted')
plt.plot(y_bt1, label = 'Original')
plt.legend()
# Blind test Process F061
x_bt2 = scaler_x.transform(x_bt2)
ypred2 = model.predict(x_bt2)
ypred2 = scaler_y.inverse_transform(ypred2)
plt.figure()
plt.title('Predict Impedansi Sumur F-061')
plt.plot(ypred2, label = 'Predicted')
plt.plot(y_bt2, label = 'Original')
plt.legend()

# crossplot
from scipy.stats import pearsonr
x = np.vstack((y_bt1, y_bt2))
y = np.vstack((ypred, ypred2))
plt.figure()
plt.title('Crossplot Zp Original vs Zp Predicted')
plt.xlabel('Predicted')
plt.ylabel('Original')
zx = x.flatten()
zy = y.flatten()
corr,_ = pearsonr(zx, zy)
print(corr)
plt.scatter(x, y, label = ('correlation %s' %corr))
plt.legend()

data_all = []
for i in range(seis.shape[1]):
    seis_t = seis[:, i]
    lfr_t = lfr[:, i]
    env_t = env[:, i]
    quad_t = quad[:, i]
    data_all.append(seis_t)
    data_all.append(lfr_t)
    data_all.append(env_t)
    data_all.append(quad_t)
data_all = np.transpose(data_all)
impedansi = []
column = xtrain.shape[1]
for i in range(seis.shape[1]):
    seis_pred = data_all[:, (0+((i+1)-1)*column):(column+((i+1)-1)*column)]
    seis_pred_scale = scaler_x.fit_transform(seis_pred)
    testing = model.predict(seis_pred_scale)
    pred2d = scaler_y.inverse_transform(testing)
    impedansi.append(pred2d)
impedansi = np.array(impedansi)
impedansi = impedansi.transpose(2, 0, 1).reshape(-1, impedansi.shape[1])
impedansi = impedansi.T
plt.figure()
plt.imshow(seis, aspect='auto', cmap='seismic')
plt.xlabel('cdp')
plt.ylabel('twt/4')
plt.colorbar(label = 'Amplitude')
plt.title('Seismic')

plt.figure()
plt.imshow(quad, aspect='auto', cmap='seismic')
plt.xlabel('cdp')
plt.ylabel('twt/4')
plt.colorbar(label = 'Amplitude')
plt.title('Quadrature')

plt.figure()
plt.imshow(env, aspect='auto', cmap='seismic')
plt.xlabel('cdp')
plt.ylabel('twt/4')
plt.colorbar()
plt.title('Amplitude Envelope')

plt.figure()
plt.imshow(lfr, aspect='auto', cmap='seismic')
plt.xlabel('cdp')
plt.ylabel('twt/4')
plt.colorbar(label = 'm/s*g/cc')
plt.title('Low Frequency Model')

from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
plt.figure()
plt.title('Penampang Impedansi P')
jet = plt.get_cmap('rainbow')
jet = jet(np.linspace(4358000, 5444000, 50))
jet = np.array(jet)
cmap = ListedColormap(jet)
bounds = np.linspace(350000, 650000, 50)
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.imshow(impedansi, aspect='auto', cmap= 'gist_rainbow',vmin = 4200000, vmax = 5500000,interpolation='bilinear')
plt.xlabel('cdp')
plt.ylabel('twt/4')
# plt.ylim([300, 100])
v1 = np.linspace(4200, 5500, 8)
plt.colorbar(label = 'm/s*g/cc', format = '%.d')

plt.tight_layout()

plt.figure()
y = np.arange(0, 145, 1)
y1 = seis_f21 * (0 )
plt.subplot(1, 4, 1)
plt.plot(seis_f21, y, 'black')
plt.title('Seismic Trace-F021')
plt.fill_betweenx(y1, seis_f21, where=(seis_f21>=y1), color = 'black')
plt.subplot(1, 4, 2)
plt.plot(quad_f21, y, 'black')
plt.title('Quadrature Trace-F021')
plt.subplot(1, 4, 3)
plt.plot(env_f21, y, 'black')
plt.title('Amp Env Trace-F021')
plt.subplot(1, 4, 4)
plt.plot(lfr_f21, y, 'black')
plt.title('LFR Trace-F021')

plt.figure()
y = np.arange(0, 145, 1)
plt.subplot(1, 4, 1)
plt.plot(seis_f32, y, 'black')
plt.title('Seismic Trace-F032')
plt.subplot(1, 4, 2)
plt.plot(quad_f32, y, 'black')
plt.title('Quadrature Trace-F032')
plt.subplot(1, 4, 3)
plt.plot(env_f32, y, 'black')
plt.title('Amp Env Trace-F032')
plt.subplot(1, 4, 4)
plt.plot(lfr_f32, y, 'black')
plt.title('LFR Trace-F032')

plt.show()
