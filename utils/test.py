import pywt


DATA = [0,1,2,3,4,5,6,7,8,9]
N_LEVELS = 2
WAVELET_NAME = 'db1'
A1,D1 = pywt.wavedec(DATA, WAVELET_NAME, level=1)

print(A1)

print(D1)