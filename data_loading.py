import numpy as np


def scale(data):
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data


def real_data_loading (data_name, seq_len):
  assert data_name in ['stock','energy']

  if data_name == 'stock':
    ori_data = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('data/energy_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'yahoo':
    ori_data = np.loadtxt('data/yahoo_stock.csv', delimiter = ",",skiprows = 1, usecols=(2, 3, 4, 5, 6, 7))

  # Flip the data to make chronological data
  ori_data = ori_data[::-1]
  # Normalize the data
  ori_data = scale(ori_data)

  # Preprocess the dataset
  temp_data = []
  # Cut data by sequence length
  for i in range(0, len(ori_data) - seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)

  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])

  return data
