import argparse
import numpy as np
from generate import generate
from data_loading import data_loading
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics

default_args = {
    'data_name': 'stock',
    'metric_iteration': 1,
    'seq_len': 24,
    'module': 'gru',
    'lr': 1e-3,
    'hidden_size': 24,
    'num_layers': 3,
    'iterations': 50000,
    'batch_size': 32,
    'device': 'gpu'
}

def main(args=default_args):
  ori_data = data_loading(args.data_name, args.seq_len)

  print(args.data_name + ' dataset is ready.')

  parameters = dict()
  parameters['module'] = args.module
  parameters['lr'] = args.lr
  parameters['hidden_size'] = args.hidden_size
  parameters['num_layers'] = args.num_layers
  parameters['iterations'] = args.iteration
  parameters['batch_size'] = args.batch_size
  parameters['device'] = args.device

  generated_data = generate(ori_data, parameters)
  print('Finish Synthetic Data Generation')

  metric_results = dict()

  discriminative_score = list()
  for _ in range(args.metric_iteration):
    temp_disc = discriminative_score_metrics(ori_data, generated_data, parameters['device'])
    discriminative_score.append(temp_disc)

  metric_results['discriminative'] = np.mean(discriminative_score)

  predictive_score = list()
  for tt in range(args.metric_iteration):
    temp_pred = predictive_score_metrics(ori_data, generated_data, parameters['device'])
    predictive_score.append(temp_pred)

  metric_results['predictive'] = np.mean(predictive_score)

  print(metric_results)

  return ori_data, generated_data, metric_results

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--exp_name',
      default='my-exp',
      type=str)
  parser.add_argument(
      '--data_name',
      choices=['yahoo','stock','energy'],
      default='stock',
      type=str)
  parser.add_argument(
      '--seq_len',
      help='sequence length',
      default=24,
      type=int)
  parser.add_argument(
      '--module',
      choices=['gru','lstm','lstmLN'],
      default='gru',
      type=str)
  parser.add_argument(
      '--hidden_size',
      help='hidden state dimensions (should be optimized)',
      default=24,
      type=int)
  parser.add_argument(
      '--num_layers',
      help='number of layers (should be optimized)',
      default=3,
      type=int)
  parser.add_argument(
      '--iteration',
      help='Training iterations (should be optimized)',
      default=1,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch (should be optimized)',
      default=32,
      type=int)
  parser.add_argument(
      '--metric_iteration',
      help='iterations of the metric computation',
      default=10,
      type=int)
  parser.add_argument(
      '--lr',
      default=1e-3,
      type=float)
  parser.add_argument(
      '--device',
      default='cpu',
      type=str)

  args = parser.parse_args()

  ori_data, generated_data, metrics = main(args=args)
  np.save(f'experiments/{args.exp_name}_gen.npy', generated_data)
  np.save(f'experiments/{args.exp_name}_ori.npy', ori_data)