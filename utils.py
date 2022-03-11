import torch
import tensorflow as tf
import os
import logging
import numpy as np


def restore_checkpoint(ckpt_dir, state, device, test=False):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    # TODO: do we need this?
    if not test:
      state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    try:
      state['scheduler'] = loaded_state['scheduler']
    except:
      pass
    return state


def load_history(file_path, history, interpolate=False):
  record = np.load(os.path.join(file_path, 'history.npz'))
  history._loss_history = record['loss_history']

  # for previously trained models, these two things may not have been saved
  try:
    history._time_history = record['time_history']
  except:
    print('time history had not been saved, skipping...')

  try:
    history._loss_counts = record['loss_counts']
  except:
    print('loss counts have not been saved, automatically warming up')
    history._loss_counts = np.ones([history.batch_size], dtype=np.int) * history.history_per_term

  # only interpolation has a saved weight history
  assert history._warmed_up()
  if interpolate:
    history._weight_history = record['weight_history']
    assert history._initialized_weights()
    print('history weights have been initialized from prior run!')

  return history


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
    # 'scheduler': state['scheduler']
  }
  torch.save(saved_state, ckpt_dir)