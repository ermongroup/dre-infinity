"""
adapted from from: https://github.com/openai/improved-diffusion/
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


class LossSecondMomentResampler(object):
  def __init__(self, batch_size, history_per_term=10):
    self.batch_size = batch_size
    self.history_per_term = history_per_term
    self.uniform_prob = 1./batch_size
    # float 64?
    self._loss_history = np.zeros([batch_size, history_per_term], dtype=np.float64)
    self._time_history = np.zeros([batch_size, history_per_term], dtype=np.float64)
    self._loss_counts = np.zeros([batch_size], dtype=np.int)

  def weights(self):
    if not self._warmed_up():
      return np.ones([self.batch_size], dtype=np.float64)
    weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
    weights /= np.sum(weights)
    weights *= 1 - self.uniform_prob
    weights += self.uniform_prob / len(weights)
    # return absolute value of the weights
    return np.abs(weights)

  def update_with_all_losses(self, ts, losses):
    for i, (t, loss) in enumerate(zip(ts, losses)):
      if self._loss_counts[i] == self.history_per_term:
        # Shift out the oldest loss term.
        self._loss_history[i, :-1] = self._loss_history[i, 1:]
        self._loss_history[i, -1] = loss
        # Save timesteps
        # at the moment _time_history won't be sorted!
        self._time_history[i, :-1] = self._time_history[i, 1:]
        self._time_history[i, -1] = t
      else:
        self._loss_history[i, self._loss_counts[i]] = loss
        self._time_history[i, self._loss_counts[i]] = t
        self._loss_counts[i] += 1

  def _warmed_up(self):
    return (self._loss_counts == self.history_per_term).all()


class InterpolateLossSecondMomentResampler(object):
  def __init__(self, batch_size, history_per_term=10):
    self.batch_size = batch_size
    self.history_per_term = history_per_term
    self.uniform_prob = 1./batch_size
    # float 64?
    self._loss_history = np.zeros([batch_size, history_per_term], dtype=np.float64)
    self._time_history = np.zeros([batch_size, history_per_term], dtype=np.float64)
    self._weight_history = np.zeros([batch_size, history_per_term], dtype=np.float64)
    self._loss_counts = np.zeros([batch_size], dtype=np.int)

  def weights(self, ts):
    if not self._warmed_up():
      return np.ones([self.batch_size], dtype=np.float64)

    # if we just finished filling up the buffer, all the weights will be 1
    # this operation will happen once
    if not self._initialized_weights():
      print('initializing weights after filling up buffer for the first time!')
      for i in range(self._weight_history.shape[1]):
        self._weight_history[:, i] = self.warmup_weights(i+1)

    # try polynomial interpolation instead
    model = make_pipeline(
      PolynomialFeatures(4),
      Ridge(alpha=1e-3)
    )
    model.fit(self._time_history.reshape(-1, 1), self._weight_history.reshape(-1, 1))  # (buffer, 1)
    w_hat = model.predict(ts.reshape(-1, 1)).reshape(-1)

    # return absolute value of the weights
    return np.abs(w_hat)

  def warmup_weights(self, i):
    # the original weighting function, modified to operate over a subset of the losses
    weights = np.sqrt(np.mean(self._loss_history[:, :i] ** 2, axis=-1))
    weights /= np.sum(weights)
    weights *= 1 - self.uniform_prob
    weights += self.uniform_prob / len(weights)

    # let's rescale for the time being
    weights /= weights.max()
    return weights

  def update_with_all_losses(self, ts, losses, weights):
    for i, (t, loss, ws) in enumerate(zip(ts, losses, weights)):
      if self._loss_counts[i] == self.history_per_term:
        # Shift out the oldest loss term.
        self._loss_history[i, :-1] = self._loss_history[i, 1:]
        self._loss_history[i, -1] = loss
        # Save timesteps
        self._time_history[i, :-1] = self._time_history[i, 1:]
        self._time_history[i, -1] = t
        self._weight_history[i, :-1] = self._weight_history[i, 1:]
        self._weight_history[i, -1] = ws
      else:
        self._loss_history[i, self._loss_counts[i]] = loss
        self._time_history[i, self._loss_counts[i]] = t
        self._weight_history[i, self._loss_counts[i]] = ws
        self._loss_counts[i] += 1

  def _warmed_up(self):
    return (self._loss_counts == self.history_per_term).all()

  def _initialized_weights(self):
    return self._weight_history.sum() != (self.batch_size * self.history_per_term)