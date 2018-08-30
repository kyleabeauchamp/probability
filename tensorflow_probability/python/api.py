# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Higher level API for Bayesian Inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six import iteritems

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import edward2 as ed
import copy
import collections


__all__ = [
    'BaseModel',
    'sample'
]


class MetaModel(type):

  def __call__(cls, *args, **kwargs):
    obj = type.__call__(cls, *args, **kwargs)
    obj._load_observed()
    obj._load_unobserved()
    return obj


class BaseModel(object):
  __metaclass__ = MetaModel

  def _load_unobserved(self):
    print("load unobs")
    unobserved_fun = self._unobserved_vars()
    self.unobserved = unobserved_fun()

  def _load_observed(self):
    self.observed = copy.copy(vars(self))

  def _unobserved_vars(self):

    def unobserved_fn(*args, **kwargs):
      unobserved_vars = collections.OrderedDict()

      def interceptor(f, *args, **kwargs):
        name = kwargs.get("name")
        rv = f(*args, **kwargs)
        if name not in self.observed:
          unobserved_vars[name] = rv.shape
        return rv

      with ed.interception(interceptor):
        self.__call__()
      return unobserved_vars

    return unobserved_fn

#   def observe(self, states):
#     for name, value in iteritems(states):
#       setattr(self, name, value)

  def target_log_prob_fn(self, *args, **kwargs):
    """Unnormalized target density as a function of unobserved states."""

    def log_joint_fn(*args, **kwargs):
      states = dict(zip(self.unobserved, args))
      states.update(self.observed)
      log_probs = []

      def interceptor(f, *args, **kwargs):
        name = kwargs.get("name")
        for name, value in iteritems(states):
          if kwargs.get("name") == name:
            kwargs["value"] = value
        rv = f(*args, **kwargs)
        log_prob = tf.reduce_sum(rv.distribution.log_prob(rv.value))
        log_probs.append(log_prob)
        return rv

      with ed.interception(interceptor):
        self.__call__()
      log_prob = sum(log_probs)
      return log_prob

    return log_joint_fn

  def get_posterior_fn(self, states={}, *args, **kwargs):
    """Get the log joint prob given arbitrary values for vars"""

    def posterior_fn(*args, **kwargs):

      def interceptor(f, *args, **kwargs):
        name = kwargs.get("name")
        for name, value in iteritems(states):
          if kwargs.get("name") == name:
            kwargs["value"] = value
        rv = f(*args, **kwargs)
        return rv

      with ed.interception(interceptor):
        return self.__call__()

    return posterior_fn

  def __call__(self):
    return self.call()

  def call(self, *args, **kwargs):
    raise NotImplementedError


# This is a really quick / hacky sample function.
# Ideally the user could choose the kernel or inference method
# e.g., I could imagine a user defining a variational approximation in the model
# and then using VI as a sample option here where the sample method looks for
# model.q()
# Also, it's relatively straightforward to see how one could return arbitrary
# diagnostics given the model.
# Todo: Add diagnostics, multiple chains, more automatic inference.
def sample(model,
           num_results=5000,
           num_burnin_steps=3000,
           step_size=.4,
           num_leapfrog_steps=3,
           numpy=True):
  initial_state = []
  for name, shape in iteritems(model.unobserved):
    initial_state.append(.5 * tf.ones(shape, name="init_{}".format(name)))

  states, kernel_results = tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=initial_state,
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=model.target_log_prob_fn(),
          step_size=step_size,
          num_leapfrog_steps=num_leapfrog_steps))

  if numpy:
    with tf.Session() as sess:
      states, is_accepted_ = sess.run([states, kernel_results.is_accepted])
      accepted = np.sum(is_accepted_)
      print("Acceptance rate: {}".format(accepted / num_results))
  return dict(zip(model.unobserved.keys(), states))
