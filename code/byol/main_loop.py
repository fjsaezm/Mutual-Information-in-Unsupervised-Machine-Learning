# Copyright 2020 DeepMind Technologies Limited.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation loops for an experiment."""
import sys,os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import time
from typing import Any, Mapping, Text, Type, Union

from absl import app
from absl import flags
from absl import logging
import jax
import numpy as np
import tensorflow as tf

from byol import byol_experiment
from byol import eval_experiment
from byol.configs import byol as byol_config
from byol.configs import eval as eval_config

flags.DEFINE_string('experiment_mode',
                    'pretrain', 'The experiment, pretrain or linear-eval')
flags.DEFINE_string('worker_mode', 'train', 'The mode, train or eval')
flags.DEFINE_string('worker_tpu_driver', '', 'The tpu driver to use')
flags.DEFINE_integer('pretrain_epochs', 1000, 'Number of pre-training epochs')
flags.DEFINE_integer('batch_size', 4096, 'Total batch size')
flags.DEFINE_string('checkpoint_root', '/tmp/byol',
                    'The directory to save checkpoints to.')
flags.DEFINE_integer('log_tensors_interval', 60, 'Log tensors every n seconds.')

FLAGS = flags.FLAGS


Experiment = Union[
    Type[byol_experiment.ByolExperiment],
    Type[eval_experiment.EvalExperiment]]


def write_tensorboard_epoch(writer,scalars,step):
    print(scalars)
    with writer.as_default():
        tf.summary.scalar('train/top_1_acc',float(scalars['top1_accuracy']),step=step)
        tf.summary.scalar('train/top_5_acc',float(scalars['top5_accuracy']),step=step)
        tf.summary.scalar('learning_rate',float(scalars['learning_rate']),step=step)
        tf.summary.scalar('train/supervised_loss',float(scalars['classif_loss']),step=step)
        tf.summary.scalar('train/repr_loss',float(scalars['repr_loss']),step=step)
        tf.summary.scalar('train/total_loss',float(scalars['loss']),step=step)

def write_tensorboard_end(writer,scalars,step):
     with writer.as_default():
        tf.summary.scalar('eval/label_top_1_accuracy',float(scalars['top1_accuracy']),step=step)
        tf.summary.scalar('eval/label_top_5_accuracy',float(scalars['top5_accuracy']),step=step)
        tf.summary.scalar('eval/regularization_loss',float(scalars['loss']),step=step)



def train_loop(experiment_class: Experiment, config: Mapping[Text, Any]):
  """The main training loop.

  This loop periodically saves a checkpoint to be evaluated in the eval_loop.

  Args:
    experiment_class: the constructor for the experiment (either byol_experiment
    or eval_experiment).
    config: the experiment config.
  """
  
  experiment = experiment_class(**config)
  rng = jax.random.PRNGKey(0)
  step = 0

  # Define logger for tensorboard
  train_summary_writer = tf.summary.create_file_writer(FLAGS.checkpoint_root)

  host_id = jax.host_id()
  last_logging = time.time()
  if config['checkpointing_config']['use_checkpointing']:
    checkpoint_data = experiment.load_checkpoint()
    if checkpoint_data is None:
      step = 0
    else:
      step, rng = checkpoint_data

  local_device_count = jax.local_device_count()
  while step < config['max_steps']:
    step_rng, rng = tuple(jax.random.split(rng))
    # Broadcast the random seeds across the devices
    step_rng_device = jax.random.split(step_rng, num=jax.device_count())
    step_rng_device = step_rng_device[
        host_id * local_device_count:(host_id + 1) * local_device_count]
    step_device = np.broadcast_to(step, [local_device_count])

    # Perform a training step and get scalars to log.
    scalars = experiment.step(global_step=step_device, rng=step_rng_device)

    # Checkpointing and logging.
    if config['checkpointing_config']['use_checkpointing']:
      experiment.save_checkpoint(step, rng)
      current_time = time.time()
      if current_time - last_logging > FLAGS.log_tensors_interval:
        logging.info('Step %d: %s', step, scalars)
        if FLAGS.experiment_mode == 'pretrain':
            write_tensorboard_epoch(train_summary_writer,scalars,step)

        last_logging = current_time
    step += 1
  logging.info('Saving final checkpoint')
  logging.info('Step %d: %s', step, scalars)
  if FLAGS.experiment_mode == 'pretrain':
      write_tensorboard_end(train_summary_writer,scalars,step)
  experiment.save_checkpoint(step, rng)


def eval_loop(experiment_class: Experiment, config: Mapping[Text, Any]):
  """The main evaluation loop.

  This loop periodically loads a checkpoint and evaluates its performance on the
  test set, by calling experiment.evaluate.

  Args:
    experiment_class: the constructor for the experiment (either byol_experiment
    or eval_experiment).
    config: the experiment config.
  """
  experiment = experiment_class(**config)
  last_evaluated_step = -1
  while True:
    checkpoint_data = experiment.load_checkpoint()
    if checkpoint_data is None:
      logging.info('No checkpoint found. Waiting for 10s.')
      time.sleep(10)
      continue
    step, _ = checkpoint_data
    if step <= last_evaluated_step:
      logging.info('Checkpoint at step %d already evaluated, waiting.', step)
      time.sleep(10)
      continue
    host_id = jax.host_id()
    local_device_count = jax.local_device_count()
    step_device = np.broadcast_to(step, [local_device_count])
    scalars = experiment.evaluate(global_step=step_device)
    if host_id == 0:  # Only perform logging in one host.
      logging.info('Evaluation at step %d: %s', step, scalars)
    last_evaluated_step = step
    if last_evaluated_step >= config['max_steps']:
      return


def main(_):
  if FLAGS.worker_tpu_driver:
    jax.config.update('jax_xla_backend', 'tpu_driver')
    jax.config.update('jax_backend_target', FLAGS.worker_tpu_driver)
    logging.info('Backend: %s %r', FLAGS.worker_tpu_driver, jax.devices())

  if FLAGS.experiment_mode == 'pretrain':
    experiment_class = byol_experiment.ByolExperiment
    config = byol_config.get_config(FLAGS.pretrain_epochs, FLAGS.batch_size)
  elif FLAGS.experiment_mode == 'linear-eval':
    experiment_class = eval_experiment.EvalExperiment
    config = eval_config.get_config(f'{FLAGS.checkpoint_root}/pretrain.pkl',
                                    FLAGS.batch_size)
  else:
    raise ValueError(f'Unknown experiment mode: {FLAGS.experiment_mode}')
  config['checkpointing_config']['checkpoint_dir'] = FLAGS.checkpoint_root

  if FLAGS.worker_mode == 'train':
    train_loop(experiment_class, config)
  elif FLAGS.worker_mode == 'eval':
    eval_loop(experiment_class, config)


if __name__ == '__main__':
  app.run(main)
