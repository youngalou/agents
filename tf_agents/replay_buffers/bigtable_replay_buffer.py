from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

import numpy as np
import tensorflow as tf
from tf_agents.replay_buffers import replay_buffer
from tf_agents.specs import array_spec
from tf_agents.utils import nest_utils
from tf_agents.trajectories import trajectory

from google.oauth2 import service_account

from tf_agents.protobuf import tf_agents_trajectory_pb2
from tf_agents.utils.gcp_io import cbt_load_table, cbt_global_iterator, cbt_global_trajectory_buffer, \
                                    cbt_read_trajectory, cbt_get_global_trajectory_buffer

#SET API CREDENTIALS
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

class BigtableReplayBuffer(replay_buffer.ReplayBuffer):
  """A Python-based replay buffer that supports uniform sampling.
  Writing and reading to this replay buffer is thread safe.
  This replay buffer can be subclassed to change the encoding used for the
  underlying storage by overriding _encoded_data_spec, _encode, _decode, and
  _on_delete.
  """

  def __init__(self, data_spec, capacity, **kwargs):
    """Creates a PyUniformReplayBuffer.
    Args:
      data_spec: An ArraySpec or a list/tuple/nest of ArraySpecs describing a
        single item that can be stored in this buffer.
      capacity: The maximum number of items that can be stored in the buffer.
    """
    super(BigtableReplayBuffer, self).__init__(data_spec, capacity)
    self.obs_shape = np.append(1, self.data_spec.observation.shape).astype(np.int32)

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    self.cbt_table = cbt_load_table(kwargs['gcp_project_id'], kwargs['cbt_instance_id'], kwargs['cbt_table_name'], credentials)
    max_row_bytes = (4*np.prod(data_spec.shape) + 64)
    self.cbt_batcher = self.cbt_table.mutations_batcher(flush_count=kwargs['num_episodes'], max_row_bytes=max_row_bytes)
    
    self.global_traj_buff_size = kwargs['global_traj_buff_size']
    self.env_type = kwargs['env_type']
    self.reset()

  def _add_batch(self, items):
    self.next_episode()
    self.bigtable_add_row(items)
    self.bigtable_write_rows()
  
  def next_episode(self):
    self.global_i = cbt_global_iterator(self.cbt_table)
    self.local_traj_buff.append(self.global_i)
    self.current_step = 0

  def bigtable_add_row(self, items):
    pb2_trajectory = tf_agents_trajectory_pb2.Trajectory()
    if self.env_type == 'tf_env':
      pb2_trajectory.step_type = items.step_type.numpy().tobytes()
      pb2_trajectory.observation = items.observation.numpy().flatten().tobytes()
      pb2_trajectory.action = items.action.numpy().astype(np.int32).tobytes()
      # pb2_trajectory.policy_info = items.policy_info
      pb2_trajectory.next_step_type = items.next_step_type.numpy().tobytes()
      pb2_trajectory.reward = items.reward.numpy().tobytes()
      pb2_trajectory.discount = items.discount.numpy().tobytes()
    else:
      pb2_trajectory.step_type = items.step_type.tobytes()
      pb2_trajectory.observation = items.observation.flatten().tobytes()
      pb2_trajectory.action = items.action.astype(np.int32).tobytes()
      # pb2_trajectory.policy_info = items.policy_info
      pb2_trajectory.next_step_type = items.next_step_type.tobytes()
      pb2_trajectory.reward = items.reward.tobytes()
      pb2_trajectory.discount = items.discount.tobytes()

    row_key = 'episode_{:05d}_step_{:05d}'.format(self.global_i, self.current_step).encode()
    row = self.cbt_table.row(row_key)
    row.set_cell(column_family_id='step',
                column='trajectory'.encode(),
                value=pb2_trajectory.SerializeToString())
    self.rows.append(row)

    self.current_step += 1

  def bigtable_write_rows(self):
    cbt_global_trajectory_buffer(self.cbt_table, np.asarray(self.local_traj_buff).astype(np.int32), self.global_traj_buff_size)
    self.cbt_batcher.mutate_rows(self.rows)
    self.cbt_batcher.flush()
    self.reset()
    
  def reset(self):
    self.local_traj_buff = []
    self.rows = []
    self.current_step = 0

  def load_item_from_bigtable(self):
    global_traj_buff = cbt_get_global_trajectory_buffer(self.cbt_table)
    for traj_i in global_traj_buff:
      item_rows = cbt_read_trajectory(self.cbt_table, traj_i)
      for row in item_rows:
        #DESERIALIZE DATA
        bytes_item = row.cells['step']['trajectory'.encode()][0].value
        pb2_trajectory = tf_agents_trajectory_pb2.Trajectory()
        pb2_trajectory.ParseFromString(bytes_item)
        item = self.item_from_trajectory()
        return item

  def item_from_trajectory(self, pb2_trajectory):
    item = trajectory.Trajectory()
    item.step_type = np.frombuffer(pb2_trajectory.step_type, dtype=np.int32)
    item.observation = np.frombuffer(pb2_trajectory.observation, dtype=np.uint8).reshape(self.obs_shape)
    item.action = np.frombuffer(pb2_trajectory.action, dtype=np.int32)
    # item.policy_info = np.frombuffer(pb2_trajectory.policy_info, dtype=np.int32)
    item.next_step_type = np.frombuffer(pb2_trajectory.next_step_type, dtype=np.int32)
    item.reward = np.frombuffer(pb2_trajectory.reward, dtype=np.float32)
    item.discount =np.frombuffer(pb2_trajectory.discount, dtype=np.float32)
    return item

  def _get_next(self,
                sample_batch_size=None,
                num_steps=None,
                time_stacked=True):
    num_steps_value = num_steps if num_steps is not None else 1
    def get_single():
      """Gets a single item from the replay buffer."""
      item = self.load_item_from_bigtable()

      if num_steps is not None and time_stacked:
        item = nest_utils.stack_nested_arrays(item)
      return item

    if sample_batch_size is None:
      return get_single()
    else:
      samples = [get_single() for _ in range(sample_batch_size)]
      return nest_utils.stack_nested_arrays(samples)

  def _as_dataset(self, sample_batch_size=None, num_steps=None,
                  num_parallel_calls=None):
    if num_parallel_calls is not None:
      raise NotImplementedError('PyUniformReplayBuffer does not support '
                                'num_parallel_calls (must be None).')

    data_spec = self._data_spec
    if sample_batch_size is not None:
      data_spec = array_spec.add_outer_dims_nest(
          data_spec, (sample_batch_size,))
    if num_steps is not None:
      data_spec = (data_spec,) * num_steps
    shapes = tuple(s.shape for s in tf.nest.flatten(data_spec))
    dtypes = tuple(s.dtype for s in tf.nest.flatten(data_spec))

    def generator_fn():
      while True:
        if sample_batch_size is not None:
          batch = [self._get_next(num_steps=num_steps, time_stacked=False)
                   for _ in range(sample_batch_size)]
          item = nest_utils.stack_nested_arrays(batch)
        else:
          item = self._get_next(num_steps=num_steps, time_stacked=False)
        yield tuple(tf.nest.flatten(item))

    def time_stack(*structures):
      time_axis = 0 if sample_batch_size is None else 1
      return tf.nest.map_structure(
          lambda *elements: tf.stack(elements, axis=time_axis), *structures)

    ds = tf.data.Dataset.from_generator(
        generator_fn, dtypes,
        shapes).map(lambda *items: tf.nest.pack_sequence_as(data_spec, items))
    if num_steps is not None:
      return ds.map(time_stack)
    else:
      return ds