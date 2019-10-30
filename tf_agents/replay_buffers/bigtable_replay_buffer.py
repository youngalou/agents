from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf

from google.oauth2 import service_account

from tf_agents.replay_buffers import replay_buffer
from tf_agents.replay_buffers import table
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from tf_agents.trajectories import trajectory

from tf_agents.protobuf import tf_agents_trajectory_pb2
from tf_agents.utils.gcp_io import cbt_load_table, cbt_global_iterator, cbt_global_trajectory_buffer, \
                                    cbt_read_trajectory, cbt_get_global_trajectory_buffer

#SET API CREDENTIALS
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

BufferInfo = collections.namedtuple('BufferInfo',
                                    ['ids', 'probabilities'])

class BigtableReplayBuffer(replay_buffer.ReplayBuffer):
  """A Python-based replay buffer that supports uniform sampling.
  Writing and reading to this replay buffer is thread safe.
  This replay buffer can be subclassed to change the encoding used for the
  underlying storage by overriding _encoded_data_spec, _encode, _decode, and
  _on_delete.
  """

  def __init__(self,
               data_spec,
               batch_size,
               max_length=1000,
               scope='TFUniformReplayBuffer',
               device='cpu:*',
               table_fn=table.Table,
               dataset_drop_remainder=False,
               dataset_window_shift=None,
               stateful_dataset=False,
               **kwargs):
    """Creates a PyUniformReplayBuffer.
    Args:
      data_spec: An ArraySpec or a list/tuple/nest of ArraySpecs describing a
        single item that can be stored in this buffer.
      capacity: The maximum number of items that can be stored in the buffer.
    """
    self._batch_size = batch_size
    self._max_length = max_length
    capacity = self._batch_size * self._max_length
    super(BigtableReplayBuffer, self).__init__(data_spec, capacity, stateful_dataset)

    self._id_spec = tensor_spec.TensorSpec([], dtype=tf.int64, name='id')
    self._capacity_value = np.int64(self._capacity)
    self._batch_offsets = (
        tf.range(self._batch_size, dtype=tf.int64) * self._max_length)
    self._scope = scope
    self._device = device
    self._table_fn = table_fn
    self._dataset_drop_remainder = dataset_drop_remainder
    self._dataset_window_shift = dataset_window_shift
    with tf.device(self._device), tf.compat.v1.variable_scope(self._scope):
      self._capacity = tf.constant(capacity, dtype=tf.int64)
      self._data_table = table_fn(self._data_spec, self._capacity_value)
      self._id_table = table_fn(self._id_spec, self._capacity_value)
      self._last_id = common.create_variable('last_id', -1)
      self._last_id_cs = tf.CriticalSection(name='last_id')

    super(BigtableReplayBuffer, self).__init__(data_spec, capacity)
    self.obs_shape = np.append(1, self.data_spec.observation.shape).astype(np.int32)

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    self.cbt_table = cbt_load_table(kwargs['gcp_project_id'], kwargs['cbt_instance_id'], kwargs['cbt_table_name'], credentials)
    max_row_bytes = (4*np.prod(self.obs_shape) + 64)
    self.cbt_batcher = self.cbt_table.mutations_batcher(flush_count=kwargs['num_episodes'], max_row_bytes=max_row_bytes)
    
    self.global_traj_buff_size = kwargs['global_traj_buff_size']
    self.env_type = kwargs['env_type']
    self.reset()

  def _add_batch(self, items):
    tf.nest.assert_same_structure(items, self._data_spec)
    with tf.device(self._device), tf.name_scope(self._scope):
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
      pb2_trajectory.observation = items.observation.numpy().astype(np.float32).flatten().tobytes()
      pb2_trajectory.action = items.action.numpy().astype(np.int32).tobytes()
      # pb2_trajectory.policy_info = items.policy_info
      pb2_trajectory.next_step_type = items.next_step_type.numpy().tobytes()
      pb2_trajectory.reward = items.reward.numpy().tobytes()
      pb2_trajectory.discount = items.discount.numpy().tobytes()
    else:
      pb2_trajectory.step_type = items.step_type.tobytes()
      pb2_trajectory.observation = items.observation.astype(np.float32).flatten().tobytes()
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
        item = self.item_from_trajectory(pb2_trajectory)
        return item

  def item_from_trajectory(self, pb2_trajectory):
    return trajectory.Trajectory(
      step_type=np.squeeze(np.frombuffer(pb2_trajectory.step_type, dtype=np.int32)),
      observation=np.squeeze(np.frombuffer(pb2_trajectory.observation, dtype=np.float32).reshape(self.obs_shape)),
      action=np.squeeze(np.frombuffer(pb2_trajectory.action, dtype=np.int32)),
      policy_info=(),#np.frombuffer(pb2_trajectory.policy_info, dtype=np.float32)
      next_step_type=np.squeeze(np.frombuffer(pb2_trajectory.next_step_type, dtype=np.int32)),
      reward=np.squeeze(np.frombuffer(pb2_trajectory.reward, dtype=np.float32)),
      discount=np.squeeze(np.frombuffer(pb2_trajectory.discount, dtype=np.float32))
    )

  def _get_next(self,
                sample_batch_size=None,
                num_steps=None,
                time_stacked=True):
    num_steps_value = num_steps if num_steps is not None else 1
    def get_single():
      """Gets a single item from the replay buffer."""
      if num_steps is not None and time_stacked:
        item = [self.load_item_from_bigtable() for _ in range(num_steps)]
        item = nest_utils.stack_nested_arrays(item)
      else:
        item = self.load_item_from_bigtable()

      buffer_info = BufferInfo(ids=0, probabilities=0)
      return item, buffer_info

    if sample_batch_size is None:
      return get_single()
    else:
      samples = [get_single() for _ in range(sample_batch_size)]
      return nest_utils.stack_nested_arrays(samples)

  def as_dataset(self,
                 sample_batch_size=None,
                 num_steps=None,
                 num_parallel_calls=None,
                 single_deterministic_pass=False):
    return super(BigtableReplayBuffer, self).as_dataset(
        sample_batch_size, num_steps, num_parallel_calls,
        single_deterministic_pass=single_deterministic_pass)

  def _as_dataset(self,
                  sample_batch_size=None,
                  num_steps=None,
                  num_parallel_calls=None):
    """Creates a dataset that returns entries from the buffer in shuffled order.
    Args:
      sample_batch_size: (Optional.) An optional batch_size to specify the
        number of items to return. See as_dataset() documentation.
      num_steps: (Optional.)  Optional way to specify that sub-episodes are
        desired. See as_dataset() documentation.
      num_parallel_calls: (Optional.) Number elements to process in parallel.
        See as_dataset() documentation.
    Returns:
      A dataset of type tf.data.Dataset, elements of which are 2-tuples of:
        - An item or sequence of items or batch thereof
        - Auxiliary info for the items (i.e. ids, probs).
    """
    def get_next(_):
      return self.get_next(sample_batch_size, num_steps, time_stacked=True)

    dataset = tf.data.experimental.Counter().map(
        get_next, num_parallel_calls=num_parallel_calls)
    return dataset