import struct
import datetime
import numpy as np

from google.oauth2 import service_account
from google.cloud import bigtable
from google.cloud.bigtable import row_filters


def cbt_load_table(gcp_project_id, cbt_instance_id, cbt_table_name, credentials):
    """ returns a bigtable object.

        gcp_project_id --  string (default none)
        cbt_instance_id -- string (default none)
        cbt_table_name -- string (default none)
        credentials -- json file path (default none)
    """
    print('-> Looking for the [{}] table.'.format(cbt_table_name))
    client = bigtable.Client(gcp_project_id, admin=True, credentials=credentials)
    instance = client.instance(cbt_instance_id)
    cbt_table = instance.table(cbt_table_name)
    if not cbt_table.exists():
        print("-> Table doesn't exist. Creating [{}] table...".format(cbt_table_name))
        max_versions_rule = bigtable.column_family.MaxVersionsGCRule(1)
        column_families = {'step': max_versions_rule, 'global': max_versions_rule}
        cbt_table.create(column_families=column_families)
        print('-> Table created. Give it ~10 seconds to initialize before loading data.')
        exit()
    else:
        print("-> Table found.")
    return cbt_table


def cbt_global_iterator(cbt_table):
    """ Fetches and sets global iterator from bigtable.

        cbt_table -- bigtable object (default none)
    """
    row_filter = row_filters.CellsColumnLimitFilter(1)
    gi_row = cbt_table.read_row('collection_global_iterator'.encode())
    if gi_row is not None:
        global_i = gi_row.cells['global']['i'.encode()][0].value
        global_i = struct.unpack('i', global_i)[0] + 1
    else:
        global_i = 0
    gi_row = cbt_table.row('collection_global_iterator'.encode())
    gi_row.set_cell(column_family_id='global',
                    column='i'.encode(),
                    value=struct.pack('i',global_i),
                    timestamp=datetime.datetime.utcnow())
    cbt_table.mutate_rows([gi_row])
    return global_i

def cbt_global_trajectory_buffer(cbt_table, local_traj_buff, global_traj_buff_size):
    row_filter = row_filters.CellsColumnLimitFilter(1)
    old_row = cbt_table.read_row('global_traj_buff'.encode())
    if old_row is not None:
        global_traj_buff = np.frombuffer(old_row.cells['global']['traj_buff'.encode()][0].value, dtype=np.int32)
        global_traj_buff = np.append(global_traj_buff, local_traj_buff)
        update_size = local_traj_buff.shape[0] - (global_traj_buff_size - global_traj_buff.shape[0])
        if update_size > 0: global_traj_buff = global_traj_buff[update_size:]
    else:
        global_traj_buff = local_traj_buff
    new_row = cbt_table.row('global_traj_buff'.encode())
    new_row.set_cell(column_family_id='global',
                 column='traj_buff'.encode(),
                 value=global_traj_buff.tobytes(),
                 timestamp=datetime.datetime.utcnow())
    cbt_table.mutate_rows([new_row])

def cbt_get_global_trajectory_buffer(cbt_table):
    row_filter = row_filters.CellsColumnLimitFilter(1)
    row = cbt_table.read_row('global_traj_buff'.encode())
    if row is not None:
        return np.flip(np.frombuffer(row.cells['global']['traj_buff'.encode()][0].value, dtype=np.int32), axis=0)
    else:
        print("Table is empty.")
        exit()

def cbt_read_trajectory(cbt_table, traj_i):
    """ Reads N(num_rows) number of rows from cbt_table, starting from the global iterator value.

        cbt_table -- bigtable object (default none)
        prefix -- string (default none)
        num_rows -- integer (default none)
        global_i -- integer (default none)
    """
    start_row_key = 'episode_{:05d}_step_{:05d}'.format(traj_i, 0).encode()
    end_row_key = 'episode_{:05d}_step_{:05d}'.format(traj_i+1, 0).encode()
    partial_rows = cbt_table.read_rows(start_row_key, end_row_key)
    return partial_rows