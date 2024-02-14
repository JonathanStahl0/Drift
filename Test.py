import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import os
import re


def load_mat_file(file, dff_type):
    mat = loadmat(file)
    mdata = mat['Data']
    mdtype = mdata.dtype
    ndata = {n: mdata[n][0, 0] for n in mdtype.names}
    subject_id = str(ndata['subject_id'][0, 0])
    session = str(ndata['session'][0, 0])
    df_behav = pd.DataFrame(ndata[dff_type])
    df_cell_x = pd.DataFrame(ndata['cell_pos_x'])
    df_cell_y = pd.DataFrame(ndata['cell_pos_y'])
    df_loc = pd.DataFrame(ndata['brain_area']).astype(str)
    df_data = df_behav
    for df in [df_cell_x, df_cell_y, df_loc]:
        df_data = pd.concat([df, df_data], axis=1, join='outer')
    # df_data.to_pickle('hi.pkl')
    num_columns = len(df_data.columns)
    new_column_names = ['Location', 'Cell_pos_x', 'Cell_pos_y'] + [i for i in range(num_columns - 3)]
    df_data.columns = new_column_names
    return df_data


if __name__ == '__main__':
    frame_rate = 2
    bins_in_seconds = [0.5, 5, 10, 20, 30, 60, 120, 180]
    file_names = ["464724_session_2"]#, "464725_session_5", "464725_session_8", "464725_session_2", "464724_session_5", "464724_session_7"]
    behaviour_types = ["dff_behav", "dff_spont"]
    
    # Calculte correlation for one data set
    for behaviour_type in behaviour_types:
        for file_name in file_names:
            df_data = load_mat_file(f"F:\Shared drives\FinkelsteinLab\File_Transfer\For_Jonathan\data_dff_subject{file_name}.mat", behaviour_type)
    