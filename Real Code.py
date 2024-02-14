import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import os


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

def calculate_correlation_for_all_bins(df_data, bin_in_seconds, file_name, behaviour_type, location):
    bins_as_indexs = calculate_indexes_from_seconds(bin_in_seconds, frame_rate)
    path = "C:\\Users\\Jonathan Stahl\\Drift\\"
    i = 0
    for bin_index in bins_as_indexs:
        full_name = file_name + "_" + behaviour_type + "_" + location +"_bin_" + str(bin_in_seconds[i])
        correlation_matrix = get_correlation_matrix(df_data, bin_index, path,  file_name, behaviour_type, location, full_name)
        create_heatmap(correlation_matrix, path, file_name, behaviour_type, location, full_name)
        diagonals_correlation, diagonals_std = calculate_correlation_for_delta_t(correlation_matrix)
        #diagonals_correlation = diagonals_correlation[30:-30]
        #diagonals_std = diagonals_std[30:-30]
        create_Line_plot([bin_in_seconds[i]*(x + 1) for x in range(len(diagonals_correlation))], diagonals_correlation, diagonals_std, path, file_name, behaviour_type, location, full_name)
        i += 1

    

def calculate_correlation_for_delta_t(correlation_matrix):
    diagonals_correlation = []
    diagonals_std = []
    for i in range(len(correlation_matrix)-1):
        d = correlation_matrix.diagonal(i+1)
        diagonals_correlation.append(np.mean(d))
        diagonals_std.append(np.std(d))
    return diagonals_correlation, diagonals_std


def get_correlation_matrix(df_data, bin_index, path, file_name, behaviour_type, location, full_name):
    directory = path + f"Data\\Correlation numpy matrices\\{file_name}\\{behaviour_type}\\{location}"
    full_path = directory + "\\" + full_name
    try:
        correlation_matrix = np.load(full_path + ".npy") 
    except:
        mean_df = pd.DataFrame()
        for i in range(3, len(df_data.columns), bin_index):
            mean_df[f'Mean_{i - 3}-{i + bin_index - 4}'] = df_data.iloc[:, i : i + bin_index].mean(axis = 1)
        correlation_matrix = mean_df.corr().to_numpy()
        os.makedirs(directory, exist_ok=True)
        np.save(full_path, correlation_matrix)
    return correlation_matrix


def create_heatmap(matrix, path, file_name, behaviour_type, location, full_name):
    directory = path + f"Figures\\Correlation matrices heatmaps\\{file_name}\\{behaviour_type}\\{location}"
    full_path = directory + "\\" + full_name + "_heatmap.png"
    make_plot(None, None, "P.V.", "P.V.", full_name, location, None, directory, full_path, True, matrix)

def create_Line_plot(delta_t, diagonals_correlation, diagonals_std, path, file_name, behaviour_type, location, full_name):
    directory = path + f"Figures\\Correlation line plot\\{file_name}\\{behaviour_type}\\{location}"
    full_path = directory + "\\" + full_name  + "_line_plot.png"
    make_plot(delta_t, diagonals_correlation, 'delta_t', 'P.V. correlation', full_name, location, diagonals_std, directory, full_path, False, None)


def make_plot(x_axis, y_axis, xlabel, ylabel, title, label, std, directory, full_path, heatmap, matrix):
    plt.figure()
    if heatmap:
        plt.imshow(matrix, cmap='hot', interpolation='nearest')
    else:
        plt.plot(x_axis, y_axis, label=label)
        plt.fill_between(x_axis, np.array(y_axis) - np.array(std), np.array(y_axis) + np.array(std), color='gray', alpha=0.3, label='Std Dev')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(full_path)
    plt.close()


def calculate_indexes_from_seconds(delta_t_as_seconds, frame_rate):
    delta_t_as_indexs = []
    index_in_seconds = 1/frame_rate
    for x in delta_t_as_seconds:
        delta_t_as_indexs.append(int(x/index_in_seconds))
    return delta_t_as_indexs


if __name__ == '__main__':

    # Insert data
    frame_rate = 2
    bins_in_seconds = [0.5, 5, 10, 20, 30, 60, 120, 180]
    file_names = ["464724_session_2"]#, "464725_session_5", "464725_session_8", "464725_session_2", "464724_session_5", "464724_session_7"]
    behaviour_types = ["dff_behav", "dff_spont"]
    
    # Calculte correlation for one data set
    for behaviour_type in behaviour_types:
        for file_name in file_names:
            df_data = load_mat_file(f"F:\Shared drives\FinkelsteinLab\File_Transfer\For_Jonathan\data_dff_subject{file_name}.mat", behaviour_type)
            #calculate_correlation_for_all_bins(df_data, bins_in_seconds, file_name, behaviour_type, "All")
            locations = df_data['Location'].unique()
            for location in locations:
                new_df = df_data[df_data["Location"] == location]
                calculate_correlation_for_all_bins(new_df, bins_in_seconds, file_name, behaviour_type, location)
    
   















# def calculate_correlation(df_data, delta_t_as_index):
#     correlation_coefficient_vector = []
#     for x in range(len(df_data.columns) - (3 + delta_t_as_index)):
#         correlation_coefficient = np.ma.corrcoef(df_data[x].values, df_data[x + delta_t_as_index].values)[0, 1]
#         correlation_coefficient_vector.append(correlation_coefficient)
#     return np.mean(correlation_coefficient_vector), np.std(correlation_coefficient_vector)
            

#correlation_matrix.values[np.triu_indices(len(mean_df.columns), k=1)]