import scipy.io as io
import h5py
import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np
import pickle
import argparse
import data_loading_helpers_modified as dh

parser = argparse.ArgumentParser(description='Specify task name for converting ZuCo v1.0 Mat file to Pickle')
parser.add_argument('-t', '--task_name',
                    help='name of the task in /dataset/ZuCo, choose from {task1-SR,task2-NR,task3-TSR}', required=True)
args = vars(parser.parse_args())

"""config"""
version = 'v2' # 'new'

task_name = args['task_name']
# task_name = 'task1-SR'
# task_name = 'task2-NR'
# task_name = 'task3-TSR'


print('##############################')
print(f'start processing ZuCo {task_name}...')

input_mat_files_dir = f'./dataset/ZuCo/{task_name}/Matlab_files'

output_dir = f'./dataset/ZuCo/{task_name}/pickle'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

"""load files"""
mat_files = glob(os.path.join(input_mat_files_dir, '*.mat'))
mat_files = sorted(mat_files)

if len(mat_files) == 0:
    print(f'No mat files found for {task_name}')
    quit()

dataset_dict = {}
max_len = -1
for mat_file in tqdm(mat_files):
    subject_name = os.path.basename(mat_file).split('_')[0].replace('results', '').strip()
    dataset_dict[subject_name] = []
    subject = mat_file.split("ts")[1].split("_")[0]


    f = h5py.File(mat_file, 'r')
    sentence_data = f['sentenceData']
    rawData = sentence_data['rawData']
    contentData = sentence_data['content']
    wordData = sentence_data['word']
    print(rawData)
    for idx in range(len(rawData)):
        # get sentence string
        obj_reference_content = contentData[idx][0]
        sent_string = dh.load_matlab_string(f[obj_reference_content])
        obj_reference_rawData = rawData[idx][0]
        t_array = f[obj_reference_rawData][:].T.astype(np.float32)
        print(t_array.shape)
        if t_array.shape[1] == 1:
            print(f'missing sent: subj:{subject} content:{sent_string}, append None')
            dataset_dict[subject].append(None)
            continue
        if t_array.shape[1] > max_len:
            max_len = t_array.shape[1]
        if t_array.shape[1] > 23631:
            print(f'too long sent: subj:{subject_name} content:{sent_string}, return None')
            dataset_dict[subject_name].append(None)
            continue
        sent_obj = {'content': sent_string}
        sent_obj['sentence_level_EEG'] = {'rawData': t_array}
        dataset_dict[subject_name].append(sent_obj)

print(max_len)
"""output"""
output_name = f'{task_name}-dataset-spectro.pickle'
# with open(os.path.join(output_dir,'task1-SR-dataset.json'), 'w') as out:
#     json.dump(dataset_dict,out,indent = 4)

with open(os.path.join(output_dir, output_name), 'wb') as handle:
    pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('write to:', os.path.join(output_dir, output_name))

"""sanity check"""
# check dataset
with open(os.path.join(output_dir, output_name), 'rb') as handle:
    whole_dataset = pickle.load(handle)
print('subjects:', whole_dataset.keys())

print('num of sent:', len(whole_dataset['YAC']))
print()
print('max length:', max_len)


