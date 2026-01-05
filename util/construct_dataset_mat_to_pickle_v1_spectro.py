import scipy.io as io
import h5py
import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(description='Specify task name for converting ZuCo v1.0 Mat file to Pickle')
parser.add_argument('-t', '--task_name', help='name of the task in /dataset/ZuCo, choose from {task1-SR,task2-NR,task3-TSR}', required=True)
args = vars(parser.parse_args())


"""config"""
version = 'v1' # 'old'

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
mat_files = glob(os.path.join(input_mat_files_dir,'*.mat'))
mat_files = sorted(mat_files)

if len(mat_files) == 0:
    print(f'No mat files found for {task_name}')
    quit()

dataset_dict = {}
max_len = -1
for mat_file in tqdm(mat_files):
    subject_name = os.path.basename(mat_file).split('_')[0].replace('results','').strip()
    dataset_dict[subject_name] = []

    matdata = io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)['sentenceData']

    for sent in matdata:
        sent_data = sent.rawData
        word_data = sent.word
        if not isinstance(sent_data, float):
            # sentence level:
            sent_obj = {'content':sent.content}
            #spectro sentence level data
            sent_obj['sentence_level_EEG'] = {'rawData':sent.rawData}
            print(sent.rawData.shape)
            #print(sent.rawData.shape[1])
            if sent.rawData.shape[1] > max_len:
                max_len = sent.rawData.shape[1]
            if sent.rawData.shape[1] > 23631:
                print(f'too long sent: subj:{subject_name} content:{sent.content}, return None')
                dataset_dict[subject_name].append(None)
                continue
            dataset_dict[subject_name].append(sent_obj)

        else:
            print(f'missing sent: subj:{subject_name} content:{sent.content}, return None')
            dataset_dict[subject_name].append(None)

            continue
    # print(dataset_dict.keys())
    # print(dataset_dict[subject_name][0].keys())
    # print(dataset_dict[subject_name][0]['content'])
    # print(dataset_dict[subject_name][0]['word'][0].keys())
    # print(dataset_dict[subject_name][0]['word'][0]['word_level_EEG']['FFD'])

"""output"""
output_name = f'{task_name}-dataset-spectro.pickle'
# with open(os.path.join(output_dir,'task1-SR-dataset.json'), 'w') as out:
#     json.dump(dataset_dict,out,indent = 4)

with open(os.path.join(output_dir,output_name), 'wb') as handle:
    pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('write to:', os.path.join(output_dir,output_name))


"""sanity check"""
# check dataset
with open(os.path.join(output_dir,output_name), 'rb') as handle:
    whole_dataset = pickle.load(handle)
print('subjects:', whole_dataset.keys())

print('num of sent:', len(whole_dataset['ZAB']))
print()
print('max length:', max_len)

