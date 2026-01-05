import scipy.io as io
import h5py
import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np
import pickle
import argparse


def read_hdf5_string(h5file, ref):
    """Read a string from HDF5 reference (MATLAB stores as uint16 array)."""
    data = h5file[ref][()]
    if data.dtype == np.uint16:
        return ''.join(chr(c) for c in data.flatten())
    return str(data)


def read_hdf5_array(h5file, ref):
    """Read an array from HDF5 reference."""
    data = h5file[ref][()]
    # HDF5 stores data in column-major order, transpose for row-major
    if isinstance(data, np.ndarray) and data.ndim == 2:
        return data.T
    return data


def is_valid_ref(ref):
    """Check if ref is a valid HDF5 object reference."""
    return isinstance(ref, h5py.Reference)


parser = argparse.ArgumentParser(description='Specify task name for converting ZuCo Mat file to Pickle (supports v7.3 HDF5 format)')
parser.add_argument('-t', '--task_name', help='name of the task in /dataset/ZuCo, choose from {task1-SR,task2-NR,task3-TSR,task2-NR-2.0}', required=True)
args = vars(parser.parse_args())


"""config"""
version = 'v1' # 'old'

task_name = args['task_name']


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

    # Use h5py for MATLAB v7.3 files (HDF5 format)
    with h5py.File(mat_file, 'r') as h5file:
        sd = h5file['sentenceData']
        
        # ZuCo 2.0 structure: sentenceData is a Group with parallel arrays
        # content, rawData, word are each (N, 1) arrays of HDF5 references
        content_refs = sd['content'][()]
        rawdata_refs = sd['rawData'][()]
        word_refs = sd['word'][()]
        
        num_sentences = content_refs.shape[0]
        
        for i in range(num_sentences):
            # Get references for this sentence (flatten from (N,1) to just the ref)
            content_ref = content_refs[i, 0] if content_refs.ndim > 1 else content_refs[i]
            rawdata_ref = rawdata_refs[i, 0] if rawdata_refs.ndim > 1 else rawdata_refs[i]
            word_ref = word_refs[i, 0] if word_refs.ndim > 1 else word_refs[i]
            
            # Read content (sentence text)
            try:
                content = read_hdf5_string(h5file, content_ref)
            except Exception as e:
                print(f'Error reading content for sentence {i}: {e}')
                dataset_dict[subject_name].append(None)
                continue
            
            # Check word data - skip if not a valid reference
            if not is_valid_ref(word_ref):
                print(f'skipping sent (no word data): subj:{subject_name} content:{content[:50]}...')
                dataset_dict[subject_name].append(None)
                continue
            
            # Read rawData
            try:
                raw_data = read_hdf5_array(h5file, rawdata_ref)
            except Exception as e:
                print(f'missing rawData: subj:{subject_name} content:{content[:50]}..., error: {e}')
                dataset_dict[subject_name].append(None)
                continue
            
            # Check if rawData is valid (not NaN/empty)
            if isinstance(raw_data, float) or (isinstance(raw_data, np.ndarray) and raw_data.size == 0):
                print(f'missing sent: subj:{subject_name} content:{content[:50]}..., return None')
                dataset_dict[subject_name].append(None)
                continue
            
            # sentence level object
            sent_obj = {'content': content}
            sent_obj['sentence_level_EEG'] = {'rawData': raw_data}
            
            print(raw_data.shape)
            
            if raw_data.shape[1] > max_len:
                max_len = raw_data.shape[1]
            
            if raw_data.shape[1] > 23631:
                print(f'too long sent: subj:{subject_name} content:{content[:50]}..., return None')
                dataset_dict[subject_name].append(None)
                continue
            
            dataset_dict[subject_name].append(sent_obj)

"""output"""
output_name = f'{task_name}-dataset-spectro.pickle'

with open(os.path.join(output_dir,output_name), 'wb') as handle:
    pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('write to:', os.path.join(output_dir,output_name))


"""sanity check"""
# check dataset
with open(os.path.join(output_dir,output_name), 'rb') as handle:
    whole_dataset = pickle.load(handle)
print('subjects:', whole_dataset.keys())

# Get first subject name for sanity check
first_subject = list(whole_dataset.keys())[0]
print(f'num of sent for {first_subject}:', len(whole_dataset[first_subject]))
print()
print('max length:', max_len)
