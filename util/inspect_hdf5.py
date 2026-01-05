"""Quick script to inspect HDF5 structure of MATLAB v7.3 files - v2"""
import h5py
import glob
import numpy as np

# Find the first mat file
files = glob.glob('./dataset/ZuCo/task2-NR-2.0/Matlab_files/*.mat')
print('Found files:', len(files))

if files:
    f = h5py.File(files[0], 'r')
    
    sd = f['sentenceData']
    print('\n=== sentenceData keys ===')
    all_keys = list(sd.keys())
    print('All keys:', all_keys)
    
    # Check content - this likely contains sentence text references
    print('\n=== content ===')
    content = sd['content']
    print('Type:', type(content))
    print('Shape:', content.shape if hasattr(content, 'shape') else 'N/A')
    
    # Try to read some values
    if hasattr(content, 'shape'):
        print('Dtype:', content.dtype)
        print('First 3 refs:', content[:3])
        
        # Dereference to get actual content
        for i in range(min(3, content.shape[0])):
            ref = content[i, 0] if content.ndim > 1 else content[i]
            try:
                dereferenced = f[ref]
                print(f'  Sentence {i}: type={type(dereferenced)}')
                if hasattr(dereferenced, 'shape'):
                    data = dereferenced[()]
                    print(f'    shape={data.shape}, dtype={data.dtype}')
                    if data.dtype == np.uint16:
                        text = ''.join(chr(c) for c in data.flatten())
                        print(f'    text: {text[:100]}...')
            except Exception as e:
                print(f'  Error accessing {i}: {e}')
    
    # Check for rawData
    print('\n=== Looking for rawData or EEG data ===')
    for key in ['rawData', 'mean_t1', 'mean_t2', 'word']:
        if key in sd:
            item = sd[key]
            print(f'{key}: type={type(item)}, shape={item.shape if hasattr(item, "shape") else "N/A"}')
    
    # Check what's in word if it exists
    if 'word' in sd:
        print('\n=== word ===')
        word = sd['word']
        print('Shape:', word.shape)
        
    f.close()
