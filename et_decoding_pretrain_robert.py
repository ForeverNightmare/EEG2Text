import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm
from transformers import BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BartConfig, \
    BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, \
    RobertaForSequenceClassification

from data_spectro import ZuCo_dataset
from model_decoding_pretrain import BrainTranslator, BrainTranslator_reverse, Reconstruction
from config import get_config
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import Rouge

trainlosslist = []

devlosslist = []

def normalize_2d(input_tensor):
    mean = torch.mean(input_tensor, dim=1, keepdim=True)
    std = torch.std(input_tensor, dim=1, keepdim=True)

    # Avoid division by zero
    std[std == 0] = 1

    normalized_tensor = (input_tensor - mean) / std
    return normalized_tensor



def bert_mlm_mask(arr, total_elements, mask_percentage=0.15):
    """
    Mask elements of a 2D float array using BERT's MLM strategy.

    Parameters:
    - arr: 2D numpy array of floats.
    - mask_percentage: Percentage of elements to mask.

    Returns:
    - Masked array.
    - Indices of masked elements.
    """
    # Find total number of elements in the first dimension (K)

    # Calculate number of elements to mask
    num_mask = int(total_elements * mask_percentage)

    # Randomly select elements to mask
    mask_indices = np.random.choice(total_elements, num_mask, replace=False)

    # Make a copy of the array to avoid modifying the original
    masked_arr = arr.clone()
    masked_arr = masked_arr.numpy()
    # Get number of elements for each masking strategy
    num_mask_80 = int(0.8 * num_mask)
    num_mask_10 = int(0.1 * num_mask)
    # Remaining will be for the unchanged strategy

    # 80% replaced by mask_value (using -9999.0 as a [MASK] representation)
    masked_arr[mask_indices[:num_mask_80], :] = 0.0

    # 10% replaced with random float value (assuming float values between 0 and 10 for simplicity)
    for idx in mask_indices[num_mask_80:num_mask_80 + num_mask_10]:
        random_value = np.random.uniform(-10.0, 10.0, arr.shape[1])
        masked_arr[idx, :] = random_value

    # 10% remain unchanged (so we do nothing)
    pad_length = 24000 - masked_arr.shape[0]
    masked_arr = np.pad(masked_arr, ((0, pad_length), (0, 0)), mode='constant', constant_values=0.0)
    #return masked_arr, mask_indices
    assert masked_arr.shape[0] == 24000
    assert masked_arr.shape[1] == 105
    return_tensor = torch.from_numpy(masked_arr)

    pad_length = 24000 - mask_indices.shape[0]
    mask_indices = np.pad(mask_indices, (0, pad_length), mode='constant', constant_values=-1)
    mask_indices = torch.from_numpy(mask_indices)
    return normalize_2d(return_tensor), mask_indices




def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25,
                checkpoint_path_best='./checkpoints/decoding/best/temp_decoding.pt',
                checkpoint_path_last='./checkpoints/decoding/last/temp_decoding.pt'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000

    intersection = False
    printornot = True
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for seq_len, target_ids, sent_mask, sent_mask_invert, sent_eeg_len, sent_level_EEG in tqdm(dataloaders[phase]):

                # load in batch
                sent_level_EEG_batch = sent_level_EEG.to(device)
                # print(sent_level_EEG_batch)
                sent_level_EEG_batch = sent_level_EEG_batch.float()

                this_batch = sent_eeg_len.shape[0]

                masked_EEG = []
                mask_indices = []
                #print(sent_eeg_len)
                for i in range(this_batch):
                    t_masked_EEG, t_mask_indices = bert_mlm_mask(sent_level_EEG[i], sent_eeg_len[i], mask_percentage=0.15)
                    masked_EEG.append(t_masked_EEG.unsqueeze(0))
                    mask_indices.append(t_mask_indices.unsqueeze(0))

                masked_EEG = torch.cat(masked_EEG, dim=0)
                mask_indices = torch.cat(mask_indices, dim=0)

                #print(masked_EEG.shape)
                #print(mask_indices.shape)
                #print(mask_indices)
                masked_EEG_batch = masked_EEG.to(device)
                masked_EEG_batch = masked_EEG_batch.float()
                mask_indices_batch = mask_indices.to(device)


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                ReconstructionOutput = model(masked_EEG_batch)

                """calculate loss"""
                # logits = seq2seqLMoutput.logits # 8*48*50265
                # logits = logits.permute(0,2,1) # 8*50265*48

                # loss = criterion(logits, target_ids_batch_label) # calculate cross entropy loss only on encoded target parts
                # NOTE: my criterion not used


                #loss = seq2seqLMoutput.loss  # use the BART language modeling loss
                mse_loss = nn.MSELoss()

                losses = []
                for i in range(sent_level_EEG_batch.size(0)):
                    if not printornot:
                        print(i)
                        print(mask_indices_batch[i][mask_indices_batch[i] >= 0])
                        print(len(mask_indices_batch[i][mask_indices_batch[i] >= 0]))
                        print(ReconstructionOutput.size())
                        print(sent_level_EEG.size())
                        k = mask_indices_batch[i][mask_indices_batch[i] >= 0][0]
                        print(sent_level_EEG[i][k])
                        print(masked_EEG[i][k])

                    valid_indices = mask_indices_batch[i][mask_indices_batch[i] >= 0].to(device)  # Filter out placeholder indices
                    tensor1_sliced = ReconstructionOutput[i][valid_indices].to(device)
                    tensor2_sliced = sent_level_EEG[i][valid_indices].to(device)

                    if not printornot:
                        print(tensor1_sliced.size())
                        print(tensor1_sliced.size())

                    # Compute the MSE loss for this batch
                    batch_loss = mse_loss(tensor1_sliced, tensor2_sliced)
                    losses.append(batch_loss)

                printornot = True

                # Average the losses
                loss = torch.stack(losses).mean()
                #print(average_loss.item())

                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * sent_level_EEG_batch.size()[0]  # batch loss

            if phase == 'train':
                scheduler.step()
                current_lr = optimizer_step2.param_groups[0]['lr']
                print(f"Current Learning Rate: {current_lr}")

            epoch_loss = running_loss / dataset_sizes[phase]

            # print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'train':
                trainlosslist.append(epoch_loss)
                print(trainlosslist)
            elif phase == 'dev':
                devlosslist.append(epoch_loss)
                print(devlosslist)

            if not intersection:
                if len(devlosslist) > 1:
                    if devlosslist[-2] > trainlosslist[-2] and devlosslist[-1] < trainlosslist[-1]:
                        intersection_path = checkpoint_path_best.replace(".pt", "-intersection.pt")
                        braintranslator_path = intersection_path.replace(".pt", "-braintranslator.pt")
                        torch.save(model.state_dict(), intersection_path)
                        torch.save(model.brain_translator.state_dict(), braintranslator_path)
                        print(f'update intersection checkpoint: {intersection_path}')
                        print("now epoch")
                        print(len(devlosslist))
                        print("dev loss")
                        print(devlosslist[-2])
                        print(devlosslist[-1])
                        print("train loss")
                        print(trainlosslist[-2])
                        print(trainlosslist[-1])
                        intersection = True

            # deep copy the model
            if phase == 'dev' and epoch_loss < best_loss:
                braintranslator_path = checkpoint_path_best.replace(".pt", "-braintranslator.pt")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                '''save checkpoint'''
                torch.save(model.state_dict(), checkpoint_path_best)
                torch.save(model.brain_translator.state_dict(), braintranslator_path)
                print(f'update best on dev checkpoint: {checkpoint_path_best}')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    torch.save(model.state_dict(), checkpoint_path_last)
    braintranslator_path = checkpoint_path_last.replace(".pt", "-braintranslator.pt")
    torch.save(model.brain_translator.state_dict(), braintranslator_path)
    print(f'update last checkpoint: {checkpoint_path_last}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)


if __name__ == '__main__':
    args = get_config('train_decoding')

    ''' config param'''
    dataset_setting = 'unique_sent'

    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']

    batch_size = args['batch_size']

    model_name = args['model_name']
    # model_name = 'BrainTranslatorNaive' # with no additional transformers
    # model_name = 'BrainTranslator'

    # task_name = 'task1'
    # task_name = 'task1_task2'
    # task_name = 'task1_task2_task3'
    # task_name = 'task1_task2_taskNRv2'
    task_name = args['task_name']

    save_path = args['save_path']

    skip_step_one = args['skip_step_one']
    load_step1_checkpoint = args['load_step1_checkpoint']
    use_random_init = args['use_random_init']

    if use_random_init and skip_step_one:
        step2_lr = 5 * 1e-4

    print(f'[INFO]using model: {model_name}')

    if skip_step_one:
        save_name = f'{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}-pretrain_robert'
    else:
        save_name = f'{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}-pretrain_robert'

    if use_random_init:
        save_name = 'randinit_' + save_name

    output_checkpoint_name_best = save_path + f'/best/{save_name}.pt'
    output_checkpoint_name_last = save_path + f'/last/{save_name}.pt'

    # subject_choice = 'ALL
    subject_choice = args['subjects']
    print(f'![Debug]using {subject_choice}')
    # eeg_type_choice = 'GD
    eeg_type_choice = args['eeg_type']
    print(f'[INFO]eeg type {eeg_type_choice}')
    # bands_choice = ['_t1']
    # bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2']
    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():
        # dev = "cuda:3"
        dev = args['cuda']
    else:
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()

    '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset-spectro.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset-spectro.pickle'
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = './dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset-spectro.pickle'
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset-spectro.pickle'
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    print()

    # train dataset
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject=subject_choice, setting=dataset_setting)
    # dev dataset
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject=subject_choice, setting=dataset_setting)
    # test dataset
    # test_set = ZuCo_dataset(whole_dataset_dict, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice)
    '''

    """save config"""
    with open(f'./config/decoding/{save_name}.json', 'w') as out_config:
        json.dump(args, out_config, indent=4)

    if model_name in ['BrainTranslator', 'BrainTranslatorNaive']:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    elif model_name == 'BertGeneration':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        config = BertConfig.from_pretrained("bert-base-cased")
        config.is_decoder = True

    with open("train_set_masked_raw_robert.pkl", "rb") as f:
        train_set = pickle.load(f)
    with open("dev_set_masked_raw_robert.pkl", "rb") as f:
        dev_set = pickle.load(f)

    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))

    # train dataloader
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    # dev dataloader
    val_dataloader = DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=4)
    # dataloaders
    dataloaders = {'train': train_dataloader, 'dev': val_dataloader}
    del train_set
    del dev_set

    ''' set up model '''
    if model_name == 'BrainTranslator':
        if use_random_init:
            config = BartConfig.from_pretrained('facebook/bart-large')
            pretrained = BartForConditionalGeneration(config)
        else:
            pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

        model = Reconstruction(in_feature=40, decoder_embedding_size=1024, additional_encoder_nhead=5,
                                additional_encoder_dim_feedforward=2048)

    model.to(device)

    ''' training loop '''

    ######################################################
    '''step one trainig: freeze most of BART params'''
    ######################################################

    # closely follow BART paper
    if model_name in ['BrainTranslator', 'BrainTranslatorNaive']:
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                    continue
                else:
                    param.requires_grad = False
    elif model_name == 'BertGeneration':
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if ('embeddings' in name) or ('encoder.layer.0' in name):
                    continue
                else:
                    param.requires_grad = False

    if skip_step_one:
        if load_step1_checkpoint:
            stepone_checkpoint = 'path_to_step_1_checkpoint.pt'
            print(f'skip step one, load checkpoint: {stepone_checkpoint}')
            model.load_state_dict(torch.load(stepone_checkpoint))
        else:
            print('skip step one, start from scratch at step two')
    else:

        ''' set up optimizer and scheduler'''
        optimizer_step1 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step1_lr, momentum=0.9)

        exp_lr_scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=20, gamma=0.1)

        ''' set up loss function '''
        criterion = nn.CrossEntropyLoss()

        print('=== start Step1 training ... ===')
        # print training layers
        show_require_grad_layers(model)
        # return best loss model from step1 training
        model = train_model(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1,
                            num_epochs=num_epochs_step1, checkpoint_path_best=output_checkpoint_name_best,
                            checkpoint_path_last=output_checkpoint_name_last)

    ######################################################
    '''step two trainig: update whole model for a few iterations'''
    ######################################################
    for name, param in model.named_parameters():
        param.requires_grad = True

    ''' set up optimizer and scheduler'''
    print("using RMSprop, xavier_uniform_, batch norm")
    optimizer_step2 = optim.RMSprop(model.parameters(), lr=step2_lr)

    exp_lr_scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=10, gamma=0.1)

    ''' set up loss function '''
    criterion = nn.CrossEntropyLoss()

    print()
    print('=== start Step2 training ... ===')
    # print training layers
    # show_require_grad_layers(model)

    '''main loop'''
    trained_model = train_model(dataloaders, device, model, criterion, optimizer_step2, exp_lr_scheduler_step2,
                                num_epochs=num_epochs_step2, checkpoint_path_best=output_checkpoint_name_best,
                                checkpoint_path_last=output_checkpoint_name_last)

    print(trainlosslist)
    print(devlosslist)
    # '''save checkpoint'''
    # torch.save(trained_model.state_dict(), os.path.join(save_path,output_checkpoint_name))