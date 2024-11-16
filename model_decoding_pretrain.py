import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import math
import numpy as np
from einops.layers.torch import Rearrange, Reduce

""" main architecture for open vocabulary EEG-To-Text decoding"""
class BrainTranslator(nn.Module):
    def __init__(self,in_feature = 40, decoder_embedding_size = 1024, additional_encoder_nhead=5, additional_encoder_dim_feedforward = 2048):
        super(BrainTranslator, self).__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 26), (1, 1)),
            nn.Conv2d(40, 40, (105, 1), (1, 1)),#105 channel
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 25)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, 40, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        # before [batch 1,105, 24000]
        # torch.Size([batch, 40,1,1594])
        #torch.Size([batch, 1594, 40])
        # change avgpool into (1,25), get 957,40
        # additional transformer encoder, following BART paper about 
        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,  dim_feedforward = additional_encoder_dim_feedforward, batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
        dropout_prob = 0.5
        self.dropout = nn.Dropout(dropout_prob)
        self.batch_norm = nn.BatchNorm1d(957)
        # print('[INFO]adding positional embedding')
        # self.positional_embedding = PositionalEncoding(in_feature)

        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)
        #[batch, 957, 1024]
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, input_embeddings_batch):
        """input_embeddings_batch: batch_size*62*128"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""

        # input_embeddings_batch = self.positional_embedding(input_embeddings_batch)
        # use src_key_padding_masks
        #batch, 24000,105
        encoded_embedding = input_embeddings_batch.unsqueeze(1)
        #batch, 1, 24000,105
        encoded_embedding = encoded_embedding.permute(0, 1, 3, 2)
        # batch, 1, 105,24000
        encoded_embedding = self.shallownet(encoded_embedding)
        encoded_embedding = self.projection(encoded_embedding)
        encoded_embedding = self.additional_encoder(encoded_embedding)

        # encoded_embedding = self.additional_encoder(input_embeddings_batch)
        #encoded_embedding = self.dropout(encoded_embedding)
        # Permute dimensions to [batch_size, in_feature, sequence_length]
        #permuted_embedding = encoded_embedding.permute(0, 2, 1)

        # Applying Batch Normalization
        encoded_embedding = self.batch_norm(encoded_embedding)

        # Permuting back to the original shape [batch_size, sequence_length, in_feature]
        #encoded_embedding = normalized_embedding.permute(0, 2, 1)

        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        #print(encoded_embedding.size())
        #print(target_ids_batch_converted.size())
        #print(target_ids_batch_converted)
        
        return encoded_embedding


class BrainTranslator_reverse(nn.Module):
    def __init__(self, in_feature=40, decoder_embedding_size=1024):
        super(BrainTranslator_reverse, self).__init__()
        self.projection_reverse = nn.Sequential(
            # You might need some adjustment here to match spatial dimensions
            Rearrange('b h w e -> b e w h'),
            nn.ConvTranspose2d(40, 40, (1, 1), stride=(1, 1))
        )

        self.shallownet_reverse = nn.Sequential(
            nn.ConvTranspose2d(40, 40, (1, 75), stride=(1, 25)),
            nn.ConvTranspose2d(40, 40, (105, 1), stride=(1, 1)),
            nn.ConvTranspose2d(40, 1, (1, 26), stride=(1, 1))
        )
        self.fc1 = nn.Linear(decoder_embedding_size, in_feature)

        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, input_embeddings_batch):
        #batch, 957,1024
        #print(input_embeddings_batch.size())
        encoded_embedding = self.fc1(input_embeddings_batch)
        #batch, 957, 40
        encoded_embedding = encoded_embedding.unsqueeze(2)
        #batch, 957, 1 40
        encoded_embedding = self.projection_reverse(encoded_embedding)
        #batch 40, 1, 957

        encoded_embedding = self.shallownet_reverse(encoded_embedding)
        #torch.Size([5, 40, 1, 23975])
        #torch.Size([5, 40, 105, 23975])
        #torch.Size([5, 1, 105, 24000])
        encoded_embedding = encoded_embedding.permute(0, 1, 3, 2)
        encoded_embedding = encoded_embedding.squeeze(1)

        return encoded_embedding


class Reconstruction(nn.Module):
    def __init__(self, in_feature = 40, decoder_embedding_size = 1024, additional_encoder_nhead=5, additional_encoder_dim_feedforward = 2048):
        super(Reconstruction, self).__init__()
        self.brain_translator = BrainTranslator(in_feature = in_feature, decoder_embedding_size = decoder_embedding_size, additional_encoder_nhead=additional_encoder_nhead, additional_encoder_dim_feedforward = additional_encoder_dim_feedforward)
        self.brain_translator_reverse = BrainTranslator_reverse(in_feature=in_feature, decoder_embedding_size=decoder_embedding_size)

    def forward(self, x):
        # Define the forward pass for the combined model
        # For example, pass x through both models in sequence
        x1 = self.brain_translator(x)
        #print(x1.size())
        x2 = self.brain_translator_reverse(x1)
        return x2


if __name__ == '__main__':
    x = torch.rand((5, 24000, 105))
    model = Reconstruction(in_feature = 40, decoder_embedding_size = 1024, additional_encoder_nhead=5, additional_encoder_dim_feedforward = 2048)
    print(x.size())
    out = model(x)
    print(out.size())

    y = torch.rand([5, 24000, 105])
    k = 10
    out_sliced = out[:, :k, :]
    y_sliced = y[:, :k, :]
    mse_loss = nn.MSELoss()
    loss = mse_loss(out_sliced, y_sliced)
    print(loss.item())