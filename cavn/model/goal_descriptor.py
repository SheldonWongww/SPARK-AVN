# cavn/model/goal_descriptor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cavn.common.utils import MultiHeadAttentionLayer

def get_params(downsample=True):
    feature_label_resolution = 10

    # Temporal pooling size
    if downsample:
        t_pool_size = [int(feature_label_resolution / 5), 1, 1] # [2, 1, 1]
    else:
        t_pool_size = [feature_label_resolution, 1, 1] # [10, 1, 1]

    params = dict(
        unique_classes = 20, # number of sound event classes
        
        nb_cnn2d_filt = 64, # number of CNN2D filters
        f_pool_size = [4, 2, 2], # frequency pooling size
        t_pool_size = t_pool_size, # temporal pooling size
        dropout_rate = 0.05, # dropout rate
        
        nb_rnn_layers = 2, # number of GRU layers
        rnn_size = 256, # GRU hidden size
        
        self_attn = True, # whether to use self-attention
        nb_heads = 8, # number of attention heads
        
        nb_fnn_layers = 1, # number of FNN layers
        fnn_size = 128, # FNN hidden size

        fs = 16000,
        hop_len_s = 0.01,
    )

    return params

class ConvBlock(nn.Module):
    """A basic convolutional block: Conv2D + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu_(x)
        return x

class CRNN(nn.Module):
    """CNN + GRU + Self-Attention + FNN for audio feature extraction"""
    def __init__(self, in_feat_shape, out_dim, params) -> None:
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.conv_block_list = nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(
                    ConvBlock(
                        in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1],
                        out_channels=params['nb_cnn2d_filt']
                    )
                )
                self.conv_block_list.append(
                    torch.nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]))
                )
                self.conv_block_list.append(
                    torch.nn.Dropout2d(p=params['dropout_rate'])
                )

        if params['nb_rnn_layers']:
            self.in_gru_size = params['nb_cnn2d_filt'] * int( np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
            # input_size=256, hidden_size=256,num_layers=2, batch_first=True, dropout=0.05, bidirectional=True
            self.gru = torch.nn.GRU(input_size=self.in_gru_size, hidden_size=params['rnn_size'],
                                    num_layers=params['nb_rnn_layers'], batch_first=True,
                                    dropout=params['dropout_rate'], bidirectional=True)
        
        self.attn = None
        if params['self_attn']:
            self.attn = MultiHeadAttentionLayer(
                hidden_size=params['rnn_size'],
                n_heads=params['nb_heads'],
                dropout=params['dropout_rate']
            )

        self.fnn_list = nn.ModuleList()
        if params['nb_rnn_layers'] and params['nb_fnn_layers']:
            for fc in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    nn.Linear(
                        in_features=params['fnn_size'] if fc else params['rnn_size'],
                        out_features=params['fnn_size'],
                        bias=True
                    )
                )
        self.fnn_list.append(
            nn.Linear(
                in_features=params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size'],
                out_features=out_dim,
                bias=True
            )
        )

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).to(device='cuda:0').unsqueeze(0)
            
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        ''' (batch_size, time_steps, feature_maps)'''

        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2] # bidirectional GRU fusion, [B, T, 256]*[B, T, 256] -> [B, T, 256]
        if self.attn is not None:
            x = self.attn.forward(x, x, x)
            x = torch.tanh(x)

        for fnn in range(len(self.fnn_list)-1):
            x = self.fnn_list[fnn](x)
        doa = torch.tanh(self.fnn_list[-1](x)) # [B, T, num_classes*3]: [B, T, 60]
        return doa

class GoalDescriptor(nn.Module):
    def __init__(
            self, 
            observation_space,
            output_size,
            audiogoal_sensor,
            pose_sensor,
            num_classes,
            encoder_type = 'CRNN',
            downsample=True,
    ):
        super().__init__()
        self._audiogoal_sensor = audiogoal_sensor
        self._pose_sensor = pose_sensor
        self.params = get_params(downsample=downsample)
        self.num_classes = num_classes

        freq_dim = observation_space.spaces[audiogoal_sensor].shape[0] # freq_bins,65
        seq_len = observation_space.spaces[audiogoal_sensor].shape[1] - 25 # time_frames,100
        channel = observation_space.spaces[audiogoal_sensor].shape[2] # channels,2

        if encoder_type == 'CRNN':
            self.cst_former = CRNN(
                in_feat_shape=(seq_len, channel, freq_dim), # [T, C, F]: [100, C, 65]
                out_dim=num_classes * 3, # 20 classes, each with x,y,z
                params=self.params,
            )

        self.lstm = nn.LSTM(
            input_size=num_classes * 3,
            hidden_size=output_size,
            num_layers=1,
            batch_first=True,
            dropout=0.1
        )

    def forward(self, observations):
        audio_observations = observations[self._audiogoal_sensor] # [B, F, T, C]
        audio_observations = audio_observations.permute(0, 3, 2, 1) # [B, C, T, F]
        audio_observations = audio_observations[:, :, :100, :] #[B, C, 100, F]

        predict = self.cst_former(audio_observations) # [B, T, num_classes*3]: [B, 50, 60]
        predict = F.avg_pool1d(predict.permute(0, 2, 1), kernel_size=10).permute(0, 2, 1) # [B,60,50/10]->[B,5,60]

        pos_x, pos_y, pos_z = predict[..., :20], predict[..., 20:40], predict[..., 40:60]
        sed = (torch.sqrt(pos_x**2 + pos_y**2 + pos_z**2) > 0.5).float() # if active sound event
        radians = torch.atan2(pos_y, pos_x) # [−π, π], relative positions
  
        pos = observations[self._pose_sensor]
        headings = pos[:, :, 2]
        headings = headings.unsqueeze(-1).repeat(1, 1, 20)
        radians = radians + sed * headings

        pos_x = torch.cos(radians)
        pos_y = torch.sin(radians)

        goal_descriptor = torch.cat([sed, pos_x, pos_y], dim=-1) # [B, 5, 20+20+20=60]

        # hidden_state: [num_layers * num_directions, batch, hidden_size], [1*1, B, output_size]
        _, (hidden_state, _) = self.lstm(goal_descriptor)

        return hidden_state[0] # [B,128]

    def load_cst_parameters(self, model_path):
        print("Load pretrained parameters from: ", model_path)
        self.cst_former.load_state_dict(
            torch.load(model_path)
        )