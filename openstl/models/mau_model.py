import math
import torch
import torch.nn as nn

from openstl.modules import MAUCell


class MAU_Model(nn.Module):
    r"""MAU Model

    Implementation of `MAU: A Motion-Aware Unit for Video Prediction and Beyond
    <https://openreview.net/forum?id=qwtfY-3ibt7>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(MAU_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.tau = configs.tau
        self.cell_mode = configs.cell_mode
        self.states = ['recall', 'normal']
        if not self.configs.model_mode in self.states:
            raise AssertionError
        cell_list = []

        width = W // configs.patch_size // configs.sr_size
        height = H // configs.patch_size // configs.sr_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = num_hidden[i - 1]
            cell_list.append(
                MAUCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                        configs.stride, self.tau, self.cell_mode)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Encoder
        n = int(math.log2(configs.sr_size))
        encoders = []
        encoder = nn.Sequential()
        encoder.add_module(name='encoder_t_conv{0}'.format(-1),
                           module=nn.Conv2d(in_channels=self.frame_channel,
                                            out_channels=self.num_hidden[0],
                                            stride=1,
                                            padding=0,
                                            kernel_size=1))
        encoder.add_module(name='relu_t_{0}'.format(-1),
                           module=nn.LeakyReLU(0.2))
        encoders.append(encoder)
        for i in range(n):
            encoder = nn.Sequential()
            encoder.add_module(name='encoder_t{0}'.format(i),
                               module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                out_channels=self.num_hidden[0],
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                kernel_size=(3, 3)
                                                ))
            encoder.add_module(name='encoder_t_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # Decoder
        decoders = []

        for i in range(n - 1):
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoder.add_module(name='c_decoder_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            decoders.append(decoder)

        if n > 0:
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        self.srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.merge = nn.Conv2d(
            self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)
        self.conv_last_sr = nn.Conv2d(
            self.frame_channel * 2, self.frame_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        device = frames_tensor.device
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch_size = frames.shape[0]
        height = frames.shape[3] // self.configs.sr_size
        width = frames.shape[4] // self.configs.sr_size
        frame_channels = frames.shape[2]
        next_frames = []
        T_t = []
        T_pre = []
        S_pre = []
        x_gen = None
        for layer_idx in range(self.num_layers):
            tmp_t = []
            tmp_s = []
            if layer_idx == 0:
                in_channel = self.num_hidden[layer_idx]
            else:
                in_channel = self.num_hidden[layer_idx - 1]
            for i in range(self.tau):
                tmp_t.append(torch.zeros(
                    [batch_size, in_channel, height, width]).to(device))
                tmp_s.append(torch.zeros(
                    [batch_size, in_channel, height, width]).to(device))
            T_pre.append(tmp_t)
            S_pre.append(tmp_s)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.pre_seq_length:
                net = frames[:, t]
            else:
                time_diff = t - self.configs.pre_seq_length
                net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
            frames_feature = net
            frames_feature_encoded = []
            for i in range(len(self.encoders)):
                frames_feature = self.encoders[i](frames_feature)
                frames_feature_encoded.append(frames_feature)
            if t == 0:
                for i in range(self.num_layers):
                    zeros = torch.zeros(
                        [batch_size, self.num_hidden[i], height, width]).to(device)
                    T_t.append(zeros)
            S_t = frames_feature
            for i in range(self.num_layers):
                t_att = T_pre[i][-self.tau:]
                t_att = torch.stack(t_att, dim=0)
                s_att = S_pre[i][-self.tau:]
                s_att = torch.stack(s_att, dim=0)
                S_pre[i].append(S_t)
                T_t[i], S_t = self.cell_list[i](T_t[i], S_t, t_att, s_att)
                T_pre[i].append(T_t[i])
            out = S_t

            for i in range(len(self.decoders)):
                out = self.decoders[i](out)
                if self.configs.model_mode == 'recall':
                    out = out + frames_feature_encoded[-2 - i]

            x_gen = self.srcnn(out)
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames[:, 1:])
        else:
            loss = None

        return next_frames, loss
#在patch_size大于1时启用
# import math
# import torch
# import torch.nn as nn
#
# from openstl.modules import MAUCell
#
#
# class MAU_Model(nn.Module):
#     r"""MAU Model
#
#     Implementation of `MAU: A Motion-Aware Unit for Video Prediction and Beyond
#     <https://openreview.net/forum?id=qwtfY-3ibt7>`_.
#
#     """
#
#     def __init__(self, num_layers, num_hidden, configs, **kwargs):
#         super(MAU_Model, self).__init__()
#         T, C, H, W = configs.in_shape
#
#         self.configs = configs
#         self.patch_size = configs.patch_size
#         self.frame_channel = configs.patch_size * configs.patch_size * C
#         self.num_layers = num_layers
#         self.num_hidden = num_hidden
#         self.tau = configs.tau
#         self.cell_mode = configs.cell_mode
#         self.states = ['recall', 'normal']
#         if not self.configs.model_mode in self.states:
#             raise AssertionError
#         cell_list = []
#
#         width = W // configs.patch_size // configs.sr_size
#         height = H // configs.patch_size // configs.sr_size
#         self.MSE_criterion = nn.MSELoss()
#
#         for i in range(num_layers):
#             in_channel = num_hidden[i - 1]
#             cell_list.append(
#                 MAUCell(in_channel, num_hidden[i], height, width, configs.filter_size,
#                         configs.stride, self.tau, self.cell_mode)
#             )
#         self.cell_list = nn.ModuleList(cell_list)
#
#         # Encoder
#         n = int(math.log2(configs.sr_size))
#         encoders = []
#         encoder = nn.Sequential()
#         encoder.add_module(name='encoder_t_conv{0}'.format(-1),
#                            module=nn.Conv2d(in_channels=self.frame_channel,
#                                             out_channels=self.num_hidden[0],
#                                             stride=1,
#                                             padding=0,
#                                             kernel_size=1))
#         encoder.add_module(name='relu_t_{0}'.format(-1),
#                            module=nn.LeakyReLU(0.2))
#         encoders.append(encoder)
#         for i in range(n):
#             encoder = nn.Sequential()
#             encoder.add_module(name='encoder_t{0}'.format(i),
#                                module=nn.Conv2d(in_channels=self.num_hidden[0],
#                                                 out_channels=self.num_hidden[0],
#                                                 stride=(2, 2),
#                                                 padding=(1, 1),
#                                                 kernel_size=(3, 3)
#                                                 ))
#             encoder.add_module(name='encoder_t_relu{0}'.format(i),
#                                module=nn.LeakyReLU(0.2))
#             encoders.append(encoder)
#         self.encoders = nn.ModuleList(encoders)
#
#         # Decoder
#         decoders = []
#
#         for i in range(n - 1):
#             decoder = nn.Sequential()
#             decoder.add_module(name='c_decoder{0}'.format(i),
#                                module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
#                                                          out_channels=self.num_hidden[-1],
#                                                          stride=(2, 2),
#                                                          padding=(1, 1),
#                                                          kernel_size=(3, 3),
#                                                          output_padding=(1, 1)
#                                                          ))
#             decoder.add_module(name='c_decoder_relu{0}'.format(i),
#                                module=nn.LeakyReLU(0.2))
#             decoders.append(decoder)
#
#         if n > 0:
#             decoder = nn.Sequential()
#             decoder.add_module(name='c_decoder{0}'.format(n - 1),
#                                module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
#                                                          out_channels=self.num_hidden[-1],
#                                                          stride=(2, 2),
#                                                          padding=(1, 1),
#                                                          kernel_size=(3, 3),
#                                                          output_padding=(1, 1)
#                                                          ))
#             decoders.append(decoder)
#         self.decoders = nn.ModuleList(decoders)
#
#         self.srcnn = nn.Sequential(
#             nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
#         )
#         self.merge = nn.Conv2d(
#             self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)
#         self.conv_last_sr = nn.Conv2d(
#             self.frame_channel * 2, self.frame_channel, kernel_size=1, stride=1, padding=0)
#
#     def pixel_to_patch(self, x, patch_size):
#         B, C, H, W = x.shape
#         x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
#         x = x.permute(0, 1, 3, 5, 2, 4)
#         x = x.reshape(B, C * patch_size * patch_size, H // patch_size, W // patch_size)
#         return x
#
#     def patch_to_pixel(self, x, patch_size):
#         B, C, H, W = x.shape
#         # C is frame_channel = original_C * P * P
#         original_C = C // (patch_size * patch_size)
#         x = x.reshape(B, original_C, patch_size, patch_size, H, W)
#         x = x.permute(0, 1, 4, 2, 5, 3)
#         x = x.reshape(B, original_C, H * patch_size, W * patch_size)
#         return x
#
#     def forward(self, frames_tensor, mask_true, **kwargs):
#         # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
#         device = frames_tensor.device
#         frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
#         mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
#
#         # Patching: [B, T, C, H, W] -> [B, T, C*P*P, H/P, W/P]
#         B, T, C, H, W = frames.shape
#         frames = frames.reshape(B * T, C, H, W)
#         frames = self.pixel_to_patch(frames, self.patch_size)
#         frames = frames.reshape(B, T, -1, H // self.patch_size, W // self.patch_size)
#
#         # Do the same for mask_true if it matches the image shape
#         if mask_true.shape == frames_tensor.permute(0, 1, 4, 2, 3).shape:
#             mask_true = mask_true.reshape(B * T, C, H, W)
#             mask_true = self.pixel_to_patch(mask_true, self.patch_size)
#             mask_true = mask_true.reshape(B, T, -1, H // self.patch_size, W // self.patch_size)
#
#         batch_size = frames.shape[0]
#         height = frames.shape[3] // self.configs.sr_size
#         width = frames.shape[4] // self.configs.sr_size
#         frame_channels = frames.shape[2]
#         next_frames = []
#         T_t = []
#         T_pre = []
#         S_pre = []
#         x_gen = None
#         for layer_idx in range(self.num_layers):
#             tmp_t = []
#             tmp_s = []
#             if layer_idx == 0:
#                 in_channel = self.num_hidden[layer_idx]
#             else:
#                 in_channel = self.num_hidden[layer_idx - 1]
#             for i in range(self.tau):
#                 tmp_t.append(torch.zeros(
#                     [batch_size, in_channel, height, width]).to(device))
#                 tmp_s.append(torch.zeros(
#                     [batch_size, in_channel, height, width]).to(device))
#             T_pre.append(tmp_t)
#             S_pre.append(tmp_s)
#
#         for t in range(self.configs.total_length - 1):
#             if t < self.configs.pre_seq_length:
#                 net = frames[:, t]
#             else:
#                 time_diff = t - self.configs.pre_seq_length
#                 net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
#             frames_feature = net
#             frames_feature_encoded = []
#             for i in range(len(self.encoders)):
#                 frames_feature = self.encoders[i](frames_feature)
#                 frames_feature_encoded.append(frames_feature)
#             if t == 0:
#                 for i in range(self.num_layers):
#                     zeros = torch.zeros(
#                         [batch_size, self.num_hidden[i], height, width]).to(device)
#                     T_t.append(zeros)
#             S_t = frames_feature
#             for i in range(self.num_layers):
#                 t_att = T_pre[i][-self.tau:]
#                 t_att = torch.stack(t_att, dim=0)
#                 s_att = S_pre[i][-self.tau:]
#                 s_att = torch.stack(s_att, dim=0)
#                 S_pre[i].append(S_t)
#                 T_t[i], S_t = self.cell_list[i](T_t[i], S_t, t_att, s_att)
#                 T_pre[i].append(T_t[i])
#             out = S_t
#
#             for i in range(len(self.decoders)):
#                 out = self.decoders[i](out)
#                 if self.configs.model_mode == 'recall':
#                     out = out + frames_feature_encoded[-2 - i]
#
#             x_gen = self.srcnn(out)
#             next_frames.append(x_gen)
#
#         # Stack output
#         next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
#
#         # Unpatch output: [B, T, C*P*P, H/P, W/P] -> [B, T, C, H, W]
#         B_out, T_out, C_out, H_out, W_out = next_frames.shape
#         next_frames = next_frames.reshape(B_out * T_out, C_out, H_out, W_out)
#         next_frames = self.patch_to_pixel(next_frames, self.patch_size)
#         next_frames = next_frames.reshape(B_out, T_out, -1, H_out * self.patch_size, W_out * self.patch_size)
#
#         # Unpatch frames (ground truth) for loss calculation
#         frames = frames.reshape(B * T, -1, H // self.patch_size, W // self.patch_size)
#         frames = self.patch_to_pixel(frames, self.patch_size)
#         frames = frames.reshape(B, T, -1, H, W)
#
#         # [B, T, C, H, W] -> [B, T, H, W, C]
#         next_frames = next_frames.permute(0, 1, 3, 4, 2).contiguous()
#         frames = frames.permute(0, 1, 3, 4, 2).contiguous()
#
#         if kwargs.get('return_loss', True):
#             loss = self.MSE_criterion(next_frames, frames[:, 1:])
#         else:
#             loss = None
#
#         return next_frames, loss
