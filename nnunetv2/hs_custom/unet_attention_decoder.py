import numpy as np
from typing import Union, Tuple, List, Type

import torch
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from torch import nn
from torch.nn.modules.dropout import _DropoutNd

from nnunetv2.hs_custom.attention_gated_unet.models.layers.grid_attention_layer import GridAttentionBlock3D
from nnunetv2.hs_custom.attention_gated_unet.models.networks.utils import UnetGridGatingSignal3, UnetUp3_CT_HS

class UNetAttentionDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None
                 ):
        """
        
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                              "resolution stages - 1 (n_stages in encoder - 1), " \
                                                              "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs
        
        attention_dsample=(2,2,2)
        nonlocal_mode='concatenation'

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        attentions = []
        ups = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s] # decoder 에서 upsampling 되는 피처맵
            input_features_skip = encoder.output_channels[-(s + 1)] # encoder 에서 바로 넘어오는 피처맵
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            # stages.append(StackedResidualBlocks(
            #     n_blocks=n_conv_per_stage[s - 1],
            #     conv_op=encoder.conv_op,
            #     input_channels=2 * input_features_skip,
            #     output_channels=input_features_skip,
            #     kernel_size=encoder.kernel_sizes[-(s + 1)],
            #     initial_stride=1,
            #     conv_bias=conv_bias,
            #     norm_op=norm_op,
            #     norm_op_kwargs=norm_op_kwargs,
            #     dropout_op=dropout_op,
            #     dropout_op_kwargs=dropout_op_kwargs,
            #     nonlin=nonlin,
            #     nonlin_kwargs=nonlin_kwargs,
            # ))
            
            stages.append(StackedResidualBlocks(
                n_blocks=n_conv_per_stage[s - 1],
                conv_op=encoder.conv_op,
                input_channels=input_features_skip + input_features_below, # attention 으로 인해 input_channels 수 바꿨음
                output_channels=input_features_skip,
                kernel_size=encoder.kernel_sizes[-(s + 1)],
                initial_stride=1,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ))
            
            attentions.append(GridAttentionBlock3D(in_channels=input_features_skip, gating_channels=input_features_below,
                                                    inter_channels=input_features_skip, sub_sample_factor=attention_dsample, mode=nonlocal_mode))

            ups.append(UnetUp3_CT_HS(scale_factor=(stride_for_transpconv[0], stride_for_transpconv[1], stride_for_transpconv[2])))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))
        
        self.gating = UnetGridGatingSignal3(encoder.output_channels[-1], encoder.output_channels[-1], kernel_size=(1, 1, 1), is_batchnorm=True)

        self.stages = nn.ModuleList(stages)
        self.attentions = nn.ModuleList(attentions)
        self.ups = nn.ModuleList(ups)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        seg_outputs = []
        gating = self.gating(skips[-1]) # center 가 꼭 들어가야하나?
        for s in range(len(self.stages)):
            g_conv, _ = self.attentions[s](skips[-(s + 2)], gating) # ???
            # x = self.transpconvs[s](lres_input)
            # x = self.transpconvs[s](gating)
            # x = torch.cat((g_conv, x), 1)
            # t = torch.cat((g_conv, x), 1)
            # print(t.shape)
            x = self.ups[s](g_conv, gating)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            gating = x # ???

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output

# if __name__ == '__main__':
#     data = torch.rand((1, 1, 96, 96, 96))

#     model = UNetAttentionDecoder(
        
#     )