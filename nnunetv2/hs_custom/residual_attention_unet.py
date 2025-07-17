import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from typing import Union, Type, List, Tuple
import torch
import torch.nn as nn

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0

### cutsom
from nnunetv2.hs_custom.unet_attention_decoder import UNetAttentionDecoder

class ResidualAttentionUNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[nn.Conv3d],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[nn.Dropout3d]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
    ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, (
            "n_blocks_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_blocks_per_stage: {n_blocks_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )
        self.encoder = ResidualEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            block,
            bottleneck_channels,
            return_skips=True,
            disable_default_stem=False,
            stem_channels=stem_channels,
        )
        self.decoder = UNetAttentionDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)

def get_default_network():
    input_channels = 1
    n_stages = 6
    features_per_stage = [32, 64, 128, 256, 320, 320]
    conv_op = nn.Conv3d
    kernel_sizes = [[3,3,3] for _ in range(6)]
    strides = [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [1,2,2]]
    n_blocks_per_stage = [1, 3, 4, 6, 6, 6]
    num_classes = 3
    n_conv_per_stage_decoder = [2,2,2,2,2]
    conv_bias = True
    norm_op = nn.InstanceNorm3d
    norm_op_kwargs = dict(eps=1e-05, affine=True)
    dropout_op = None
    dropout_op_kwargs = None
    nonlin = nn.LeakyReLU
    nonlin_kwargs = dict(inplace=True)
    deep_supervision = True # default 는 False 였음
    block = BasicBlockD
    bottleneck_channels = None
    stem_channels = None   
    
    model = ResidualAttentionUNet(
        input_channels = input_channels,
        n_stages = n_stages,
        features_per_stage = features_per_stage,
        conv_op = conv_op,
        kernel_sizes = kernel_sizes,
        strides = strides,
        n_blocks_per_stage = n_blocks_per_stage,
        num_classes = num_classes,
        n_conv_per_stage_decoder = n_conv_per_stage_decoder,
        conv_bias = conv_bias,
        norm_op = norm_op,
        norm_op_kwargs = norm_op_kwargs,
        dropout_op = dropout_op,
        dropout_op_kwargs = dropout_op_kwargs,
        nonlin = nonlin,
        nonlin_kwargs = nonlin_kwargs,
        deep_supervision = deep_supervision, # default 는 False 였음
        block = block,
        bottleneck_channels = bottleneck_channels,
        stem_channels = stem_channels        
    )
    
    return model

if __name__ == "__main__":
    data = torch.rand((4, 1, 128, 128, 128))
    
    
    # encoder = ResidualEncoder(
    #         input_channels,
    #         n_stages,
    #         features_per_stage,
    #         conv_op,
    #         kernel_sizes,
    #         strides,
    #         n_blocks_per_stage,
    #         conv_bias,
    #         norm_op,
    #         norm_op_kwargs,
    #         dropout_op,
    #         dropout_op_kwargs,
    #         nonlin,
    #         nonlin_kwargs,
    #         block,
    #         bottleneck_channels,
    #         return_skips=True,
    #         disable_default_stem=False,
    #         stem_channels=stem_channels,
    #     )
    # decoder = UNetResDecoder(encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)
    # decoder = UNetAttentionDecoder(encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)
    
    model = get_default_network()

    y = model(data)
    print('Residual Attention UNet')
    for x in y:
        print(x.shape)

    # print('Residual Encoder')
    # e = encoder(data)
    # for x in e:
    #     print(x.shape)

    # print('Residual Decoder')
    # d = decoder(e)
    # for x in d:
    #     print(x.shape)
    
    