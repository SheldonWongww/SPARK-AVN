import torch
import torch.nn as nn
from torchvision.models.resnet import conv3x3, conv1x1
from habitat_sim.utils.common import d3_40_colors_rgb

from cavn.common.utils import ResizeCenterCropper

class CustomBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=16, base_width=16, dilation=1, norm_layer=None):
        super(CustomBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.GroupNorm
        if groups != 16 or base_width != 16:
            raise ValueError('CustomBasicBlock only supports groups=16 and base_width=16')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in CustomBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(groups, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(groups, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class CustomResNet(nn.Module):
    def __init__(self, block, layers, num_input_channels=3, num_classes=64,
                 zero_init_residual=False, groups=16, width_per_group=16,
                 replace_stride_with_dilation=None, norm_layer=None):
        super(CustomResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.GroupNorm
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(num_input_channels, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(groups, self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, groups, 16, layers[0])
        self.layer2 = self._make_layer(block, groups, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, groups, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, groups, 128, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # Assumes that the input is a 64x64 image
        self.fc = nn.Linear(128 * block.expansion * 8 * 8, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): # actually only GroupNorm here
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, ngroups, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(ngroups, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(block, layers, **kwargs):
    model = CustomResNet(block, layers, **kwargs)
    return model

def custom_resnet18(**kwargs):
    """Implements a custom ResNet18 model used in Scene Memory Transformer.
    It takes as input a 64x64 image, and outputs a 64-d feature vector.
    When compared to the original ResNet18, the number of conv filters are reduced by 4, 
    the stride of the first conv layer is set to 1, the MaxPool and AdaptiveAvgPool layers are removed.
    """
    return _resnet(CustomBasicBlock, [2, 2, 2, 2], **kwargs)

def convert_semantics_to_rgb(semantics):
    """Converts semantic IDs to RGB images."""
    semantics = semantics.long() % 40
    mapping_rgb = torch.from_numpy(d3_40_colors_rgb).to(semantics.device)
    semantics_r = torch.take(mapping_rgb[:, 0], semantics)
    semantics_g = torch.take(mapping_rgb[:, 1], semantics)
    semantics_b = torch.take(mapping_rgb[:, 2], semantics)
    semantics_rgb = torch.stack([semantics_r, semantics_g, semantics_b], -1)

    return semantics_rgb

class SMTCNN(nn.Module):
    r"""A modified ResNet-18 architecture from https://arxiv.org/abs/1903.03878.

    Takes in observations and produces an embedding of the rgb and/or depth
    and/or semantic components.

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """
    def __init__(
        self,
        observation_space,
        obs_transform: nn.Module = ResizeCenterCropper(size=(64, 64)),
    ):
        super().__init__()

        self.obs_transform = obs_transform
        if self.obs_transform is not None:
            observation_space = obs_transform.transform_observation_space(observation_space)

        self._feat_dims = 0
        self.input_modalities = []
        if "rgb" in observation_space.spaces:
            self.input_modalities.append("rgb")
            n_input_rgb = observation_space.spaces["rgb"].shape[2]
            self.rgb_encoder = custom_resnet18(num_input_channels=n_input_rgb)
            self._feat_dims += 64

        if "depth" in observation_space.spaces:
            self.input_modalities.append("depth")
            n_input_depth = observation_space.spaces["depth"].shape[2]
            self.depth_encoder = custom_resnet18(num_input_channels=n_input_depth)
            self._feat_dims += 64

        # Semantic instance segmentation
        if "semantic" in observation_space.spaces:
            # Semantic object segmentation
            self.input_modalities.append("semantic")
            self.input_modalities.append("semantic_object")
            self.semantic_encoder = custom_resnet18(num_input_channels=6)
            self._feat_dims += 64

        self.layer_init()

    def layer_init(self):
        def weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

        self.apply(weights_init)

    def forward(self, observations):
        cnn_features = []
        if "rgb" in self.input_modalities:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            if self.obs_transform:
                rgb_observations = self.obs_transform(rgb_observations)
            cnn_features.append(self.rgb_encoder(rgb_observations))

        if "depth" in self.input_modalities:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            if self.obs_transform:
                depth_observations = self.obs_transform(depth_observations)
            cnn_features.append(self.depth_encoder(depth_observations))

        if "semantic" in self.input_modalities:
            assert "semantic_object" in observations.keys(), \
                "SMTCNN: Both instance and class segmentations must be available"
            semantic_observations = convert_semantics_to_rgb(observations["semantic"]).float()
            semantic_object_observations = observations["semantic_object"].float()
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            semantic_observations = torch.cat([semantic_observations, semantic_object_observations], -1)
            semantic_observations = semantic_observations.permute(0, 3, 1, 2) / 255.0
            if self.obs_transform:
                semantic_observations = self.obs_transform(semantic_observations)
            cnn_features.append(self.semantic_encoder(semantic_observations))

        cnn_features = torch.cat(cnn_features, dim=1)

        return cnn_features

    @property
    def feature_dims(self):
        return self._feat_dims

    @property
    def output_shape(self):
        return (self._feat_dims, )

    @property
    def is_blind(self):
        return False
