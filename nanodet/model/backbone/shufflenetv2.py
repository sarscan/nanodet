import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from ..module.activation import act_layers

model_urls = {
    "shufflenetv2_0.5x": "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",  # noqa: E501
    "shufflenetv2_1.0x": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",  # noqa: E501
    "shufflenetv2_1.5x": "https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth",  # noqa: E501
    "shufflenetv2_2.0x": "https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth",  # noqa: E501
}


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride, activation="ReLU"):    # 24 116 2, 116 116 1；116 232 2，232 232 1
        super(ShuffleV2Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2  # 向下取整 58 其实就是取一半
        assert (self.stride != 1) or (inp == branch_features << 1)  # 第一个block stride=2，但out channel不等于in channel，而后面的repeats，stride=1，但in和out channel相等

        if self.stride > 1:
            self.branch1 = nn.Sequential(  # this is a depth wise separable conv, to reduce computation and weight volume
                self.depthwise_conv(
                    inp, inp, kernel_size=3, stride=self.stride, padding=1
                ),
                nn.BatchNorm2d(inp),
                nn.Conv2d(  # when kernel size to be 1, it called point-wise conv, 其实就是把channel方向上的值按权重相加
                    inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(branch_features),
                act_layers(activation),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            act_layers(activation),
            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
            ),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            act_layers(activation),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)  # repeat 3个block是不对前一半channel做conv的
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)  # 第一个block的输出，就是在channel维度上cat两个branch

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        model_size="1.5x",
        out_stages=(2, 3, 4),
        with_last_conv=False,
        kernal_size=3,
        activation="ReLU",
        pretrain=True,
    ):
        super(ShuffleNetV2, self).__init__()
        # out_stages can only be a subset of (2, 3, 4)
        assert set(out_stages).issubset((2, 3, 4))

        print("model size is ", model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        self.out_stages = out_stages
        self.with_last_conv = with_last_conv
        self.kernal_size = kernal_size
        self.activation = activation
        if model_size == "0.5x":
            self._stage_out_channels = [24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            self._stage_out_channels = [24, 116, 232, 464, 1024]  # 一共5个stage，这里写的是每个stage的输出的channel数量 最后一个1024由于没开last conv就没用
        elif model_size == "1.5x":
            self._stage_out_channels = [24, 176, 352, 704, 1024]
        elif model_size == "2.0x":
            self._stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channels = 3  # gsc*打开onnx能看到input的shape是1x3x416x416，你看下面conv是用到channel 3了，N直接在导出模型时改一下就行，H和W好像是训练的时候设置了
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            act_layers(activation),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]  # format就是用x去替换{}
        for name, repeats, output_channels in zip(
            stage_names, self.stage_repeats, self._stage_out_channels[1:]   # (stage2 4 116), (stage3 8 232)，(stage4 4 464) 3个tuple
        ):
            seq = [
                ShuffleV2Block(
                    input_channels, output_channels, 2, activation=activation   # 大致了解第一个stage2的结构，后面stage3，4都是一样的，只不过half out c刚好等于in c，
                )
            ]
            for i in range(repeats - 1):    # 重复n-1次的意思，就是说上面一个是1/2采样，后面是不采样的，一共是n个
                seq.append(
                    ShuffleV2Block(
                        output_channels, output_channels, 1, activation=activation
                    )
                )
            setattr(self, name, nn.Sequential(*seq))    # 这里要的是self.stageX=, 写成self.name就不对了
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        if self.with_last_conv:
            conv5 = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channels),
                act_layers(activation),
            )
            self.stage4.add_module("conv5", conv5)  # backbone这么简单的吗？竟然只有conv bn relu
        self._initialize_weights(pretrain)

    def forward(self, x):
        x = self.conv1(x)   # a squence of modules
        x = self.maxpool(x) # a module
        output = []
        for i in range(2, 5):   # i: 2,3,4
            stage = getattr(self, "stage{}".format(i))  # a squence of ShuffleV2Block, 其实也是module呀
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)    # tuple 是不可变的sequence，相当于把stage2 3 4的输出都搞出来了，去看FPN的forward函数，输入肯定是这个turple
                                # 我在onnx上看到这个3个边，后面接的是point wise conv，3个out的channel都变成96了

    def _initialize_weights(self, pretrain=True):
        print("init weights...")
        for name, m in self.named_modules():
            # print(name)     # gsc* 打印出来根本就没有first
            if isinstance(m, nn.Conv2d):
                if "first" in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])   # conv2d的weight和bias在init里明明已经随机初始化了，而且值看上去更合理
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # BN里的gamma和beta也叫weight和bias，构造函数里已经初始化为1和0了
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
        if pretrain:
            url = model_urls["shufflenetv2_{}".format(self.model_size)]
            if url is not None:
                pretrained_state_dict = model_zoo.load_url(url)     # 下载backbone对应的训练好的模型参数, 那外面的checkpoint应该就是fpn和head的参数
                print("=> loading pretrained model {}".format(url))
                self.load_state_dict(pretrained_state_dict, strict=False)   # 加载模型参数，上面设置的参数是说没有预训练参数的情况呀
                # print(self.state_dict())  # 这个也能打印显然就是weight和bias这种tensor
