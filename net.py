from mxnet.base import numeric_types
from mxnet import nd
from mxnet.gluon import nn

def conv_block(kernel_size, channels, stride, pad, dropout):
    out = nn.HybridSequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size, strides=stride, padding=pad, use_bias=False, weight_initializer='xavier')
    )
    if dropout > 0:
        out.add(nn.Dropout(dropout))
    return out

class layer_block(nn.HybridBlock):
    def __init__(self, growth_rate, dp, **kwargs):
        super(layer_block, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        self.net.add(conv_block(kernel_size=3, channels=growth_rate, stride=1, pad=1, dropout=dp))

    def hybrid_forward(self, F, x):
        out = self.net(x)
        x = F.concat(x, out, dim=1)
        return x

class bl_layer_block(nn.HybridBlock):
    def __init__(self, growth_rate, dp, width, **kwargs):
        super(bl_layer_block, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        self.net.add(
            conv_block(kernel_size=1, channels=int(width*growth_rate), stride=1, pad=0, dropout=dp),
            conv_block(kernel_size=3, channels=growth_rate, stride=1, pad=1, dropout=dp)
        )

    def hybrid_forward(self, F, x):
        out = self.net(x)
        x = F.concat(x, out, dim=1)
        return x

class bl_layer_block2(nn.HybridBlock):
    def __init__(self, growth_rate, dp, width, **kwargs):
        super(bl_layer_block2, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        self.net.add(
            conv_block(kernel_size=1, channels=int(width*growth_rate), stride=1, pad=0, dropout=dp),
            conv_block(kernel_size=3, channels=growth_rate, stride=2, pad=1, dropout=dp),
            nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), ceil_mode=True),
            conv_block(kernel_size=1, channels=growth_rate, stride=1, pad=0, dropout=dp)
        )

    def hybrid_forward(self, F, x):
        out1 = x
        out2 = x
        for layer in self.net[0:2]:
            out1 = layer(out1)
        for layer in self.net[2:]:
            out2 = layer(out2)
        x = F.concat(out1, out2, dim=1)
        return x

def transition_block(ch, dp):
    out = nn.HybridSequential()
    out.add(
        conv_block(kernel_size=1, channels=ch, stride=1, pad=0, dropout=dp),
        nn.MaxPool2D(pool_size=(2 ,2), strides=(2, 2), ceil_mode=True)
    )
    return out

def transition_block3x3(ch, dp):
    out = nn.HybridSequential()
    out.add(conv_block(kernel_size=3, channels=ch, stride=2, pad=0, dropout=dp))
    return out

def transition_w_o_block(ch, dp):
    out = nn.HybridSequential()
    out.add(conv_block(kernel_size=1, channels=ch, stride=1, pad=0, dropout=dp))
    return out

class DSOD300(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(DSOD300, self).__init__(**kwargs)
        growth_rate = 48
        dropout = 0
        nchannels = 128
        with self.name_scope():
            self.net1 = nn.HybridSequential()
            self.net1.add(
                nn.Conv2D(64, 3, strides=2, padding=1, use_bias=False, weight_initializer='xavier'),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(64, 3, strides=1, padding=1, use_bias=False, weight_initializer='xavier'),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(128, 3, strides=1, padding=1, use_bias=False, weight_initializer='xavier'),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), ceil_mode=True)
            )

            times = 1
            for i in range(6):
                self.net1.add(bl_layer_block(growth_rate, dropout, 4))
                nchannels += growth_rate
            nchannels = int(nchannels/times)
            self.net1.add(transition_block(nchannels, dropout))
            for i in range(8):
                self.net1.add(bl_layer_block(growth_rate, dropout, 4))
                nchannels += growth_rate
            nchannels = int(nchannels/times)
            self.net1.add(transition_w_o_block(nchannels, dropout))
            
            self.net2 = nn.HybridSequential()
            self.net2.add(nn.MaxPool2D(pool_size=(2, 2),strides=(2, 2), ceil_mode=True))
            for i in range(8):
                self.net2.add(bl_layer_block(growth_rate, dropout, 4))
                nchannels += growth_rate
            nchannels = int(nchannels/times)
            self.net2.add(transition_w_o_block(nchannels, dropout))
            for i in range(8):
                self.net2.add(bl_layer_block(growth_rate, dropout, 4))
                nchannels += growth_rate
            self.net2.add(transition_w_o_block(256, dropout))

            self.net3 = nn.HybridSequential()
            self.net3.add(
                nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), ceil_mode=True),
                conv_block(kernel_size=1, channels=256, stride=1, pad=0, dropout=dropout)
            )

            self.net4 = nn.HybridSequential()
            self.net4.add(bl_layer_block2(256, dropout, 1))

            self.net5 = nn.HybridSequential()
            self.net5.add(bl_layer_block2(128, dropout, 1))

            self.net6 = nn.HybridSequential()
            self.net6.add(bl_layer_block2(128, dropout, 1))

            self.net7 = nn.HybridSequential()
            self.net7.add(bl_layer_block2(128, dropout, 1))

    def hybrid_forward(self, F, x):
        out1 = self.net1(x)
        z1 = self.net2(out1)
        z2 = self.net3(out1)
        out2 = F.concat(z1, z2, dim=1)
        out3 = self.net4(out2)
        out4 = self.net5(out3)
        out5 = self.net6(out4)
        out6 = self.net7(out5)

        return out1, out2, out3, out4, out5, out6
