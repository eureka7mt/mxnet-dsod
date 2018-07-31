from mxnet.base import numeric_types
from mxnet import nd
from mxnet.gluon import nn


def conv_block(kernel_size, channels, stride, pad, dropout):
    out = nn.HybridSequential()
    out.add(
        nn.Conv2D(channels, kernel_size, strides=stride, padding=pad, use_bias=False, weight_initializer='xavier'),
        nn.BatchNorm(epsilon=1e-4),
        nn.Activation('relu'),
    )
    if dropout > 0:
        out.add(nn.Dropout(dropout))
    return out

def trans_block(kernel_size, channels, stride, dropout):
    out = nn.HybridSequential()
    out.add(
        nn.Conv2DTranspose(channels=channels, kernel_size=kernel_size, strides=stride, use_bias=False, weight_initializer='bilinear'),
        nn.BatchNorm(epsilon=1e-4),
        nn.Activation('relu')
    )
    if dropout > 0:
        out.add(nn.Dropout(dropout))
    return out

def dilation_conv_block(kernel_size, channels, stride, pad, dropout, dilation):
    out = nn.HybridSequential()
    out.add(
        nn.Conv2D(channels, kernel_size, strides=stride, padding=pad, dilation=dilation, use_bias=False, weight_initializer='xavier'),
        nn.BatchNorm(epsilon=1e-4),
        nn.Activation('relu')
    )
    if dropout > 0:
        out.add(nn.Dropout(dropout))
    return out

def recurrent_layer(kernel_size, channels, dropout):
    out = nn.HybridSequential()
    out = trans_block(kernel_size=kernel_size, channels=channels, stride=2, dropout=dropout)
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

class channel_level(nn.HybridBlock):
    def __init__(self, channel_num, **kwargs):
        super(channel_level, self).__init__(**kwargs)
        self.net1 = nn.HybridSequential()
        self.net1.add(nn.Dense(channel_num, activation='sigmoid', prefix=self.prefix))
        self.net2 = nn.HybridSequential()
        self.net2.add(nn.Activation('relu'))

    def hybrid_forward(self, F, x, y):
        sigmoid = self.net1(y)
        sigmoid = sigmoid.reshape((0 ,0, 1, 1))
        out = F.broadcast_mul(lhs=sigmoid, rhs=x)
        out = self.net2(out)
        return out
#x:from_layer,y:relu_name,z:att_name
class global_level(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(global_level, self).__init__(**kwargs)
        self.net1 = nn.HybridSequential()
        self.net1.add(nn.Dense(1, activation='sigmoid', prefix=self.prefix))
        #self.net2 = nn.HybridSequential()
        #self.net2.add(nn.Activation('relu'))

    def hybrid_forward(self, F, x, y, z):
        sigmoid = self.net1(y)
        sigmoid = sigmoid.reshape((0 ,0, 1, 1))
        scale =  F.broadcast_mul(lhs=sigmoid, rhs=z)
        #relu = self.net2(scale)
        residual = F.broadcast_add(lhs=x, rhs=scale)
        return residual




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

class GRP_DSOD320(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(GRP_DSOD320, self).__init__(**kwargs)
        growth_rate = 48
        dropout = 0
        nchannels = 128
        with self.name_scope():
            self.net0 = nn.HybridSequential()
            self.net0.add(
                nn.Conv2D(64, 3, strides=2, padding=1, use_bias=False, weight_initializer='xavier'),
                nn.BatchNorm(epsilon=1e-4),
                nn.Activation('relu'),
                nn.Conv2D(64, 3, strides=1, padding=1, use_bias=False, weight_initializer='xavier'),
                nn.BatchNorm(epsilon=1e-4),
                nn.Activation('relu'),
                nn.Conv2D(128, 3, strides=1, padding=1, use_bias=False, weight_initializer='xavier'),
                nn.BatchNorm(epsilon=1e-4),
                nn.Activation('relu')
            )
            self.net1 = nn.HybridSequential()
            self.net1.add(nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), ceil_mode=True))

            times = 1
            for i in range(6):
                self.net1.add(bl_layer_block(growth_rate, dropout, 4))
                nchannels += growth_rate
            nchannels = int(nchannels/times)
            self.net1.add(transition_w_o_block(nchannels, dropout))

            self.net2 = nn.HybridSequential()
            self.net2.add(nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), ceil_mode=True))
            for i in range(8):
                self.net2.add(bl_layer_block(growth_rate, dropout, 4))
                nchannels += growth_rate
            nchannels = int(nchannels/times)
            self.net2.add(transition_w_o_block(nchannels, dropout))
            
            self.extra0 = nn.HybridSequential()
            self.extra0.add(
                nn.MaxPool2D(pool_size=(4, 4), strides=(4, 4), ceil_mode=True),
                conv_block(kernel_size=1, channels=128, stride=1, pad=0, dropout=dropout)
            )

            self.extra1 = nn.HybridSequential()
            self.extra1.add(
                nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), ceil_mode=True),
                conv_block(kernel_size=1, channels=128, stride=1, pad=0, dropout=dropout)
            )

            self.net3 = nn.HybridSequential()
            self.net3.add(nn.MaxPool2D(pool_size=(2, 2),strides=(2, 2), ceil_mode=True))
            for i in range(8):
                self.net3.add(bl_layer_block(growth_rate, dropout, 4))
                nchannels += growth_rate
            nchannels = int(nchannels/times)
            self.net3.add(transition_w_o_block(nchannels, dropout))
            for i in range(8):
                self.net3.add(bl_layer_block(growth_rate, dropout, 4))
                nchannels += growth_rate
            self.net3.add(transition_w_o_block(171, dropout))

            self.first = nn.HybridSequential()
            self.first.add(
                nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), ceil_mode=True),
                conv_block(kernel_size=1, channels=171, stride=1, pad=0, dropout=dropout)
            )

            self.net4 = nn.HybridSequential()
            self.net4.add(bl_layer_block2(86, dropout, 1))

            self.net5 = nn.HybridSequential()
            self.net5.add(bl_layer_block2(86, dropout, 1))

            self.net6 = nn.HybridSequential()
            self.net6.add(bl_layer_block2(86, dropout, 1))

            self.net7 = nn.HybridSequential()
            self.net7.add(bl_layer_block2(128, dropout, 1))

            self.Recurrent1 = nn.HybridSequential()
            self.Recurrent1.add(recurrent_layer(2, 128, dropout))

            self.Recurrent2 = nn.HybridSequential()
            self.Recurrent2.add(recurrent_layer(2, 171, dropout))

            self.Recurrent3 = nn.HybridSequential()
            self.Recurrent3.add(recurrent_layer(2, 86, dropout))

            self.Recurrent4 = nn.HybridSequential()
            self.Recurrent4.add(recurrent_layer(1, 86, dropout))

            self.Recurrent5 = nn.HybridSequential()
            self.Recurrent5.add(recurrent_layer(1, 86,dropout))

    def hybrid_forward(self, F, x):
        out1 = self.net0(x)
        out2 = self.net1(out1)
        out3 = self.net2(out2)
        Extra0 = self.extra0(out1)
        Extra1 = self.extra1(out2)
        First = F.concat(out3, Extra0, Extra1, dim=1)
        out4 = self.net3(out3)
        f_first = self.first(First)
        Second = F.concat(out4, f_first, dim=1)
        Third = self.net4(Second)
        Fourth = self.net5(Third)
        Fifth = self.net6(Fourth)
        Sixth = self.net7(Fifth)
        recurrent1 = self.Recurrent1(Second)
        Recurrent1 = F.concat(recurrent1, First, dim=1)
        recurrent2 = self.Recurrent2(Third)
        Recurrent2 = F.concat(recurrent2, Second, dim=1)
        recurrent3 = self.Recurrent3(Fourth)
        Recurrent3 = F.concat(recurrent3, Third, dim=1)
        recurrent4 = self.Recurrent4(Fifth)
        Recurrent4 = F.concat(recurrent4, Fourth, dim=1)
        recurrent5 = self.Recurrent5(Sixth)
        Recurrent5 = F.concat(recurrent5, Fifth, dim=1)
        Recurrent6 = Sixth

        return Recurrent1, Recurrent2, Recurrent3, Recurrent4, Recurrent5, Recurrent6

class Gate_layer(nn.HybridBlock):
    def __init__(self, channel_nums=[], **kwargs):
        super(Gate_layer, self).__init__(**kwargs)
        self.num = int(len(channel_nums))
        self.channel_nums = channel_nums
        self.pre = nn.HybridSequential
        self.pre = GRP_DSOD320(prefix='GRP_')
        self.pool = nn.HybridSequential()
        self.pool.add(nn.GlobalAvgPool2D())

        self.denses = nn.HybridSequential()
        self.channel_levels = nn.HybridSequential()
        self.global_level = nn.HybridSequential()

        for i in range(self.num):
            self.channel_num = channel_nums[i]
            self.denses.add(nn.Dense(int(self.channel_num/16.0), activation='relu', prefix='den%2d_'%i))
            self.channel_levels.add(channel_level(self.channel_num, prefix='channel%2d_'%i)) 
            self.global_level.add(global_level(prefix='global%2d_'%i))

    def hybrid_forward(self, F, x):
        from_layers = self.pre(x)
        gates = [0]*self.num
        for i in range(self.num):
            from_layer = from_layers[i]
            #channel_num = self.channel_nums[i]
            relu = self.pool(from_layer)
            relu = self.denses[i](relu)
            att = self.channel_levels[i](from_layer, relu)
            gates[i] = self.global_level[i](from_layer, relu, att)
        return gates
