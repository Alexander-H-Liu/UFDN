import torch
import torch.nn as nn
from torch.autograd import Variable

def get_act(name):
    if name == 'LeakyReLU':
        return nn.LeakyReLU(0.2)
    elif name == 'ReLU':
        return nn.ReLU()
    elif name == 'Tanh':
        return nn.Tanh()
    elif name == '':
        return None
    else:
        raise NameError('Unknown activation:'+name)

def LoadModel(name,parameter,img_size,input_dim):
    if name == 'vae':
        code_dim = parameter['code_dim']
        enc_list = []

        for layer,para in enumerate(parameter['encoder']):
            if para[0] == 'conv':
                if layer==0:
                    init_dim = input_dim
                next_dim,kernel_size,stride,pad,bn,act = para[1:7]
                act = get_act(act)
                enc_list.append((para[0],(init_dim, next_dim,kernel_size,stride,pad,bn,act)))
                init_dim = next_dim
            else:
                raise NameError('Unknown encoder layer type:'+para[0])

        dec_list = []
        for layer,para in enumerate(parameter['decoder']):
            if para[0] == 'conv':
                next_dim,kernel_size,stride,pad,bn,act,insert_code = para[1:8]
                act = get_act(act)
                dec_list.append((para[0],(init_dim, next_dim,kernel_size,stride,pad,bn,act),insert_code))
                init_dim = next_dim
            else:
                raise NameError('Unknown decoder layer type:'+para[0])
        return UFDN(enc_list,dec_list,code_dim)
    elif name == 'nn':
        dnet_list = []
        init_dim = input_dim
        for para in parameter['dnn']:
            if para[0] == 'fc':
                next_dim,bn,act,dropout = para[1:5]
                act = get_act(act)
                dnet_list.append((para[0],(init_dim, next_dim,bn,act,dropout)))
                init_dim = next_dim
            else:
                raise NameError('Unknown nn layer type:'+para[0])
        return Discriminator(dnet_list)
    elif name == 'cnn':
        dnet_list = []
        init_dim = input_dim
        cur_img_size = img_size
        reshaped = False
        for layer,para in enumerate(parameter['dnn']):
            if para[0] == 'conv':
                next_dim,kernel_size,stride,pad,bn,act = para[1:7]
                act = get_act(act)
                dnet_list.append((para[0],(init_dim, next_dim,kernel_size,stride,pad,bn,act)))
                init_dim = next_dim
                cur_img_size /= 2
            elif para[0] == 'fc':
                if not reshaped:
                    init_dim = int(cur_img_size*cur_img_size*init_dim)
                    reshaped = True
                next_dim,bn,act,dropout = para[1:5]
                act = get_act(act)
                dnet_list.append((para[0],(init_dim, next_dim,bn,act,dropout)))
                init_dim = next_dim
            else:
                raise NameError('Unknown encoder layer type:'+para[0])
        return Discriminator(dnet_list)
    else:
        raise NameError('Unknown model type:'+name)


# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

# create a Convolution/Deconvolution block
def ConvBlock(c_in, c_out, k=4, s=2, p=1, norm='bn', activation=None, transpose=False, dropout=None):
    layers = []
    if transpose:
        layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=s, padding=p))
    else:
        layers.append(         nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p))
    if dropout:
        layers.append(nn.Dropout2d(dropout))
    if norm == 'bn':
        layers.append(nn.BatchNorm2d(c_out))
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)

# create a fully connected layer
def FC(c_in, c_out, norm='bn', activation=None, dropout=None):
    layers = []
    layers.append(nn.Linear(c_in,c_out))
    if dropout:
        if dropout>0:
            layers.append(nn.Dropout(dropout))
    if norm == 'bn':
        layers.append(nn.BatchNorm1d(c_out))
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)

# UFDN model
# Reference : https://github.com/pytorch/examples/blob/master/vae/main.py
# list of layer should be a list with each element being (layer type,(layer parameter))
# fc should occur after/before any convblock if used in encoder/decoder
# e.g. ('conv',( input_dim, neurons, kernel size, stride, padding, normalization, activation))
#      ('fc'  ,( input_dim, neurons, normalization, activation))
class UFDN(nn.Module):
    def __init__(self, enc_list, dec_list, attr_dim):
        super(UFDN, self).__init__()

        ### Encoder
        self.enc_layers = []

        for l in range(len(enc_list)):
            self.enc_layers.append(enc_list[l][0])
            if enc_list[l][0] == 'conv':
                c_in,c_out,k,s,p,norm,act = enc_list[l][1]
                if l == len(enc_list) -1 :
                    setattr(self, 'enc_mu', ConvBlock(c_in,c_out,k,s,p,norm,act,transpose=False))
                    setattr(self, 'enc_logvar', ConvBlock(c_in,c_out,k,s,p,norm,act,transpose=False))
                else:
                    setattr(self, 'enc_'+str(l), ConvBlock(c_in,c_out,k,s,p,norm,act,transpose=False))
            elif enc_list[l][0] == 'fc':
                c_in,c_out,norm,act = enc_list[l][1]
                if l == len(enc_list) -1 :
                    setattr(self, 'enc_mu', FC(c_in,c_out,norm,act))
                    setattr(self, 'enc_logvar', FC(c_in,c_out,norm,act))
                else:
                    setattr(self, 'enc_'+str(l), FC(c_in,c_out,norm,act))
            else:
                raise ValueError('Unreconized layer type')

        ### Decoder
        self.dec_layers = []
        self.attr_dim = attr_dim

        for l in range(len(dec_list)):
            self.dec_layers.append((dec_list[l][0],dec_list[l][2]))
            if dec_list[l][0] == 'conv':
                c_in,c_out,k,s,p,norm,act = dec_list[l][1]
                if dec_list[l][2]: c_in += self.attr_dim
                setattr(self, 'dec_'+str(l), ConvBlock(c_in,c_out,k,s,p,norm,act,transpose=True))
            elif dec_list[l][0] == 'fc':
                c_in,c_out,norm,act = dec_list[l][1]
                if dec_list[l][2]: c_in += self.attr_dim
                setattr(self, 'dec_'+str(l), FC(c_in,c_out,norm,act))
            else:
                raise ValueError('Unreconized layer type')

        self.apply(weights_init)

    def encode(self, x):
        for l in range(len(self.enc_layers)-1):
            if (self.enc_layers[l] == 'fc')  and (len(x.size())>2):
                batch_size = x.size()[0]
                x = x.view(batch_size,-1)
            x = getattr(self, 'enc_'+str(l))(x)

        if (self.enc_layers[-1] == 'fc')  and (len(x.size())>2):
            batch_size = x.size()[0]
            x = x.view(batch_size,-1)
        mu = getattr(self, 'enc_mu')(x)
        logvar = getattr(self, 'enc_logvar')(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, insert_attrs = None):
        for l in range(len(self.dec_layers)):
            if (self.dec_layers[l][0] != 'fc') and (len(z.size()) != 4):
                z = z.unsqueeze(-1).unsqueeze(-1)
            if (insert_attrs is not None) and (self.dec_layers[l][1]):
                if len(z.size()) == 2:
                    z = torch.cat([z,insert_attrs],dim=1)
                else:
                    H,W = z.size()[2], z.size()[3]
                    z = torch.cat([z,insert_attrs.unsqueeze(-1).unsqueeze(-1).repeat(1,1,H,W)],dim=1)
            z = getattr(self, 'dec_'+str(l))(z)
        return z

    def forward(self, x, insert_attrs = None, return_enc = False):
        batch_size = x.size()[0]
        mu, logvar = self.encode(x)
        if len(mu.size()) > 2:
            mu = mu.view(batch_size,-1)
            logvar = logvar.view(batch_size,-1)
        z = self.reparameterize(mu, logvar)
        if return_enc:
            return z
        else:
            return self.decode(z,insert_attrs), mu, logvar


class Discriminator(nn.Module):
    def __init__(self, layer_list):
        super(Discriminator, self).__init__()

        self.layer_list = []

        for l in range(len(layer_list)-1):
            self.layer_list.append(layer_list[l][0])
            if layer_list[l][0] == 'conv':
                c_in,c_out,k,s,p,norm,act = layer_list[l][1]
                setattr(self, 'layer_'+str(l), ConvBlock(c_in,c_out,k,s,p,norm,act,transpose=False))
            elif layer_list[l][0] == 'fc':
                c_in,c_out,norm,act,drop = layer_list[l][1]
                setattr(self, 'layer_'+str(l), FC(c_in,c_out,norm,act,drop))
            else:
                raise ValueError('Unreconized layer type')


        self.layer_list.append(layer_list[-1][0])
        c_in,c_out,norm,act,_ = layer_list[-1][1]
        if not isinstance(c_out, list):
            c_out = [c_out]
        self.output_dim = len(c_out)

        for idx,d in enumerate(c_out):
            setattr(self, 'layer_out_'+str(idx), FC(c_in,d,norm,act,0))

        self.apply(weights_init)

    def forward(self, x):
        for l in range(len(self.layer_list)-1):
            if (self.layer_list[l] == 'fc') and (len(x.size()) != 2):
                batch_size = x.size()[0]
                x = x.view(batch_size,-1)
            x = getattr(self, 'layer_'+str(l))(x)

        output = []
        for d in range(self.output_dim):
            output.append(getattr(self,'layer_out_'+str(d))(x))

        if self.output_dim == 1:
            return output[0]
        else:
            return tuple(output)