import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, grad

### Messy code for continuous translation
def interpolate_vae_3d(vae,img_1,img_2,img_3,attr_inters = 5,id_inters = 3, attr_max = 1.0, attr_dim=2,
                    random_test=False,return_each_layer=False, sd =1, disentangle_dim=None):
    
    attr_min = 1.0-attr_max

    alphas = np.linspace(attr_min, attr_max, attr_inters)
    if disentangle_dim:
        alphas = [torch.FloatTensor([*([1 - alpha]*int((attr_dim-disentangle_dim)/3)),
                                     *([0]*int((attr_dim-disentangle_dim)/3)),
                                     *([alpha]*int((attr_dim-disentangle_dim)/3)),
                                     *([ v for i in range(int(disentangle_dim/2)) for v in [1-alpha,alpha]])]) for alpha in alphas]\
                +[torch.FloatTensor([*([0]*int((attr_dim-disentangle_dim)/3)),
                                     *([alpha]*int((attr_dim-disentangle_dim)/3)),
                                     *([1-alpha]*int((attr_dim-disentangle_dim)/3)),
                                     *([ v for i in range(int(disentangle_dim/2)) for v in [alpha,1-alpha]])]) for alpha in alphas[1:]]\
                +[torch.FloatTensor([*([alpha]*int((attr_dim-disentangle_dim)/3)),
                                     *([1 - alpha]*int((attr_dim-disentangle_dim)/3)),
                                     *([0]*int((attr_dim-disentangle_dim)/3)),
                                     *([ v for i in range(int(disentangle_dim/2)) for v in [1-alpha,alpha]])]) for alpha in alphas[1:-1]]
    else:
        alphas = [torch.FloatTensor([*([1 - alpha]*int(attr_dim/3)),
                                     *([0]*int(attr_dim/3)),
                                     *([alpha]*int(attr_dim/3))]) for alpha in alphas]\
                +[torch.FloatTensor([*([0]*int(attr_dim/3)),
                                     *([alpha]*int(attr_dim/3)),
                                     *([1-alpha]*int(attr_dim/3))]) for alpha in alphas[1:]]\
                +[torch.FloatTensor([*([alpha]*int(attr_dim/3)),
                                     *([1 - alpha]*int(attr_dim/3)),
                                     *([0]*int(attr_dim/3))]) for alpha in alphas[1:-1]]
    
    enc_1 = vae(Variable(torch.FloatTensor(img_1).unsqueeze(0).cuda()),return_enc=True).cpu().data.numpy()
    enc_2 = vae(Variable(torch.FloatTensor(img_2).unsqueeze(0).cuda()),return_enc=True).cpu().data.numpy()
    enc_3 = vae(Variable(torch.FloatTensor(img_3).unsqueeze(0).cuda()),return_enc=True).cpu().data.numpy()

    if random_test:
        np.random.seed(sd)
        enc_1 = np.random.randn(*[i for i in enc_1.shape])
        enc_2 = np.random.randn(*[i for i in enc_2.shape])
        enc_3 = np.random.randn(*[i for i in enc_3.shape])
    
    if return_each_layer:
        pass

    else:
        d1_outputs = []
        d2_outputs = []
        d3_outputs = []

        for i in range(id_inters+1):
        
            # ID 1 -> 2
            row = []
            tmp_input = Variable(torch.FloatTensor(enc_1+i*((enc_2-enc_1)/id_inters)).cuda())
            for alpha in alphas:
                alpha = Variable(alpha.unsqueeze(0).expand((1, attr_dim)).cuda())
                tmp = vae.decode( tmp_input, insert_attrs = alpha)
                row.append(tmp.cpu().data.numpy()[0].transpose(-2,-1,-3))
            d1_outputs.append(row)
            # ID 2 -> 3
            row = []
            tmp_input = Variable(torch.FloatTensor(enc_2+i*((enc_3-enc_2)/id_inters)).cuda())
            for alpha in alphas:
                alpha = Variable(alpha.unsqueeze(0).expand((1, attr_dim)).cuda())
                tmp = vae.decode( tmp_input, insert_attrs = alpha)
                row.append(tmp.cpu().data.numpy()[0].transpose(-2,-1,-3))
            d2_outputs.append(row)
            # ID 3 -> 1
            row = []
            tmp_input = Variable(torch.FloatTensor(enc_3+i*((enc_1-enc_3)/id_inters)).cuda())
            for alpha in alphas:
                alpha = Variable(alpha.unsqueeze(0).expand((1, attr_dim)).cuda())
                tmp = vae.decode( tmp_input, insert_attrs = alpha)
                row.append(tmp.cpu().data.numpy()[0].transpose(-2,-1,-3))
            d3_outputs.append(row)
            

        
        fig = []
        for i in range(id_inters+1):
            fig.append(np.concatenate(d1_outputs[i],axis=-2))
        for i in range(id_inters+1):
            fig.append(np.concatenate(d2_outputs[i],axis=-2))
        for i in range(id_inters+1):
            fig.append(np.concatenate(d3_outputs[i],axis=-2))
        fig = np.concatenate(fig,axis=-3)
        return fig

def interpolate_vae(vae,img_1,img_2,attr_inters = 7,id_inters = 6, attr_max = 1.2,
                    attr_dim=2,random_test=False,return_each_layer=False, sd =1,
                    disentangle_dim=None):
    
    attr_min = 1.0-attr_max
    
    alphas = np.linspace(attr_min, attr_max, attr_inters)
    if disentangle_dim:
        alphas = [torch.FloatTensor([*([1 - alpha]*int((attr_dim-disentangle_dim)/2)),
                                     *([alpha]*int((attr_dim-disentangle_dim)/2)),
                                     *([ v for i in range(int(disentangle_dim/2)) for v in [1-alpha,alpha]])]) for alpha in alphas]
    else:
        alphas = [torch.FloatTensor([*([1 - alpha]*int(attr_dim/2)), *([alpha]*int(attr_dim/2))]) for alpha in alphas]
   
    
    enc_1 = vae(Variable(torch.FloatTensor(img_1).unsqueeze(0).cuda()),return_enc=True).cpu().data.numpy()
    enc_2 = vae(Variable(torch.FloatTensor(img_2).unsqueeze(0).cuda()),return_enc=True).cpu().data.numpy()

    if random_test:
        np.random.seed(sd)
        enc_1 = np.random.randn(*[i for i in enc_1.shape])
        enc_2 = np.random.randn(*[i for i in enc_2.shape])
    
    if return_each_layer:
        pass

    else:
        outputs = []
        for i in range(id_inters+1):
            tmp_input = Variable(torch.FloatTensor(enc_1+i*((enc_2-enc_1)/id_inters)).cuda())
            for alpha in alphas:
                alpha = Variable(alpha.unsqueeze(0).expand((1, attr_dim)).cuda())
                tmp = vae.decode(tmp_input, alpha)
                if len(tmp.size())==2:
                    bs = tmp.size()[0]
                    tmp = tmp.view(bs,1,32,32)
                outputs.append(tmp.cpu().data.numpy()[0].transpose(-2,-1,-3))
        
        fig = []
        for i in range(id_inters+1):
            fig.append(np.concatenate(outputs[i*attr_inters:(i+1)*attr_inters],axis=-2))
        fig = np.concatenate(fig,axis=-3)
        return fig

# reference : https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
def calc_gradient_penalty(netD, real_data, fake_data, use_gpu = True, dec_output=2):
    alpha = torch.rand(real_data.shape[0], 1)
    if len(real_data.shape) == 4:
        alpha = alpha.unsqueeze(2).unsqueeze(3)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_gpu else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_gpu:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    if dec_output==2:
        disc_interpolates,_ = netD(interpolates)
    elif dec_output == 3:
        disc_interpolates,_,_ = netD(interpolates)
    else:
        disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                    grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_gpu else torch.ones(
                                  disc_interpolates.size()),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def vae_loss(recon_x, x, mu, logvar, rec_loss):
    loss = rec_loss(recon_x,x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loss,KLD