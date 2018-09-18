import yaml
import os
import sys
import shutil
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable, grad
from torch.backends import cudnn

from src.data import LoadDataset
from src.ufdn import LoadModel
from src.util import interpolate_vae, vae_loss

from tensorboardX import SummaryWriter 


# Load config file for experiment
cudnn.benchmark = True
config_path = sys.argv[1]
conf = yaml.load(open(config_path,'r'))
exp_name = conf['exp_setting']['exp_name']
img_size = conf['exp_setting']['img_size']
img_depth = conf['exp_setting']['img_depth']

trainer_conf = conf['trainer']
data_augment = trainer_conf['data_augment']

if trainer_conf['save_checkpoint']:
    model_path = conf['exp_setting']['checkpoint_dir']
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = model_path+exp_name+'/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

if trainer_conf['save_log'] or trainer_conf['save_fig']:
    if os.path.exists(conf['exp_setting']['log_dir']+exp_name):
        shutil.rmtree(conf['exp_setting']['log_dir']+exp_name)
    writer = SummaryWriter(conf['exp_setting']['log_dir']+exp_name)


# Fix seed
np.random.seed(conf['exp_setting']['seed'])
_ = torch.manual_seed(conf['exp_setting']['seed'])

# Load dataset
src_domain = conf['exp_setting']['source_domain']
tgt_domain = conf['exp_setting']['target_domain']
data_root = conf['exp_setting']['data_root']
batch_size = conf['trainer']['batch_size']
shuffle_source = conf['exp_setting']['shuffle_source']
shuffle_target = conf['exp_setting']['shuffle_target'] 

src_loader = LoadDataset(src_domain,data_root,batch_size,'train',shuffle=shuffle_source)
tgt_loader = LoadDataset(tgt_domain,data_root,batch_size,'train',shuffle=shuffle_target)

src_test = LoadDataset(src_domain,data_root,100,'test')
tgt_test = LoadDataset(tgt_domain,data_root,100,'test')

for (d1,_),(d2,_) in zip(src_test,tgt_test):
    if d1[0].shape[0] == 1:
        src_test_sample = d1[0].clone().repeat(3,1,1)*2-1
    else:
        src_test_sample = d1[0]*2-1
    if d2[0].shape[0] == 1:
        tgt_test_sample = d2[0].clone().repeat(3,1,1)*2-1
    else:
        tgt_test_sample = d2[0]*2-1
    break

# Load Model
enc_dim = conf['model']['vae']['encoder'][-1][1]
code_dim = conf['model']['vae']['code_dim']
vae_learning_rate = conf['model']['vae']['lr']
vae_betas = tuple(conf['model']['vae']['betas'])
df_learning_rate = conf['model']['D_feat']['lr']
df_betas = tuple(conf['model']['D_feat']['betas'])

vae = LoadModel('vae',conf['model']['vae'],img_size,img_depth)
d_feat = LoadModel('nn',conf['model']['D_feat'],img_size,enc_dim)
d_digit = LoadModel('nn',conf['model']['D_digit'],img_size,enc_dim)

reconstruct_loss = torch.nn.MSELoss()
clf_loss = nn.BCEWithLogitsLoss()
digit_clf_loss = nn.CrossEntropyLoss(ignore_index=-1)

vae = vae.cuda()
d_feat = d_feat.cuda()
d_digit = d_digit.cuda()

reconstruct_loss = reconstruct_loss.cuda()
clf_loss = clf_loss.cuda()
digit_clf_loss = digit_clf_loss.cuda()

# Optmizer
opt_vae = optim.Adam(list(vae.parameters())+list(d_digit.parameters()), lr=vae_learning_rate, betas=vae_betas)
opt_df = optim.Adam(list(d_feat.parameters()), lr=df_learning_rate, betas=df_betas)

# Training
global_step = 0
best_acc = 0
vae.train()
d_feat.train()
d_digit.train()

# Domain code setting
domain_code = np.concatenate([np.repeat(np.array([[*([1]*int(code_dim/2)),*([0]*int(code_dim/2))]]),batch_size,axis=0),
               np.repeat(np.array([[*([0]*int(code_dim/2)),*([1]*int(code_dim/2))]]),batch_size,axis=0)],
               axis=0)
invert_code = 1- domain_code

domain_code = Variable(torch.FloatTensor(domain_code).cuda(),requires_grad=False)
invert_code = Variable(torch.FloatTensor(invert_code).cuda(),requires_grad=False)

# Loss lambda setting
loss_lambda = {}
for k in trainer_conf['lambda'].keys():
    init = trainer_conf['lambda'][k]['init']
    final = trainer_conf['lambda'][k]['final']
    step = trainer_conf['lambda'][k]['step']
    loss_lambda[k] = {}
    loss_lambda[k]['cur'] = init
    loss_lambda[k]['inc'] = (final-init)/step
    loss_lambda[k]['final'] = final




while global_step < trainer_conf['total_step']:    
    for (src_img,src_label),(tgt_img,tgt_label) in zip(src_loader,tgt_loader):
        # Make all images 3-channel and perform augmentation if specified
        if src_img.shape[1] == 1:
            src_img = src_img.repeat(1,3,1,1)
            if data_augment and (global_step % 2 ==1):
                src_img = 1- src_img
        if tgt_img.shape[1] == 1:
            tgt_img = tgt_img.repeat(1,3,1,1)
            if data_augment and (global_step % 2 ==1):
                tgt_img = 1- tgt_img

        input_img = torch.cat([src_img,tgt_img],dim=0)
        input_img = Variable((input_img*2-1).cuda(),requires_grad=False)

        
        # Only using label from src domain
        tgt_label.fill_(-1)
        digit_label = Variable(torch.cat([src_label,tgt_label],dim=0).cuda(),requires_grad=False)
                
        
        # Train Feature Discriminator
        opt_df.zero_grad()
        
        enc_x = vae(input_img,return_enc=True).detach()
        domain_pred = d_feat(enc_x)

        df_loss = clf_loss(domain_pred,domain_code)
        df_loss.backward()
     
        opt_df.step()
        

        # Train VAE
        opt_vae.zero_grad()
        
        ### Reconstruction Phase
        recon_batch, mu, logvar = vae(input_img,insert_attrs = domain_code)
        mse,kl = vae_loss(recon_batch.view(batch_size,-1), input_img.view(batch_size,-1), mu, logvar, reconstruct_loss)
        recon_loss = (loss_lambda['pix_recon']['cur']*mse+loss_lambda['kl']['cur']*kl)
        recon_loss.backward()

        
        ### Adversarial Phase       
        enc_x = vae(input_img,return_enc=True)
        domain_pred = d_feat(enc_x)
        domain_loss = clf_loss(domain_pred,invert_code)            

        adv_loss = loss_lambda['feat_domain']['cur']*domain_loss
        adv_loss.backward()
            
        
        ### Digit Phase
        enc_x = vae(input_img,return_enc=True)
        digit_pred = d_digit(enc_x)
        dg_loss = digit_clf_loss(digit_pred,digit_label)
        dg_loss.backward()
        
        
        opt_vae.step()
        
        
        # End of step        
        global_step += 1
        
        # Records
        if trainer_conf['save_log'] and (global_step % trainer_conf['verbose_step'] ==0):
            writer.add_scalar('MSE', mse.data[0], global_step)
            writer.add_scalar('KL',  kl.data[0], global_step)

            writer.add_scalars('Domain_loss', {'VAE':adv_loss.data[0],
                                              'D':df_loss.data[0]}, global_step)
            writer.add_scalar('Digit_loss', dg_loss.data[0], global_step)


            
        # update lambda
        for k in loss_lambda.keys():
            if loss_lambda[k]['inc']*loss_lambda[k]['cur'] < loss_lambda[k]['inc']*loss_lambda[k]['final']:    
                loss_lambda[k]['cur'] += loss_lambda[k]['inc'] 


        if global_step%trainer_conf['checkpoint_step']==0 and trainer_conf['save_checkpoint'] and not trainer_conf['save_best_only']:
            torch.save(vae,model_path.format(global_step)+'.vae')
            torch.save(d_digit,model_path.format(global_step)+'.dnet')


        ### Show result
        if global_step% trainer_conf['plot_step'] ==0:
            vae.eval()
            d_digit.eval()

            # Reconstruct
            tmp = interpolate_vae(vae, src_test_sample, tgt_test_sample,attr_max=1.0,attr_dim=code_dim)
            fig1 = (tmp+1)/2

            # UDA test
            source_acc = []
            target_acc = []
            
            for test_batch,test_label in src_test:
                if test_batch.shape[1] == 1:
                    test_batch = test_batch.repeat(1,3,1,1)
                test_batch = Variable(test_batch.cuda())
                label_pred = d_digit(vae((test_batch*2-1),return_enc=True))
                acc = float(sum(np.argmax(label_pred.cpu().data.numpy(),axis=-1)==test_label.numpy().reshape(-1)))/len(test_label)
                
                source_acc.append(acc)
                
            for test_batch,test_label in tgt_test:
                if test_batch.shape[1] == 1:
                    test_batch = test_batch.repeat(1,3,1,1)
                test_batch = Variable(test_batch.cuda())
                label_pred = d_digit(vae((test_batch*2-1),return_enc=True))
                
                acc = float(sum(np.argmax(label_pred.cpu().data.numpy(),axis=-1)==test_label.numpy().reshape(-1)))/len(test_label)
                target_acc.append(acc)
                
                
            source_acc = sum(source_acc)/len(source_acc)
            target_acc = sum(target_acc)/len(target_acc)

            if target_acc > best_acc:
                best_acc = target_acc
                if trainer_conf['save_best_only'] and trainer_conf['save_checkpoint']:
                    with open(model_path+'acc.txt','w') as f:
                        f.write(str(global_step)+'\t'+str(best_acc)+'\n')
                    torch.save(vae,model_path+'vae')
                    torch.save(d_digit,model_path+'dnet')
            
            
            if trainer_conf['save_log']:
                writer.add_scalars('Accuracy',{'source':source_acc,
                                               'target':target_acc}, global_step)
            if trainer_conf['save_fig']:
                writer.add_image('interpolate', torch.FloatTensor(np.transpose(fig1,(2,0,1))), global_step)
                
            vae.train()
            d_digit.train()
