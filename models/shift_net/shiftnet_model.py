import torch
from torch.nn import functional as F
import util.util as util
from models import networks
from models.shift_net.base_model import BaseModel
import time
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image



# Two shifts: the latter shift concontrates on the 1/4 of the region of square mask.
# 
class ShiftNetModel(BaseModel):
    def name(self):
        return 'ShiftNetModel'


    def create_random_mask(self):
        if self.mask_type == 'random':
            if self.opt.mask_sub_type == 'fractal':
                mask = util.create_walking_mask()  # create an initial random mask.

            elif self.opt.mask_sub_type == 'rect':
                mask, rand_t, rand_l = util.create_rand_mask(self.opt)
                self.rand_t = rand_t
                self.rand_l = rand_l
                return mask

            elif self.opt.mask_sub_type == 'island':
                mask = util.wrapper_gmask(self.opt)
        return mask

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN_f', 'G_GAN_l', 'G_L1_f', 'G_L1_l']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.opt.show_flow:
            self.visual_names = ['real_A', 'fake_B_f', 'fake_B_l', 'real_B', 'flow_srcs']
        else:
            self.visual_names = ['real_A', 'fake_B_f', 'fake_B_l', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_f', 'G_l','D_f', 'D_l']
        else:  # during test time, only load Gs
            self.model_names = ['G_f', 'G_l']


        # batchsize should be 1 for mask_global
        self.mask_global = torch.ByteTensor(1, 1, \
                                 opt.fineSize, opt.fineSize)

        # Here we need to set an artificial mask_global(not to make it broken, so center hole is ok.)
        self.mask_global.zero_()
        self.mask_global[:, :, int(self.opt.fineSize/4) + self.opt.overlap : int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap,\
                                int(self.opt.fineSize/4) + self.opt.overlap: int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap] = 1

        self.mask_type = opt.mask_type
        self.gMask_opts = {}


        self.wgan_gp = False
        # added for wgan-gp
        if opt.gan_type == 'wgan_gp':
            self.gp_lambda = opt.gp_lambda
            self.wgan_gp = True


        if len(opt.gpu_ids) > 0:
            self.use_gpu = True
            self.mask_global = self.mask_global.to(self.device)

        # load/define networks
        # self.ng_innerCos_list is the constraint list in netG inner layers.
        # self.ng_mask_list is the mask list constructing shift operation.
        if opt.add_mask2input:
            input_nc = opt.input_nc + 1
        else:
            input_nc = opt.input_nc

        self.netG_f, self.ng_innerCos_list_f, self.ng_shift_list_f = networks.define_G(input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt, self.mask_global, opt.norm, opt.use_spectral_norm_G, opt.init_type, self.gpu_ids, opt.init_gain) # add opt, we need opt.shift_sz and other stuffs
        self.netG_l, self.ng_innerCos_list_l, self.ng_shift_list_l = networks.define_G(input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt, self.mask_global, opt.norm, opt.use_spectral_norm_G, opt.init_type, self.gpu_ids, opt.init_gain) # add opt, we need opt.shift_sz and other stuffs
        if self.isTrain:
            use_sigmoid = False
            if opt.gan_type == 'vanilla':
                use_sigmoid = True  # only vanilla GAN using BCECriterion
            # don't use cGAN
            self.netD_f = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.use_spectral_norm_D, opt.init_type, self.gpu_ids, opt.init_gain)
            self.netD_l = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.use_spectral_norm_D, opt.init_type, self.gpu_ids, opt.init_gain)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN_f = networks.GANLoss(gan_type=opt.gan_type).to(self.device)
            self.criterionL1_f = torch.nn.L1Loss()
            self.criterionL1_mask_f = util.Discounted_L1(opt).to(self.device) # make weights/buffers transfer to the correct device
            self.criterionGAN_l = networks.GANLoss(gan_type=opt.gan_type).to(self.device)
            self.criterionL1_l = torch.nn.L1Loss()
            self.criterionL1_mask_l = util.Discounted_L1(opt, 'quarter').to(self.device) # make weights/buffers transfer to the correct device
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            if self.wgan_gp:
                opt.beta1 = 0
                self.optimizer_G_f = torch.optim.Adam(self.netG_f.parameters(),
                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_f = torch.optim.Adam(self.netD_f.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_G_l = torch.optim.Adam(self.netG_l.parameters(),
                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_l = torch.optim.Adam(self.netD_l.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G_f = torch.optim.Adam(self.netG_f.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_f = torch.optim.Adam(self.netD_f.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_G_l = torch.optim.Adam(self.netG_l.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_l = torch.optim.Adam(self.netD_l.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G_f)
            self.optimizers.append(self.optimizer_D_f)
            self.optimizers.append(self.optimizer_G_l)
            self.optimizers.append(self.optimizer_D_l)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
 
        self.print_networks(opt.verbose)

    def set_input(self, input):
        self.image_paths = input['A_paths']
        real_A = input['A'].to(self.device)
        real_B = input['B'].to(self.device)

        # Add mask to real_A
        if self.opt.mask_type == 'center':
            self.mask_global.zero_()
            self.mask_global[:, :, int(self.opt.fineSize/4) + self.opt.overlap : int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap,\
                                int(self.opt.fineSize/4) + self.opt.overlap: int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap] = 1
            self.rand_t, self.rand_l = int(self.opt.fineSize/4) + self.opt.overlap, int(self.opt.fineSize/4) + self.opt.overlap
        elif self.opt.mask_type == 'random':
            self.mask_global = self.create_random_mask().type_as(self.mask_global).view_as(self.mask_global)
        else:
            raise ValueError("Mask_type [%s] not recognized." % self.opt.mask_type)
        # Add 
        if not self.opt.isTrain and self.opt.offline_testing:
            self.mask_global = Image.open(os.path.join('masks', os.path.splitext(os.path.basename(self.image_paths[0]))[0]+'_mask.png'))
            self.mask_global = transforms.ToTensor()(self.mask_global).unsqueeze(0).type_as(real_A).byte()
            


        self.set_latent_mask(self.mask_global)

        real_A.narrow(1,0,1).masked_fill_(self.mask_global, 0.)#2*123.0/255.0 - 1.0
        real_A.narrow(1,1,1).masked_fill_(self.mask_global, 0.)#2*104.0/255.0 - 1.0
        real_A.narrow(1,2,1).masked_fill_(self.mask_global, 0.)#2*117.0/255.0 - 1.0

        if self.opt.add_mask2input:
            # make it 4 dimensions.
            # Mention: the extra dim, the masked part is filled with 0, non-mask part is filled with 1.
            real_A = torch.cat((real_A, (1 - self.mask_global).expand(real_A.size(0), 1, real_A.size(2), real_A.size(3)).type_as(real_A)), dim=1)

        self.real_A = real_A
        self.real_B = real_B
    

    def set_latent_mask(self, mask_global):
        for ng_shift in self.ng_shift_list_f: # ITERATE OVER THE LIST OF ng_shift_list
            ng_shift.set_mask(mask_global)
        for ng_shift in self.ng_shift_list_l: # ITERATE OVER THE LIST OF ng_shift_list
            ng_shift.set_mask(mask_global)
        for ng_innerCos in self.ng_innerCos_list_f: # ITERATE OVER THE LIST OF ng_innerCos_list:
            ng_innerCos.set_mask(mask_global)
        for ng_innerCos in self.ng_innerCos_list_f: # ITERATE OVER THE LIST OF ng_innerCos_list:
            ng_innerCos.set_mask(mask_global)

    def set_gt_latent(self):
        if not self.opt.skip:
            if self.opt.add_mask2input:
                # make it 4 dimensions.
                # Mention: the extra dim, the masked part is filled with 0, non-mask part is filled with 1.
                real_B = torch.cat([self.real_B, (1 - self.mask_global).expand(self.real_B.size(0), 1, self.real_B.size(2), self.real_B.size(3)).type_as(self.real_B)], dim=1)
            else:
                real_B = self.real_B
            self.netG_f(real_B) # input ground truth
            self.netG_l(real_B) # input ground truth


    def forward(self):
        self.fake_B_f = self.netG_f(self.real_A)
        # concat a mask with self.fake_B_f
        self.fake_B_f_c = torch.cat((self.fake_B_f, (1 - self.mask_global).expand(self.real_A.size(0), 1, self.real_A.size(2), self.real_A.size(3)).type_as(self.real_A)), dim=1)
        self.fake_B_l = self.netG_l(self.fake_B_f_c)

    # Just assume one shift layer.
    def set_flow_src(self):
        self.flow_srcs = self.ng_shift_list[0].get_flow()
        self.flow_srcs = F.interpolate(self.flow_srcs, scale_factor=8, mode='nearest')
        # Just to avoid forgetting setting show_map_false
        self.set_show_map_false()

    # Just assume one shift layer.
    def set_show_map_true(self):
        self.ng_shift_list[0].set_flow_true()

    def set_show_map_false(self):
        self.ng_shift_list[0].set_flow_false()

    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        fake_B_f = self.fake_B_f
        fake_B_l = self.fake_B_l
        # Real
        real_B = self.real_B # GroundTruth

        # Has been verfied, for square mask, let D discrinate masked patch, improves the results.
        if self.opt.mask_type == 'center' or self.opt.mask_sub_type == 'rect': 
            # Using the cropped fake_B as the input of D.
            fake_B_f = self.fake_B_f[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]
            fake_B_l = self.fake_B_l[:, :, self.rand_t + self.opt.fineSize//8:self.rand_t+self.opt.fineSize*3//8, \
                                           self.rand_l + self.opt.fineSize//8:self.rand_l+self.opt.fineSize*3//8]
            real_B_f = self.real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]
            real_B_l = self.real_B[:, :, self.rand_t + self.opt.fineSize//8:self.rand_t+self.opt.fineSize*3//8, \
                                           self.rand_l + self.opt.fineSize//8:self.rand_l+self.opt.fineSize*3//8]

        self.pred_fake_f = self.netD_f(fake_B_f.detach())
        self.pred_real_f = self.netD_f(real_B_f) # 120*120
        self.pred_fake_l = self.netD_l(fake_B_l.detach())
        self.pred_real_l = self.netD_l(real_B_l) # 64*64

        if self.wgan_gp:
            # Do not support two shifts for wgan-gp.
            self.loss_D_fake = torch.mean(self.pred_fake)
            self.loss_D_real = torch.mean(self.pred_real)

            # calculate gradient penalty
            alpha = torch.rand(real_B.size()).to(self.device)
            x_hat = alpha * real_B.detach() + (1 - alpha) * fake_B.detach()
            x_hat.requires_grad_(True)
            pred_hat = self.netD(x_hat)

            gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(self.device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

            gradient_penalty = self.gp_lambda * ((gradients.view(gradients.size(0), -1).norm(2, 1) - 1) ** 2).mean()

            self.loss_D = self.loss_D_fake - self.loss_D_real + gradient_penalty
        else:

            if self.opt.gan_type in ['vanilla', 'lsgan']:
                self.loss_D_fake_f = self.criterionGAN_f(self.pred_fake_f, False)
                self.loss_D_real_f = self.criterionGAN_f (self.pred_real_f, True)
                self.loss_D_fake_l = self.criterionGAN_l(self.pred_fake_l, False)
                self.loss_D_real_l = self.criterionGAN_l (self.pred_real_l, True)

                self.loss_D_f = (self.loss_D_fake_f + self.loss_D_real_f) * 0.5
                self.loss_D_l = (self.loss_D_fake_l + self.loss_D_real_l) * 0.5

            elif self.opt.gan_type == 're_s_gan':
                self.loss_D = self.criterionGAN(self.pred_real - self.pred_fake, True)

            elif self.opt.gan_type == 're_avg_gan':
                self.loss_D =  (self.criterionGAN (self.pred_real - torch.mean(self.pred_fake), True) \
                               + self.criterionGAN (self.pred_fake - torch.mean(self.pred_real), False)) / 2.
        # for `re_avg_gan`, need to retain graph of D.
        if self.opt.gan_type == 're_avg_gan':
            self.loss_D.backward(retain_graph=True)
        else:
            # Maybe we should add retain_graph.
            self.loss_D_f.backward()
            self.loss_D_l.backward()


    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_B_f = self.fake_B_f
        fake_B_l = self.fake_B_l
        # Has been verfied, for square mask, let D discrinate masked patch, improves the results.
        if self.opt.mask_type == 'center' or self.opt.mask_sub_type == 'rect': 
        # Using the cropped fake_B as the input of D.
            fake_B_f = self.fake_B_f[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]
            fake_B_l = self.fake_B_l[:, :, self.rand_t + self.opt.fineSize//8:self.rand_t+self.opt.fineSize*3//8, \
                                           self.rand_l + self.opt.fineSize//8:self.rand_l+self.opt.fineSize*3//8]
            real_B_f = self.real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]
            real_B_l = self.real_B[:, :, self.rand_t + self.opt.fineSize//8:self.rand_t+self.opt.fineSize*3//8, \
                                           self.rand_l + self.opt.fineSize//8:self.rand_l+self.opt.fineSize*3//8]
        pred_fake_f = self.netD_f(fake_B_f)
        pred_fake_l = self.netD_l(fake_B_l)


        if self.wgan_gp:
            self.loss_G_GAN = torch.mean(pred_fake)
        else:
            if self.opt.gan_type in ['vanilla', 'lsgan']:
                self.loss_G_GAN_f = self.criterionGAN_f(pred_fake_f, True)
                self.loss_G_GAN_l = self.criterionGAN_l(pred_fake_l, True)

            elif self.opt.gan_type == 're_s_gan':
                pred_real = self.netD (real_B)
                self.loss_G_GAN = self.criterionGAN (pred_fake - pred_real, True)

            elif self.opt.gan_type == 're_avg_gan':
                self.pred_real = self.netD(real_B)
                self.loss_G_GAN =  (self.criterionGAN (self.pred_real - torch.mean(self.pred_fake), False) \
                               + self.criterionGAN (self.pred_fake - torch.mean(self.pred_real), True)) / 2.


        # If we change the mask as 'center with random position', then we can replacing loss_G_L1_m with 'Discounted L1'.
        self.loss_G_L1_f, self.loss_G_L1_m_f, self.loss_G_L1_l, self.loss_G_L1_m_l = 0, 0, 0, 0
        self.loss_G_L1_f += self.criterionL1_f(fake_B_f, real_B_f) * self.opt.lambda_A
        self.loss_G_L1_l += self.criterionL1_l(fake_B_l, real_B_l) * self.opt.lambda_A
        # calcuate mask construction loss
        # When mask_type is 'center' or 'random_with_rect', we can add additonal mask region construction loss (traditional L1).
        # Only when 'discounting_loss' is 1, then the mask region construction loss changes to 'discounting L1' instead of normal L1.
        if self.opt.mask_type == 'center' or self.opt.mask_sub_type == 'rect': 
            mask_patch_fake_f = self.fake_B_f[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, \
                                                self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]
            mask_patch_fake_l = self.fake_B_l[:, :, self.rand_t + self.opt.fineSize//8:self.rand_t+self.opt.fineSize*3//8, \
                                           self.rand_l + self.opt.fineSize//8:self.rand_l+self.opt.fineSize*3//8]
            mask_patch_real_f = self.real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, \
                                        self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]
            mask_patch_real_l = self.real_B[:, :, self.rand_t + self.opt.fineSize//8:self.rand_t+self.opt.fineSize*3//8, \
                                           self.rand_l + self.opt.fineSize//8:self.rand_l+self.opt.fineSize*3//8]
            # Using Discounting L1 loss
            self.loss_G_L1_m_f += self.criterionL1_mask_f(mask_patch_fake_f, mask_patch_real_f)*self.opt.mask_weight
            self.loss_G_L1_m_l += self.criterionL1_mask_l(mask_patch_fake_l, mask_patch_real_l)*self.opt.mask_weight

        if self.wgan_gp:
            self.loss_G = self.loss_G_L1 + self.loss_G_L1_m - self.loss_G_GAN * self.opt.gan_weight
        else:
            self.loss_G_f = self.loss_G_L1_f + self.loss_G_L1_m_f + self.loss_G_GAN_f * self.opt.gan_weight
            self.loss_G_l = self.loss_G_L1_l + self.loss_G_L1_m_l + self.loss_G_GAN_l * self.opt.gan_weight


        # Third add additional netG contraint loss!
        self.ng_loss_value_f = 0
        self.ng_loss_value_l = 0
        if not self.opt.skip:
            for gl in self.ng_innerCos_list_f:
                self.ng_loss_value_f += gl.loss
            for gl in self.ng_innerCos_list_l:
                self.ng_loss_value_l += gl.loss
            self.loss_G_f += self.ng_loss_value_f
            self.loss_G_l += self.ng_loss_value_l

        self.loss_G_f.backward(retain_graph=True)
        self.loss_G_l.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD_f, True)
        self.set_requires_grad(self.netD_l, True)
        self.optimizer_D_f.zero_grad()
        self.optimizer_D_l.zero_grad()
        self.backward_D()
        self.optimizer_D_f.step()
        self.optimizer_D_l.step()

        # update G
        self.set_requires_grad(self.netD_f, False)
        self.set_requires_grad(self.netD_l, False)
        self.optimizer_G_f.zero_grad()
        self.optimizer_G_l.zero_grad()
        self.backward_G()
        self.optimizer_G_f.step()
        self.optimizer_G_l.step()


