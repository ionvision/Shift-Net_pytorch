from util.NonparametricShift import Modified_NonparametricShift
import torch.nn as nn
import torch
import util.util as util
from torch.nn import Parameter

import numpy as np


class InnerBilinearShiftTripleModule(nn.Module):
    def __init__(self, dim, inner_nc, shift_sz, stride, activation='ReLU'):
        super(InnerBilinearShiftTripleModule, self).__init__()

        self.dim = dim
        self.inner_nc = inner_nc
        self.shift_sz = shift_sz
        self.stride = stride

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor
        self.U = Parameter(self.Tensor(self.dim, self.dim))
        self.V = Parameter(self.Tensor(self.dim, self.dim))
        self.v = Parameter(self.Tensor(inner_nc, 1))

        nn.init.uniform_(self.U)
        nn.init.uniform_(self.V)
        nn.init.uniform_(self.v)

        self.act_func = getattr(nn, activation)()


    def forward(self, input, mask, stride, triple_w, flag, show_flow):
        assert input.dim() == 4, "Input Dim has to be 4"
        self.triple_w = triple_w
        self.flag = flag
        self.show_flow = show_flow

        self.bz, c_real, self.h, self.w = input.size()
        c = c_real

        self.ind_lst = self.Tensor(self.bz, self.h * self.w, self.h * self.w).zero_()

        # former and latter are all tensors
        former_all = input.narrow(1, 0, c//2) ### decoder feature
        latter_all = input.narrow(1, c//2, c//2) ### encoder feature
        shift_masked_all = torch.Tensor(former_all.size()).type_as(former_all).fill_(0) # addition feature

        shift_masked_dim1 = shift_masked_all.size(1)
        assert mask.dim() == 2, "Mask dimension must be 2"

        if torch.cuda.is_available:
            self.flag = self.flag.cuda()

        # None batch version
        self.shift_offsets = []

        for idx in range(self.bz):
            latter = latter_all.narrow(0, idx, 1) ### encoder feature
            former = former_all.narrow(0, idx, 1) ### decoder feature

            # (196, 256), (828, 256), (196), (256,1), 256, 32, 32, 1
            X, Y, P, flag, i_1, i_2, i_3, i_4 = util.format_data(former, latter, self.flag, self.shift_sz, self.stride)

            v = torch.squeeze(self.v)# NOT VERY CLEA, BUT XAVIER NEEDS 2d TENSOR

            # return: U: 196*196
            #         V: 828*196
            U, V = util.filter_M(self.U, self.V, flag)
            A, XT_U, VT_y = util.bilinear_attention_map(X, Y, U, V, v, P)
            # print(torch.mean(self.U.data))
            # print(torch.mean(self.V))
            # print(torch.mean(self.v))
            # print('....')
            attention_tensor = util.bilinear_attention(XT_U, A, VT_y, v)

            # print(attention_tensor.size)

            # INDEXES INSIDE THE MASK
            mask_indexes = (self.flag == 1).nonzero()

            # GET HOLDER FOR LATER
            shift_holder = shift_masked_all[idx].clone()
            shift_holder = shift_holder.view(shift_masked_dim1, -1)
            shift_holder[:, mask_indexes] = attention_tensor.t().unsqueeze(-1)

            shift_masked_all[idx] = shift_holder.view(former.shape)

        return torch.cat((former_all, latter_all, shift_masked_all), 1)

    def get_flow_src(self):
        return self.flow_srcs
