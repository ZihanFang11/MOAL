import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn as nn
import sys



class FusionLayer(nn.Module):
    def __init__(self, num_views, fusion_type, in_size, hidden_size=64):
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        if self.fusion_type == 'weight':
            self.weight = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)
        if self.fusion_type == 'attention':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 32, bias=False),
                nn.Tanh(),
                nn.Linear(32, 1, bias=False)
            )

    def forward(self, emb_list):
        if self.fusion_type == "average":
            common_emb = sum(emb_list) / len(emb_list)
        elif self.fusion_type == "weight":
            weight = F.softmax(self.weight, dim=0)
            common_emb = sum([w * e for e, w in zip(weight, emb_list)])
        elif self.fusion_type == 'attention':
            emb_ = torch.stack(emb_list, dim=1)
            w = self.encoder(emb_)
            weight = torch.softmax(w, dim=1)
            common_emb = (weight * emb_).sum(1)
        else:
            sys.exit("Please using a correct fusion type")
        return common_emb


class RLNet(nn.Module):
    def __init__(self, nfeats, n_view, n_classes, n, args, device):
        """
        :param nfeats: list of feature dimensions for each view
        :param n_view: number of views
        :param n_classes: number of clusters
        :param n: number of samples
        :param args: Relevant parameters required to build the network
        """
        super(RLNet, self).__init__()
        self.n_classes = n_classes
        #  number of differentiable blocks
        self.layer_num = args.layer_num
        # the initial value of the threshold
        self.theta = nn.Parameter(torch.FloatTensor([args.thre]), requires_grad=True).to(device)
        self.n_view = n_view
        self.n = n
        self.device = device
        self.fusionlayer = FusionLayer(n_view,  args.fusion_type,self.n_classes, hidden_size=64)
        self.view_nets = nn.ModuleList([ view_net(n_classes,feat,args.gamma, device).to(device) for feat in nfeats])
        self.ZZ_init = nn.ModuleList([nn.Linear(feat, n_classes).to(device) for feat in nfeats])

    def self_active_l1(self, u):
        return F.selu(u - self.theta) - F.selu(-1.0 * u - self.theta)

    def forward(self, features, lap):
        output_z = []
        for j in range(0, self.n_view):
            out_tmp = self.ZZ_init[j](features[j] / 1.0)
            output_z.append( out_tmp.to(self.device))
        Z=self.fusionlayer(output_z)
        for i in range(0, self.layer_num):
            output_z = []

            for j in range(0, self.n_view):
                h= self.view_nets[j](Z, lap[j], features[j] / 1.0)
                output_z.append(self.self_active_l1(h))
            Z = self.fusionlayer(output_z)


        return Z


class view_net(Module):
    def __init__(self, out_features, nfea, gamma, device):
        super(view_net, self).__init__()
        self.S = nn.Linear(out_features, out_features).to(device)

        self.G = nn.Linear(out_features, out_features).to(device)

        self.U = nn.Linear(nfea, out_features).to(device)
        self.gamma=gamma
        self.device = device

    def forward(self, input, lap, view):
        input1 = self.S(input)
        input2 = self.U((view))
        output = torch.mm(lap, input)
        output = self.G(output)
        output = input1 + input2 - self.gamma * output
        return output
