import pdb
import numpy as np

import torch
from torchvision.models.resnet import BasicBlock

from . import networks
from .base_model import BaseModel
from .resnet_related import *



class DualResNet(torch.nn.Module):
    def __init__(self):
        super(DualResNet, self).__init__()
        self.branch1 = BaseResNetBranch(BasicBlock, [2, 2, 2, 2])
        self.branch2 = BaseResNetBranch(BasicBlock, [2, 2, 2, 2])
        self.fc1 = nn.Linear(512, 1220 * 3)
        self.fc2 = nn.Linear(512, 6)

    def forward(self, img_local, img_global):
        out1 = self.branch1(img_local)
        out2 = self.branch2(img_global)
        pred1 = self.fc1(out1)
        pred2 = self.fc2(out2)
        return pred1, pred2





class SimpleDualResNetModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--loss_verts_weight', type=float, default=5.0, help='weight for loss_verts.')
        parser.add_argument('--loss_6dof_weight', type=float, default=1.0, help='weight for loss_6dof.')
        return parser



    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['total', 'verts', '6dof']
        self.model_names = ['R']

        self.netR = networks.init_net(
            DualResNet(),
            gpu_ids=self.gpu_ids
        )
        self.criterion = torch.nn.L1Loss()

        if self.isTrain:
            self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_R)



    def set_input(self, input):
        self.img_local = input['img_local'].to(self.device)
        self.img_global = input['img_global'].to(self.device)
        if not self.opt.use_test_set:
            self.label_verts = input['label_verts'].view(-1, 1220 * 3).to(self.device)
            self.label_6dof = input['label_6dof'].to(self.device)



    def forward(self):
        self.pred_verts, self.pred_6dof = self.netR(self.img_local, self.img_global)



    def backward_R(self):
        self.loss_verts = self.criterion(self.pred_verts, self.label_verts) * self.opt.loss_verts_weight
        self.loss_6dof = self.criterion(self.pred_6dof, self.label_6dof) * self.opt.loss_6dof_weight
        self.loss_total = self.loss_verts + self.loss_6dof
        self.loss_total.backward()



    def optimize_parameters(self):
        self.netR.train()
        self.forward()

        self.optimizer_R.zero_grad()
        self.backward_R()
        self.optimizer_R.step()



    def init_evaluation(self):
        self.inference_data = {
            'loss_verts': [], 'loss_6dof': [], 'loss_total': [],
            'batch_size': [],
            'pitch_mae': [], 'yaw_mae': [], 'roll_mae': [],
            'tx_mae': [], 'ty_mae': [], 'tz_mae': [],
            '3DRecon': []
        }
        self.netR.eval()



    def inference_curr_batch(self):
        with torch.no_grad():
            self.forward()

        bs = self.img_local.size(0)
        cur_loss_verts = self.criterion(self.pred_verts, self.label_verts).item() * self.opt.loss_verts_weight
        cur_loss_6dof = self.criterion(self.pred_6dof, self.label_6dof).item() * self.opt.loss_6dof_weight
        cur_loss_total = cur_loss_verts + cur_loss_6dof
        self.inference_data['loss_verts'].append(cur_loss_verts)
        self.inference_data['loss_6dof'].append(cur_loss_6dof)
        self.inference_data['loss_total'].append(cur_loss_total)
        self.inference_data['batch_size'].append(bs)



    def compute_metrics(self):
        bs_list = np.array(self.inference_data['batch_size'])
        loss_verts_list = np.array(self.inference_data['loss_verts'])
        loss_6dof_list = np.array(self.inference_data['loss_6dof'])
        loss_total_list = np.array(self.inference_data['loss_total'])

        loss_verts = (loss_verts_list * bs_list).sum() / bs_list.sum()
        loss_6dof = (loss_6dof_list * bs_list).sum() / bs_list.sum()
        loss_total = (loss_total_list * bs_list).sum() / bs_list.sum()
        metrics = {
            'loss_total': np.around(loss_total, decimals=6),
            'loss_verts': np.around(loss_verts, decimals=6),
            'loss_6dof': np.around(loss_6dof, decimals=6)
        }
        return metrics




