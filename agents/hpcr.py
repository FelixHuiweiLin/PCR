import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, transforms_aug
from utils.utils import maybe_cuda
from utils.loss import SupConLoss
import numpy as np
from torch.nn import functional as F

class HolisticProxyContrastiveReplay(ContinualLearner):
    """
        Holistic Proxy-based Contrastive Replay,
        Implements the strategy defined in
        "HPCR: Holistic Proxy-based Contrastive Replay for Online Continual Learning"
        https://arxiv.org/pdf/2309.15038

        This strategy has been designed and tested in the
        Online Setting (OnlineCLScenario). However, it
        can also be used in non-online scenarios
        """
    def __init__(self, model, opt, params):
        super(HolisticProxyContrastiveReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.max_temp = params.hpcr_temp_max
        self.min_temp = params.hpcr_temp_min
        self.PCD_alpha = params.hpcr_PCD_alpha 
        self.SCD_beta = params.hpcr_SCD_beta 

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()
        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                tau_s = (self.max_temp - self.min_temp) * 0.5 * (1 - np.cos(2 * np.pi * i / 500)) + self.min_temp
                # batch update
                batch_x, batch_y = batch_data
                batch_x_aug = torch.stack([transforms_aug[self.data](batch_x[idx].cpu())
                                           for idx in range(batch_x.size(0))])
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_x_aug = maybe_cuda(batch_x_aug, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                batch_x_combine = torch.cat((batch_x, batch_x_aug))
                batch_y_combine = torch.cat((batch_y, batch_y))
                for j in range(self.mem_iters):
                    logits, feas= self.model.pcrForward(batch_x_combine)
                    novel_loss = 0*self.criterion(logits, batch_y_combine)
                    save_logits = logits[0:self.params.batch,:].detach()
                    save_features = feas[0:self.params.batch,:].detach()
                    self.opt.zero_grad()
                    mem_x, mem_y, last_logits, last_feas, _, _ = self.buffer.retrieve(x=batch_x, y=batch_y)
                    if mem_x.size(0) > 0:
                        mem_x_aug = torch.stack([transforms_aug[self.data](mem_x[idx].cpu())
                                                 for idx in range(mem_x.size(0))])
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        mem_x_combine = torch.cat([mem_x, mem_x_aug])
                        mem_y_combine = torch.cat([mem_y, mem_y])
                        mem_logits, mem_fea= self.model.pcrForward(mem_x_combine)
                        #-------------L_PCRCT----------------
                        combined_feas = torch.cat([mem_fea, feas])
                        combined_labels = torch.cat((mem_y_combine, batch_y_combine))
                        combined_feas_aug = self.model.pcrLinear.L.weight[combined_labels]
                        combined_feas_norm = torch.norm(combined_feas, p=2, dim=1).unsqueeze(1).expand_as(combined_feas)
                        combined_feas_normalized = combined_feas.div(combined_feas_norm + 0.000001)
                        combined_feas_aug_norm = torch.norm(combined_feas_aug, p=2, dim=1).unsqueeze(1).expand_as(
                            combined_feas_aug)
                        combined_feas_aug_normalized = combined_feas_aug.div(combined_feas_aug_norm + 0.000001)
                        cos_features = torch.cat([combined_feas_normalized.unsqueeze(1),
                                                  combined_feas_aug_normalized.unsqueeze(1)],
                                                 dim=1)
                        PSC = SupConLoss(temperature=self.params.pcr_temp, contrast_mode=self.params.pcr_type)
                        novel_loss += PSC(features=cos_features, labels=combined_labels)
                        novel_loss *= tau_s
                        
                        if self.i>0:
                        #-------------L_PCD----------------
                            temp_logits = torch.cat([last_logits, last_logits]).cuda()
                            mem_logits = mem_logits[:, combined_labels]
                            temp_logits = temp_logits[:, combined_labels]
                            distilled_loss = self.PCD_alpha * F.mse_loss(mem_logits, temp_logits)
                            novel_loss += distilled_loss
                        #-------------L_SCD----------------
                            temp_features = torch.cat([last_feas, last_feas]).cuda()
                            temp_features_norm = torch.norm(temp_features, p=2, dim=1).unsqueeze(1).expand_as(
                                temp_features)
                            temp_features = temp_features.div(temp_features_norm + 0.000001)
                            mem_fea_norm = torch.norm(mem_fea, p=2, dim=1).unsqueeze(1).expand_as(
                                mem_fea)
                            mem_fea = mem_fea.div(mem_fea_norm + 0.000001)
                            new_sim = torch.div(torch.matmul(mem_fea, mem_fea.T),self.params.pcr_temp)
                            logits_mask = torch.scatter(
                                torch.ones_like(new_sim),
                                1,
                                torch.arange(new_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                                0
                            )
                            logits_max1, _ = torch.max(new_sim * logits_mask, dim=1, keepdim=True)
                            new_sim = new_sim - logits_max1.detach()
                            row_size = new_sim.size(0)
                            logits_new = torch.exp(new_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
                                new_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
                        
                            old_sim = torch.div(torch.matmul(temp_features, temp_features.T),self.params.pcr_temp)
                            logits_max2, _ = torch.max(old_sim * logits_mask, dim=1, keepdim=True)
                            old_sim = old_sim - logits_max2.detach()
                            logits_old = torch.exp(old_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
                                old_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
                        
                            structure_loss = self.SCD_beta * (-logits_old * torch.log(logits_new)).sum(1).mean()
                            novel_loss += structure_loss

                    novel_loss.backward()
                    self.opt.step()
                # update mem
                self.buffer.update(batch_x, batch_y, save_logits, save_features)

        self.after_train()