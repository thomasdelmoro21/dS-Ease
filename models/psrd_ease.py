import logging
import numpy as np
import torch
import itertools
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import PSRDEaseNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
import copy

from psrd.losses import SupConLoss

num_workers = 8
    

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = PSRDEaseNet(args, True)
        
        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]
        if self.inc != self.init_cls:
            raise ValueError("Increment should be equal to init_cls fro running PSRD-EASE")

        self.use_exemplars = args["use_old_data"]
        self.use_init_ptm = args["use_init_ptm"]
        self.use_diagonal = args["use_diagonal"]
        
        self.recalc_sim = args["recalc_sim"]
        self.alpha = args["alpha"] # forward_reweight is divide by _cur_task
        self.beta = args["beta"]

        self.moni_adam = args["moni_adam"]
        self.adapter_num = args["adapter_num"]

        self.supcon_loss = SupConLoss(temperature=self.args["supcon_temperature"], device=self._device)
        self.first_task_id = 0
        if self.args["half_iid"]:
            self.first_task_id = (self.args["num_tasks"] // 2) - 1

        self.prototypes_coef = self.args["prototypes_coef"]
        self.distill_coef = self.args["distill_coef"]
        self.distill_temp = self.args["distill_temp"]

    def after_task(self):
        self.freeze_prev_model()

        self._known_classes = self._total_classes
        #self._network.freeze()
        self._network.backbone.add_adapter_to_list()

    def freeze_prev_model(self):
        self._network.backbone.freeze()
        for i in range(self._cur_task + 1):
            self._network.prototypes.heads[str(i)].weight.requires_grad = False
    
    def get_cls_range(self, task_id):
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc
        
        return start_cls, end_cls
    
    def _distillation_loss(self, current_out: torch.FloatTensor, prev_out: torch.FloatTensor) -> torch.FloatTensor:

        log_p = torch.log_softmax(current_out / self.distill_temp, dim=1)  # student
        q = torch.softmax(prev_out / self.distill_temp, dim=1)  # teacher
        result = torch.nn.KLDivLoss(reduction="batchmean")(log_p, q)
        return result

    def relation_distillation_loss(  # Ld
        self, features: torch.FloatTensor, data: torch.FloatTensor, current_task_id: int) -> torch.FloatTensor:

        if self._cur_task == 0:
            return 0.0

        old_model_preds = dict()
        new_model_preds = dict()

        with torch.inference_mode():
            old_features = self._network(data, test=True, prev_model=True)["features"]

        for task_id in range(self.first_task_id, current_task_id):
            with torch.inference_mode():
                old_model_preds[task_id] = self._get_scores(
                    old_features[:, task_id * self._network.out_dim : (task_id+1) * self._network.out_dim], prototypes=self._network.prototypes, task_id=task_id
                )

            new_model_preds[task_id] = self._get_scores(
                features, prototypes=self._network.prototypes, task_id=task_id
            )

        dist_loss = 0
        for task_id in old_model_preds.keys():
            dist_loss += self._distillation_loss(
                current_out=new_model_preds[task_id],
                prev_out=old_model_preds[task_id].clone(),
            )

        return dist_loss
    
    def _get_scores(self, features: torch.FloatTensor, prototypes, task_id: int) -> torch.FloatTensor:
        nobout = F.linear(features, prototypes.heads[str(task_id)].weight)
        wnorm = torch.norm(prototypes.heads[str(task_id)].weight, dim=1, p=2)
        nobout = nobout / wnorm
        return nobout
    
    def linear_loss(  # Lp
        self,
        features: torch.FloatTensor,
        labels: torch.Tensor,
        current_task_id: int,
        lam: int = 1,
    ) -> torch.FloatTensor:

        if lam == 0:
            features = features.detach().clone()  # [0:labels.size(0)]

        nobout = F.linear(features, self._network.prototypes.heads[str(current_task_id)].weight)
        wnorm = torch.norm(
            self._network.prototypes.heads[str(current_task_id)].weight, dim=1, p=2
        )
        nobout = nobout / wnorm
        feat_norm = torch.norm(features, dim=1, p=2)

        if not current_task_id == self.first_task_id:
            labels -= current_task_id * self.inc  # shift targets
        indecies = labels.unsqueeze(1)
        out = nobout.gather(1, indecies).squeeze()
        out = out / feat_norm
        loss = sum(1 - out) / out.size(0)

        return loss

    def _prototypes_contrast_loss(self, task_id: int):
        # anchor = self.prototypes.heads[str(task_id)].weight.
        contrast_prot = []
        for key, head in self._network.prototypes.heads.items():
            if int(key) < task_id:
                contrast_prot.append(copy.deepcopy(head).weight.data)

        if len(contrast_prot) == 0:
            return 0.0

        contrast_prot = F.normalize(torch.cat(contrast_prot, dim=-1), dim=1, p=2)
        anchors = F.normalize(
            self._network.prototypes.heads[str(task_id)].weight.data, dim=1, p=2
        )

        logits = torch.div(
            torch.matmul(anchors.T, contrast_prot), self.args.supcon_temperature
        )
        log_prob = torch.log(torch.exp(logits).sum(1))
        loss = -log_prob.sum() / log_prob.size(0)

        return loss


    def stable_rep(self, out0, out1, labels, logit_scale):
        device = out0.device

        # compute the mask based on labels
        mask = torch.eq(labels.view(-1, 1),
                        labels.contiguous().view(1, -1)).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
        mask = mask * logits_mask

        # compute logits
        logits = torch.matmul(out0, out1.T) * logit_scale
        logits = logits - (1 - logits_mask) * 1e9

        # optional: minus the largest logit to stabilize logits
        logits = self.stablize_logits(logits)

        # compute ground-truth distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        loss = self.compute_cross_entropy(p, logits)

        return loss

    def compute_cross_entropy(self, p, q):
        q = F.log_softmax(q, dim=-1)
        loss = torch.sum(p * q, dim=-1)
        return - loss.mean()

    def stablize_logits(self, logits):
        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits = logits - logits_max.detach()
        return logits


    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._all_classes = data_manager.nb_classes
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        # self._network.show_trainable_params()
        
        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train", )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        
        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        self.all_cls_test_dataset = data_manager.get_dataset(np.arange(0, self._all_classes), source="test", mode="test" )
        self.all_cls_test_loader = DataLoader(self.all_cls_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        
        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        
        if self._cur_task == 0 or self.init_cls == self.inc:
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])
        else:
            # for base 0 setting, the later_lr and later_epochs are not used
            # for base N setting, the later_lr and later_epochs are used
            if "later_lr" not in self.args or self.args["later_lr"] == 0:
                self.args["later_lr"] = self.args["init_lr"]
            if "later_epochs" not in self.args or self.args["later_epochs"] == 0:
                self.args["later_epochs"] = self.args["init_epochs"]

            optimizer = self.get_optimizer(lr=self.args["later_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["later_epochs"])

        self._init_train(train_loader, test_loader, optimizer, scheduler)
    
    def get_optimizer(self, lr):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                momentum=0.9, 
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr, 
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr, 
                weight_decay=self.weight_decay
            )

        return optimizer
    
    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        if self.moni_adam:
            if self._cur_task > self.adapter_num - 1:
                return
        
        if self._cur_task == 0:
            epochs = self.args['init_epochs']
        else:
            epochs = self.args['later_epochs']
        
        prog_bar = tqdm(range(epochs))
            
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0., 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                x1, x2 = inputs.to(self._device), inputs.to(self._device)
                targets = targets.to(self._device)
                aug_data = torch.cat((x1, x2), dim=0)
                bsz = x1.size(0)

                out = self._network(aug_data, test=False)
                features = out["features"]
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)

                # Supervised Contrastive Loss
                proj_features = out["proj_features"]
                proj_features = F.normalize(proj_features, dim=1)
                proj_f1, proj_f2 = torch.split(proj_features, [bsz, bsz], dim=0)
                proj_features = torch.cat([proj_f1.unsqueeze(1), proj_f2.unsqueeze(1)], dim=1)
                #supcon_loss = self.supcon_loss(proj_features, labels=targets)
                logit_scale = 1 / 0.07
                supcon_loss = self.stable_rep(proj_f1, proj_f2, targets, logit_scale)  # Stable Rep

                # Distillation Loss
                loss_d = self.relation_distillation_loss(features, data=aug_data, current_task_id=self._cur_task)

                # Prototypes Loss
                loss_p = self.linear_loss(features.detach().clone(), labels=targets.repeat(2), current_task_id=self._cur_task)

                # Cross Entropy Loss
                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes >= 0,
                    aux_targets - self._known_classes,
                    -1,
                )
                logits = self._network.prototypes.heads[str(self._cur_task)](f1)
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(targets)

                cross_entropy_loss = F.cross_entropy(logits, aux_targets)

                loss = supcon_loss + self.prototypes_coef * loss_p + self.distill_coef * loss_d
                # loss = cross_entropy_loss + self.prototypes_coef * loss_p + self.distill_coef * loss_d

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                print(f"Epoch: {epoch + 1} / {epochs} | {i} / {len(train_loader)} - Loss: {loss}", end="\r",)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def predict_task(self, features, task_id):
        self._network.eval()
        return self._network.predict_task(features, task_id)
    
    def predict(self, features, task_id):
        self._network.eval()
        return self._network.predict(features, task_id)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.forward(inputs, test=True)["logits"]
            predicts = torch.max(outputs, dim=1)[1]          
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader, calc_task_acc=True):
        
        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0
            
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                features = self._network(inputs, test=True)["features"]
                outputs = self.predict(features, self._cur_task)
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            
            # calculate the accuracy by using task_id
            if calc_task_acc:
                task_ids = (targets - self.init_cls) // self.inc + 1
                task_logits = torch.zeros(outputs.shape).to(self._device)
                for i, task_id in enumerate(task_ids):
                    if task_id == 0:
                        start_cls = 0
                        end_cls = self.init_cls
                    else:
                        start_cls = self.init_cls + (task_id-1)*self.inc
                        end_cls = self.init_cls + task_id*self.inc
                    task_logits[i, start_cls:end_cls] += outputs[i, start_cls:end_cls]
                # calculate the accuracy of task_id
                pred_task_ids = (torch.max(outputs, dim=1)[1] - self.init_cls) // self.inc + 1
                task_correct += (pred_task_ids.cpu() == task_ids).sum()
                
                pred_task_y = torch.max(task_logits, dim=1)[1]
                task_acc += (pred_task_y.cpu() == targets).sum()
                total += len(targets)

        if calc_task_acc:
            logging.info("Task correct: {}".format(tensor2numpy(task_correct) * 100 / total))
            logging.info("Task acc: {}".format(tensor2numpy(task_acc) * 100 / total))
                
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def eval_transfer(self):
        self._network.eval()
        print("Evaluating backward and forward transfer performance...")
        accys = dict()
        for task in range(0, self.data_manager.nb_tasks):
            offset1 = self.init_cls + (task - 1) * self.inc
            offset2 = self.init_cls + task * self.inc
            task_labels = np.arange(offset1, offset2)
            task_dataset = self.data_manager.get_dataset(task_labels, source="test", mode="test")
            task_loader = DataLoader(task_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

            predictions, gts = torch.tensor([]), torch.tensor([])
            for _, (_, inputs, targets) in enumerate(task_loader):
                inputs = inputs.to(self._device)
                
                with torch.no_grad():
                    outputs = self._network.forward(inputs, test=True)["logits"]
                    if offset1 > 0:
                        outputs[:, :offset1].data.fill_(-10e10)
                    if offset2 < self._all_classes:
                        outputs[:, offset2:self._all_classes].data.fill_(-10e10)
                preds = torch.max(outputs, dim=1)[1]

                predictions = torch.cat((predictions, preds.cpu()), dim=0)
                gts = torch.cat((gts, targets), dim=0)

            acc = (predictions == gts).sum() * 100 / len(gts)
            accys["{}-{}".format(offset1, offset2)] = acc.item()
        return accys