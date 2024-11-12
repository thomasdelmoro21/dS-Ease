import logging
import numpy as np
import copy
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import EaseNet, SimplexEaseNet, HocSimplexEaseNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, count_parameters, l2_norm

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = HocSimplexEaseNet(args, True)
        
        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        self.use_exemplars = args["use_old_data"]
        self.use_init_ptm = args["use_init_ptm"]
        self.use_diagonal = args["use_diagonal"]
        
        self.recalc_sim = args["recalc_sim"]
        self.alpha = args["alpha"] # forward_reweight is divide by _cur_task
        self.beta = args["beta"]

        self.moni_adam = args["moni_adam"]
        self.adapter_num = args["adapter_num"]
        
        # HOC D-SIMPLEX ATTRIBUTES
        self._lambda = args["lambda"]
        self._mu = args["mu"]
        self.previous_network = None
        self.model_checkpoint_path = args["model_checkpoint_path"]

    def after_task(self):
        self._known_classes = self._total_classes
        self._network.freeze()
        self._network.backbone.add_adapter_to_list()
    
    def get_cls_range(self, task_id):
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc
        
        return start_cls, end_cls
        
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._all_classes = data_manager.nb_classes
        #self._network.update_fc(self._total_classes)

        # Load previous network weights
        if self._cur_task > 0:
            self.previous_network = torch.load(self.model_checkpoint_path)
            for param in self.previous_network.parameters():
                param.requires_grad = False
            print("Previous Network parameters: {}".format(count_parameters(self.previous_network)))

        self._network.expand_junction()  # Add a junction layer after the current adapter and increment current task id
            
        print("Current Network parameters: {} \n".format(count_parameters(self._network)))
        logging.info("Total trainable params: {}".format(count_parameters(self._network, True)))
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        # self._network.show_trainable_params()
        
        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train", )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        
        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        
        self.all_cls_test_dataset = data_manager.get_dataset(np.arange(0, self._all_classes), source="test", mode="test" )
        self.all_cls_test_loader = DataLoader(self.all_cls_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        #self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        #self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        #self.replace_fc(self.train_loader_for_protonet)

        # Save Model Checkpoint
        torch.save(self._network, self.model_checkpoint_path)

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
        self._network.update_junctions()
        for param in self._network.backbone.cur_adapter.parameters():
            param.requires_grad = False
        
        if self._cur_task > 0:
            opt_jun = optim.SGD(self._network.junction.parameters(), lr=self.args["hoc_lr"], momentum=0.9, weight_decay=self.weight_decay)
            self._train_junction(train_loader, test_loader, opt_jun)
    
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
        
        if self._cur_task == 0 or self.init_cls == self.inc:
            epochs = self.args['init_epochs']
        else:
            epochs = self.args['later_epochs']
        
        prog_bar = tqdm(range(epochs))
            
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                aux_targets = targets.clone()
                '''
                aux_targets = torch.where(
                    aux_targets - self._known_classes >= 0,
                    aux_targets - self._known_classes,
                    -1,
                )
                '''
                # Train adapter with proxy junction
                output = self._network(inputs, hoc=False)
                logits = output["logits"]
                loss = F.cross_entropy(logits, aux_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)

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

    def _train_junction(self, train_loader, test_loader, optimizer):
        epochs = self.args['hoc_epochs']

        print("Trainable params junction: {}".format(count_parameters(self._network.junction, trainable=True)))
        
        prog_bar = tqdm(range(epochs))

        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                aux_targets = targets.clone()
                '''
                aux_targets = torch.where(
                    aux_targets - self._known_classes >= 0,
                    aux_targets - self._known_classes,
                    -1,
                )
                '''
                # Train junction layer with hoc loss
                output = self._network(inputs, hoc=True)
                logits = output["logits"]
                loss = F.cross_entropy(logits, aux_targets)
                
                if self.previous_network is not None:
                    with torch.no_grad():
                        feature_old = self.previous_network(inputs, hoc=True)["features"]
                    norm_feature_old = l2_norm(feature_old)
                    norm_feature_new = l2_norm(output["features"])
                    hoc = HocLoss(self._mu)
                    loss_feat = hoc(norm_feature_new, norm_feature_old, aux_targets)
                    loss = loss * self._lambda + (1 - self._lambda) * loss_feat
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Junction Layer Train: Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)


    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.forward(inputs, hoc=True)["logits"]
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
                outputs = self._network.forward(inputs, hoc=True)["logits"]
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


class HocLoss(nn.Module):
    def __init__(self, mu_):
        super(HocLoss, self).__init__()
        self.mu_ = mu_
    
    def forward(self, 
                feat_new, 
                feat_old, 
                labels, 
               ):
        loss = self._loss(feat_old, feat_new, labels)
        return loss

    def _loss(self, out0, out1, labels):
        """Calculates infoNCE loss.
        This code implements Equation 4 in "Stationary Representations: Optimally Approximating Compatibility and Implications for Improved Model Replacements"

        Args:
            feat_old:
                features extracted with the old model.
                Shape: (batch_size, embedding_size)
            feat_new:
                features extracted with the new model.
                Shape: (batch_size, embedding_size)
            labels:
                Labels of the images.
                Shape: (batch_size,)
        Returns:
            Mean loss over the mini-batch.
        """
        ## create diagonal mask that only selects similarities between
        ## representations of the same images
        batch_size = out0.shape[0]
        diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)
        sim_01 = torch.einsum("nc,mc->nm", out0, out1) *  self.mu_

        positive_loss = -sim_01[diag_mask]
        # Get the labels of out0 and out1 samples
        labels_0 = labels.unsqueeze(1).expand(-1, batch_size)  # Shape: (batch_size, batch_size)
        labels_1 = labels.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, batch_size)

        # Mask similarities between the same class
        class_mask = labels_0 == labels_1
        sim_01 = (sim_01* (~class_mask)).view(batch_size, -1)

        negative_loss_01 = torch.logsumexp(sim_01, dim=1)
        return (positive_loss + negative_loss_01).mean()