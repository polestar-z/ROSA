import dgl
import torch
from tqdm import tqdm
from ..models import build_model
from . import BaseFlow, register_flow
from ..utils import EarlyStopping, to_hetero_idx, to_homo_feature, to_homo_idx
from ..utils.loss import AsymmetricLoss
import warnings
from torch.utils.tensorboard import SummaryWriter
import dgl.graphbolt as gb
import tkinter as tk
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import os

@register_flow("rosa_node_classification")
class ROSANodeClassification(BaseFlow):
    r"""
    Node classification flow.
    """

    def __init__(self, args):
        super(ROSANodeClassification, self).__init__(args)
        
        self.args.category = self.task.dataset.category
        self.category = self.args.category
        self.num_classes = self.task.dataset.num_classes

        if not hasattr(self.task.dataset, 'out_dim') or args.out_dim != self.num_classes:
            args.out_dim = self.num_classes
        self.args.out_node_type = [self.category]
        ds = self.task.dataset
        
        if hasattr(ds, "meta_paths_dict") and ds.meta_paths_dict is not None:
            self.args.meta_paths_dict = ds.meta_paths_dict
        elif hasattr(self.hg, "meta_paths_dict"):
            self.args.meta_paths_dict = self.hg.meta_paths_dict
        else:
            raise ValueError("No meta_paths_dict found on dataset or graph!")

        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)
        self.use_distributed = args.use_distributed
        if self.use_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.device], output_device=self.device, find_unused_parameters=True
            )

        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(),
                                                                  lr=args.lr, weight_decay=args.weight_decay)

        self.train_idx, self.val_idx, self.test_idx = self.task.get_split()
        self.pred_idx = getattr(self.task.dataset, 'pred_idx', None)

        self.labels = self.task.get_labels().to(self.device)
        self.num_nodes_dict = {ntype: self.hg.num_nodes(ntype) for ntype in self.hg.ntypes}
        self.to_homo_flag = getattr(self.model, 'to_homo_flag', False)
        self.writer = SummaryWriter(f'./openhgnn/output/{self.model_name}/')

        self.is_multi_label = getattr(args, 'multi_label', False)
        self.threshold = getattr(args, 'threshold', 0.5)

        if self.is_multi_label:
            gamma_neg = getattr(args, 'asl_gamma_neg', 4)
            gamma_pos = getattr(args, 'asl_gamma_pos', 1)
            clip = getattr(args, 'asl_clip', 0.05)

            if not isinstance(self.labels, torch.FloatTensor) and self.labels.dtype != torch.float32:
                raise TypeError(f"Labels must be FloatTensor for multi-label mode.")

            self.loss_fn = AsymmetricLoss(
                gamma_neg=gamma_neg,
                gamma_pos=gamma_pos,
                clip=clip
            )

            self.use_resampling = False
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.use_resampling = getattr(args, 'use_resampling', True)
            self.resampling_strategy = getattr(args, 'resampling_strategy', 'balanced')

            if self.use_resampling:
                self._setup_singlelabel_resampling()

        self.lambda_head_l1 = getattr(args, 'lambda_head_l1', 0.0)

        self.use_consistency_loss = getattr(args, 'use_consistency_loss', False)
        self.lambda_consistency = getattr(args, 'consistency_loss_weight', 0.1)

        if self.to_homo_flag:
            self.g = dgl.to_homogeneous(self.hg)

        if self.args.mini_batch_flag and self.model_name.upper() == 'ROSA':
            self.args.mini_batch_flag = False

        if self.args.mini_batch_flag:
            if not hasattr(args, 'fanout'):
                warnings.warn("please set fanout when using mini batch training.")
                args.fanout = -1
            if isinstance(args.fanout, list):
                self.fanouts = args.fanout
            else:
                self.fanouts = [args.fanout] * self.args.num_layers
            sampler = dgl.dataloading.MultiLayerNeighborSampler(self.fanouts)
            use_uva = self.args.use_uva

            if self.to_homo_flag:
                loader_g = self.g
            else:
                loader_g = self.hg

            if self.train_idx is not None:
                if self.to_homo_flag:
                    loader_train_idx = to_homo_idx(self.hg.ntypes, self.num_nodes_dict,
                                                {self.category: self.train_idx}).to(self.device)
                else:
                    loader_train_idx = {self.category: self.train_idx.to(self.device)}
                self.args.batch_size = getattr(self.args, 'batch_size', 64)
                self.train_loader = dgl.dataloading.DataLoader(loader_g, loader_train_idx, sampler,
                                                            batch_size=self.args.batch_size, device=self.device,
                                                            shuffle=True, use_uva=use_uva, use_ddp=self.use_distributed)
            if self.valid_idx is not None:
                if self.to_homo_flag:
                    loader_val_idx = to_homo_idx(self.hg.ntypes, self.num_nodes_dict, {self.category: self.val_idx}).to(
                        self.device)
                else:
                    loader_val_idx = {self.category: self.val_idx.to(self.device)}
                self.val_loader = dgl.dataloading.DataLoader(loader_g, loader_val_idx, sampler,
                                                            batch_size=self.args.batch_size, device=self.device,
                                                            shuffle=True, use_uva=use_uva)
            if self.args.test_flag:
                if self.test_idx is not None:
                    if self.to_homo_flag:
                        loader_test_idx = to_homo_idx(self.hg.ntypes, self.num_nodes_dict,
                                                    {self.category: self.test_idx}).to(self.device)
                    else:
                        loader_test_idx = {self.category: self.test_idx.to(self.device)}
                    self.test_loader = dgl.dataloading.DataLoader(loader_g, loader_test_idx, sampler,
                                                                batch_size=self.args.batch_size, device=self.device,
                                                                shuffle=True, use_uva=use_uva)
            if self.args.prediction_flag:
                if self.pred_idx is not None:
                    if self.to_homo_flag:
                        loader_pred_idx = to_homo_idx(self.hg.ntypes, self.num_nodes_dict,
                                                    {self.category: self.pred_idx}).to(self.device)
                    else:
                        loader_pred_idx = {self.category: self.pred_idx.to(self.device)}
                    self.pred_loader = dgl.dataloading.DataLoader(loader_g, loader_pred_idx, sampler,
                                                                batch_size=self.args.batch_size, device=self.device,
                                                                shuffle=True, use_uva=use_uva)

        def create_loader(Item_set,graph):
            datapipe = gb.ItemSampler(Item_set, batch_size=self.args.batch_size, shuffle=True)
            datapipe = datapipe.copy_to(self.device)
            datapipe = datapipe.sample_neighbor(graph, self.fanouts)
            return gb.DataLoader(datapipe)
        
        if self.args.mini_batch_flag and self.args.graphbolt:
            dataset = gb.OnDiskDataset(self.task.dataset_GB.base_dir).load()
            graph = dataset.graph.to(self.device)
            tasks = dataset.tasks
            nc_task = tasks[0]
            self.train_GB_loader = create_loader(nc_task.train_set, graph)
            self.val_GB_loader = create_loader(nc_task.validation_set, graph)
            self.test_GB_loader = create_loader(nc_task.test_set, graph)

    def _setup_singlelabel_resampling(self):
        import numpy as np
        train_labels = self.labels[self.train_idx].cpu().numpy()
        unique_classes, class_counts = np.unique(train_labels, return_counts=True)

        if self.resampling_strategy == 'oversample':
            max_count = class_counts.max()
            class_weights = {cls: max_count / count for cls, count in zip(unique_classes, class_counts)}
        elif self.resampling_strategy == 'balanced':
            from sklearn.utils.class_weight import compute_class_weight
            class_weight_array = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
            class_weights = {cls: weight for cls, weight in zip(unique_classes, class_weight_array)}
        else:
            raise ValueError(f"Unsupported resampling strategy: {self.resampling_strategy}")

        sample_weights = np.array([class_weights[label] for label in train_labels])
        self.sample_weights = torch.from_numpy(sample_weights).float()

        if self.resampling_strategy == 'oversample':
            self.resampled_train_size = int(max_count * len(unique_classes))
        else:
            self.resampled_train_size = len(self.train_idx)

    def _get_resampled_train_idx(self):
        if self.is_multi_label:
            return self.train_idx

        if not self.use_resampling or not hasattr(self, 'sample_weights'):
            return self.train_idx

        from torch.utils.data import WeightedRandomSampler

        sampler = WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=self.resampled_train_size,
            replacement=True
        )

        resampled_indices = list(sampler)
        resampled_train_idx = self.train_idx[resampled_indices]

        return resampled_train_idx

    def preprocess(self):
        super(ROSANodeClassification, self).preprocess()

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter = tqdm(range(self.max_epoch))

        import time
        for epoch in epoch_iter:
            epoch_start = time.time()

            train_start = time.time()
            if self.args.mini_batch_flag:
                train_loss = self._mini_train_step()
            else:
                train_loss = self._full_train_step()
            train_time = time.time() - train_start

            if epoch % self.evaluate_interval == 0:
                eval_start = time.time()
                modes = ['train', 'valid']
                if self.args.test_flag:
                    modes = modes + ['test']
                if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                    metric_dict, losses = self._mini_test_step(modes=modes)
                else:
                    metric_dict, losses = self._full_test_step(modes=modes)
                val_loss = losses['valid']
                eval_time = time.time() - eval_start

                self.logger.train_info(
                    f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Valid loss: {val_loss:.4f}. " +
                    self.logger.metric2str(metric_dict)
                )
                
                early_stop = stopper.loss_step(val_loss, self.model)
                if early_stop:
                    self.logger.train_info('Early Stop!\tEpoch:' + str(epoch))
                    break

        stopper.load_model(self.model)

        checkpoint_dir = './checkpoint'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir,
                                      f'{self.model_name}_{self.args.dataset}_best.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        self.logger.train_info(f'Best model saved to: {checkpoint_path}')

        if self.args.prediction_flag:
            if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                indices, y_predicts = self._mini_prediction_step()
            else:
                y_predicts = self._full_prediction_step()
                indices = torch.arange(self.hg.num_nodes(self.category))
            return indices, y_predicts

        if self.args.test_flag:
            if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                metric_dict, _ = self._mini_test_step(modes=['valid', 'test'])
            else:
                metric_dict, _ = self._full_test_step(modes=['valid', 'test'])
            self.logger.train_info('[Test Info]' + self.logger.metric2str(metric_dict))
            return dict(metric=metric_dict, epoch=epoch)

    def _full_train_step(self):
        self.model.train()
        h_dict = self.model.input_feature()
        self.hg = self.hg.to(self.device)

        train_idx = self._get_resampled_train_idx() if self.use_resampling else self.train_idx

        if self.use_consistency_loss and hasattr(self.model, 'consistency_loss') and self.model.consistency_loss is not None:
            logits_dict, h_coa_dict, h_aoa_dict = self.model(self.hg, h_dict, return_branch_features=True)
            logits = logits_dict[self.category]

            task_loss = self.loss_fn(logits[train_idx], self.labels[train_idx])

            train_mask = {self.category: train_idx}
            consistency_loss = self.model.consistency_loss(h_coa_dict, h_aoa_dict, mask=train_mask)

            total_loss = task_loss + self.lambda_consistency * consistency_loss
        else:
            logits = self.model(self.hg, h_dict)[self.category]
            task_loss = self.loss_fn(logits[train_idx], self.labels[train_idx])
            total_loss = task_loss

        if hasattr(self, 'lambda_head_l1') and self.lambda_head_l1 > 0:
            from ..models.attention_pruning import collect_head_importance_losses
            head_l1_loss = collect_head_importance_losses(self.model)
            total_loss = total_loss + self.lambda_head_l1 * head_l1_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
    
    def _mini_train_step(self):
        if getattr(self.args, "graphbolt", False):
            self.model.train()
            loss_all = 0.0
            for i, data in enumerate(self.train_GB_loader):
                input_nodes = data.input_nodes
                seeds = data.seeds
                for key in input_nodes:
                    input_nodes[key] = input_nodes[key].to(self.device)
                emb = self.model.input_feature.forward_nodes(input_nodes)
                label = data.labels[self.category].to(self.device)
                logits = self.model(data.blocks, emb)[self.category]
                loss = self.loss_fn(logits, label)
                loss_all += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return loss_all / (i + 1)

        else:
            self.model.train()
            loss_all = 0.0
            loader_tqdm = tqdm(self.train_loader, ncols=120)
            for i, (input_nodes, seeds, blocks) in enumerate(loader_tqdm):
                if self.to_homo_flag:
                    seeds = to_hetero_idx(self.g, self.hg, seeds)
                elif isinstance(input_nodes, dict):
                    for key in input_nodes:
                        input_nodes[key] = input_nodes[key].to(self.device)
                emb = self.model.input_feature.forward_nodes(input_nodes)
                lbl = self.labels[seeds[self.category]].to(self.device)
                logits = self.model(blocks, emb)[self.category]
                loss = self.loss_fn(logits, lbl)
                loss_all += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return loss_all / (i + 1)

    def _full_test_step(self, modes, logits=None):
        self.model.eval()
        with torch.no_grad():
            h_dict = self.model.input_feature()
            h_dict = {k: e.to(self.device) for k, e in h_dict.items()}
            logits = logits if logits else self.model(self.hg, h_dict)[self.category]

            masks = {}
            for mode in modes:
                if mode == "train":
                    masks[mode] = self.train_idx
                elif mode == "valid":
                    masks[mode] = self.val_idx
                elif mode == "test":
                    masks[mode] = self.test_idx

            metric_dict = {key: {} for key in masks}
            loss_dict = {}

            if self.is_multi_label:
                from sklearn.metrics import hamming_loss, accuracy_score, f1_score, precision_score, recall_score
                import time

                for key, mask in masks.items():
                    y_true = self.labels[mask].cpu().numpy()
                    y_logits = logits[mask]
                    y_pred_prob = torch.sigmoid(y_logits).cpu().numpy()
                    y_pred = (y_pred_prob > self.threshold).astype(int)

                    metric_dict[key]['hamming_loss'] = hamming_loss(y_true, y_pred)
                    metric_dict[key]['subset_accuracy'] = accuracy_score(y_true, y_pred)
                    metric_dict[key]['sample_f1'] = f1_score(y_true, y_pred, average='samples', zero_division=0)
                    metric_dict[key]['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
                    metric_dict[key]['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
                    metric_dict[key]['micro_precision'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
                    metric_dict[key]['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
                    metric_dict[key]['micro_recall'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
                    metric_dict[key]['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)

                    loss_dict[key] = self.loss_fn(y_logits, self.labels[mask]).item()

            else:
                for key in masks:
                    metric_dict[key] = self.task.evaluate(logits, mode=key)

                y_trues = self.labels[masks['test']].to(self.device)
                y_predicts = logits[masks['test']]
                accuracy = (y_predicts.argmax(dim=1) == y_trues).float().mean()
                metric_dict['test']['accuracy'] = accuracy.item()

                y_trues_cpu = y_trues.cpu().numpy()
                y_predicts_cpu = y_predicts.argmax(dim=1).cpu().numpy()

                precision = precision_score(y_trues_cpu, y_predicts_cpu, average='weighted', zero_division=0)
                micro_f1 = f1_score(y_trues_cpu, y_predicts_cpu, average='micro', zero_division=0)
                macro_f1 = f1_score(y_trues_cpu, y_predicts_cpu, average='macro', zero_division=0)

                for key, mask in masks.items():
                    loss_dict[key] = self.loss_fn(logits[mask], self.labels[mask]).item()

            return metric_dict, loss_dict

    def _mini_test_step(self, modes):
        if self.args.graphbolt:
            self.model.eval()
            with torch.no_grad():
                metric_dict = {}
                loss_dict = {}
                loss_all = 0.0
               
                for mode in modes:  
                    if mode == 'train':
                        loader = self.train_GB_loader
                    elif mode == 'valid':
                        loader = self.val_GB_loader
                    elif mode == 'test':
                        loader = self.test_GB_loader
                    y_trues = []
                    y_predicts = []
                    for i, data in enumerate(loader):                        
                        input_nodes = data.input_nodes
                        seeds = data.seeds   
                        if not isinstance(input_nodes, dict):
                            input_nodes = {self.category: input_nodes}
                        emb = self.model.input_feature.forward_nodes(input_nodes)
                        label = data.labels[self.category].to(self.device)
                        logits = self.model(data.blocks, emb)[self.category]   
                        loss = self.loss_fn(logits, label)
                        loss_all += loss.item() 
                        
                        y_trues.append(label.detach().cpu())
                        y_predicts.append(logits.detach().cpu())
                    loss_all /= (i + 1)
                    y_trues = torch.cat(y_trues, dim=0)
                    y_predicts = torch.cat(y_predicts, dim=0)
                    evaluator = self.task.get_evaluator(name='f1')
                    
                    metric_dict[mode] = evaluator(y_trues, y_predicts.argmax(dim=1).to('cpu'))
                    loss_dict[mode] = loss_all
            return metric_dict, loss_dict
       
        else:
            self.model.eval()
            with torch.no_grad():
                metric_dict = {}
                loss_dict = {}
                loss_all = 0.0
                for mode in modes:
                    if mode == 'train':
                        loader_tqdm = tqdm(self.train_loader, ncols=120)
                    elif mode == 'valid':
                        loader_tqdm = tqdm(self.val_loader, ncols=120)
                    elif mode == 'test':
                        loader_tqdm = tqdm(self.test_loader, ncols=120)
                    y_trues = []
                    y_predicts = []
                    for i, (input_nodes, seeds, blocks) in enumerate(loader_tqdm):
                        if self.to_homo_flag:
                            seeds = to_hetero_idx(self.g, self.hg, seeds)
                        elif not isinstance(input_nodes, dict):
                            input_nodes = {self.category: input_nodes}
                        emb = self.model.input_feature.forward_nodes(input_nodes)
                        lbl = self.labels[seeds[self.category]].to(self.device)
                        logits = self.model(blocks, emb)[self.category]
                        loss = self.loss_fn(logits, lbl)
                        loss_all += loss.item()
                        y_trues.append(lbl.detach().cpu())
                        y_predicts.append(logits.detach().cpu())
                    loss_all /= (i + 1)
                    y_trues = torch.cat(y_trues, dim=0)
                    y_predicts = torch.cat(y_predicts, dim=0)
                    evaluator = self.task.get_evaluator(name='f1')
                    metric_dict[mode] = evaluator(y_trues, y_predicts.argmax(dim=1).to('cpu'))
                    loss_dict[mode] = loss
            return metric_dict, loss_dict

    def _full_prediction_step(self):
        self.model.eval()
        with torch.no_grad():
            h_dict = self.model.input_feature()
            h_dict = {k: e.to(self.device) for k, e in h_dict.items()}
            logits = self.model(self.hg, h_dict)[self.category]
            return logits

    def _mini_prediction_step(self):
        self.model.eval()
        with torch.no_grad():
            loader_tqdm = tqdm(self.pred_loader, ncols=120)
            indices = []
            y_predicts = []
            for i, (input_nodes, seeds, blocks) in enumerate(loader_tqdm):
                if self.to_homo_flag:
                    input_nodes = to_hetero_idx(self.g, self.hg, input_nodes)
                    seeds = to_hetero_idx(self.g, self.hg, seeds)
                elif not isinstance(input_nodes, dict):
                    input_nodes = {self.category: input_nodes}
                emb = self.model.input_feature.forward_nodes(input_nodes)
                if self.to_homo_flag:
                    emb = to_homo_feature(self.hg.ntypes, emb)
                logits = self.model(blocks, emb)[self.category]
                seeds = seeds[self.category]
                indices.append(seeds.detach().cpu())
                y_predicts.append(logits.detach().cpu())
            indices = torch.cat(indices, dim=0)
            y_predicts = torch.cat(y_predicts, dim=0)
        return indices, y_predicts
