import dgl
import torch
from tqdm import tqdm
from ..models import build_model
from . import BaseFlow, register_flow
from ..utils import EarlyStopping, to_hetero_idx, to_homo_feature, to_homo_idx
import warnings
from torch.utils.tensorboard import SummaryWriter
import dgl.graphbolt as gb
import tkinter as tk
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

@register_flow("node_classification")
class NodeClassification(BaseFlow):
    r"""
    Node classification flow,
    The task is to classify the nodes of target nodes.
    Note: If the output dim is not equal the number of classes, we will modify the output dim with the number of classes.
    """

    def __init__(self, args):
        """

        Attributes
        ------------
        category: str
            The target node type to predict
        num_classes: int
            The number of classes for category node type

        """

        super(NodeClassification, self).__init__(args)

                               
                          
                                        
                                           
                                 
                                   
                                      
                                         
        
        self.args.category = self.task.dataset.category
        self.category = self.args.category
                                                                                                                        

        self.num_classes = self.task.dataset.num_classes

        if not hasattr(self.task.dataset, 'out_dim') or args.out_dim != self.num_classes:
            self.logger.info('[NC Specific] Modify the out_dim with num_classes')
            args.out_dim = self.num_classes
        self.args.out_node_type = [self.category]

             
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
            self.logger.info(f'[Multi-label Mode] Using BCEWithLogitsLoss, threshold={self.threshold}')
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

        if self.to_homo_flag:
            self.g = dgl.to_homogeneous(self.hg)

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
            if self.train_idx is not None:
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

    def preprocess(self):
        r"""
        Preprocess for training and prepare the dataloader for train/validation/test.
        """
        super(NodeClassification, self).preprocess()

    def train(self):
        self.preprocess()
        print("Using node trainer")
        stopper = EarlyStopping(self.args.patience, self._checkpoint)                              
        epoch_iter = tqdm(range(self.max_epoch))                        
        for epoch in epoch_iter:

            if self.args.output_widget != None:
                                                                      
                self.args.output_widget.insert(tk.END, f"Current epoch: {epoch}\n")
                self.args.output_widget.see(tk.END)
                self.args.output_widget.update_idletasks()
                    

            if self.args.mini_batch_flag:
                train_loss = self._mini_train_step()
            else:
                train_loss = self._full_train_step()
                                                      
            if epoch % self.evaluate_interval == 0:               
                modes = ['train', 'valid']
                if self.args.test_flag:
                    modes = modes + ['test']
                if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                    metric_dict, losses = self._mini_test_step(modes=modes)
                else:
                    metric_dict, losses = self._full_test_step(modes=modes)
                val_loss = losses['valid']
                self.logger.train_info(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Valid loss: {val_loss:.4f}. "
                                       + self.logger.metric2str(metric_dict))

                                                                      
                if self.args.output_widget != None:
                    self.args.output_widget.insert(
                        tk.END,
                        f"Epoch {epoch}: "
                        f"Train loss: {train_loss:.4f}, "
                        f"Valid loss: {val_loss:.4f}, "
                        f"Metrics: {self.logger.metric2str(metric_dict)}.\n"
                    )
                    self.args.output_widget.see(tk.END)
                    self.args.output_widget.update_idletasks()
                                    


                self.writer.add_scalars('loss', {'train': train_loss, 'valid': val_loss}, global_step=epoch)
                for mode in modes:
                    self.writer.add_scalars(f'metric_{mode}', metric_dict[mode], global_step=epoch)
                early_stop = stopper.loss_step(val_loss, self.model)
                if early_stop:
                    self.logger.train_info('Early Stop!\tEpoch:' + str(epoch))
                    break
        stopper.load_model(self.model)
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
            self.logger.info('trainerflow  finished  ')
            return dict(metric=metric_dict, epoch=epoch)
        
        self.writer.close()

        

    def _full_train_step(self):
        self.model.train()                             
        h_dict = self.model.input_feature()       
        self.hg = self.hg.to(self.device)
                                                            
        logits = self.model(self.hg, h_dict)[self.category] 
                                                          
                                                                      

        loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _mini_train_step(self):
        if getattr(self.args, "graphbolt", False):
            print("Using GraphBolt")
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
            print("Using DGL default dataloader")
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
        """
        Parameters
        ----------
        mode: list[str]
            `train`, 'test', 'valid' are optional in list.
        logits: dict[str, th.Tensor]
            given logits, default `None`.

        Returns
        -------
        metric_dict: dict[str, float]
            score of evaluation metric
        info: dict[str, str]
            evaluation information
        loss: dict[str, float]
            the loss item
        """
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

                               
            if self.is_multi_label:
                from sklearn.metrics import hamming_loss, accuracy_score

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

                                       
                    if key == 'test':
                        self.logger.info(f"[Multi-label Eval - {key}]")
                        self.logger.info(f"  Hamming Loss: {metric_dict[key]['hamming_loss']:.4f}")
                        self.logger.info(f"  Subset Accuracy: {metric_dict[key]['subset_accuracy']:.4f}")
                        self.logger.info(f"  Sample F1: {metric_dict[key]['sample_f1']:.4f}")
                        self.logger.info(f"  Micro-F1: {metric_dict[key]['micro_f1']:.4f}")
                        self.logger.info(f"  Macro-F1: {metric_dict[key]['macro_f1']:.4f}")

                               
            else:
                for key in masks:
                    metric_dict[key] = self.task.evaluate(logits, mode=key)

                                  
                if 'test' in masks:
                    y_trues = self.labels[masks['test']].to(self.device)
                    y_predicts = logits[masks['test']]
                    accuracy = (y_predicts.argmax(dim=1) == y_trues).float().mean()
                    metric_dict['test']['accuracy'] = accuracy.item()

                                      
                    y_trues_cpu = y_trues.cpu().numpy()
                    y_predicts_cpu = y_predicts.argmax(dim=1).cpu().numpy()

                                  
                    precision = precision_score(y_trues_cpu, y_predicts_cpu, average='weighted', zero_division=0)
                    micro_f1 = f1_score(y_trues_cpu, y_predicts_cpu, average='micro', zero_division=0)
                    macro_f1 = f1_score(y_trues_cpu, y_predicts_cpu, average='macro', zero_division=0)

                    self.logger.info(f"[Single-label Eval - test]")
                    self.logger.info(f"  Precision: {precision:.4f}")
                    self.logger.info(f"  Micro-F1: {micro_f1:.4f}")
                    self.logger.info(f"  Macro-F1: {macro_f1:.4f}")

                  
            loss_dict = {key: self.loss_fn(logits[mask], self.labels[mask]).item() for key, mask in masks.items()}

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

                                       
                    if self.is_multi_label:
                        from sklearn.metrics import hamming_loss, accuracy_score

                        y_true_np = y_trues.numpy()
                        y_pred_prob = torch.sigmoid(y_predicts).numpy()
                        y_pred_np = (y_pred_prob > self.threshold).astype(int)

                        metric_dict[mode] = {
                            'hamming_loss': hamming_loss(y_true_np, y_pred_np),
                            'micro_f1': f1_score(y_true_np, y_pred_np, average='micro', zero_division=0),
                            'macro_f1': f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
                        }
                                       
                    else:
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

                                       
                    if self.is_multi_label:
                        from sklearn.metrics import hamming_loss, accuracy_score

                        y_true_np = y_trues.numpy()
                        y_pred_prob = torch.sigmoid(y_predicts).numpy()
                        y_pred_np = (y_pred_prob > self.threshold).astype(int)

                        metric_dict[mode] = {
                            'hamming_loss': hamming_loss(y_true_np, y_pred_np),
                            'micro_f1': f1_score(y_true_np, y_pred_np, average='micro', zero_division=0),
                            'macro_f1': f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
                        }
                                       
                    else:
                        evaluator = self.task.get_evaluator(name='f1')
                        metric_dict[mode] = evaluator(y_trues, y_predicts.argmax(dim=1).to('cpu'))

                    loss_dict[mode] = loss
            return metric_dict, loss_dict

    def _full_prediction_step(self):
        """

        Returns
        -------
        """
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
