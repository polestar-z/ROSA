from ..sampler import AOASampler
import dgl
from tqdm import tqdm
import torch
from . import BaseFlow, register_flow
from ..models import build_model
from ..utils import EarlyStopping


@register_flow("aoa_nc_trainer")
class AOANodeClassification(BaseFlow):
    r"""
    AOA node classification flow.
    """

    def __init__(self, args):
        super(AOANodeClassification, self).__init__(args)
        
                                                  
                                                                     
        
                       
                                         
        if hasattr(self.hg, 'meta_paths_dict') and self.hg.meta_paths_dict is not None:
            self.args.meta_paths_dict = self.hg.meta_paths_dict
            self.logger.info(f"Successfully loaded meta_paths_dict from graph: {self.args.meta_paths_dict.keys()}")
        else:
                                                                              
            self.logger.error("Error: self.hg.meta_paths_dict is not found or is None after dataset loading.")
                                         
            raise AttributeError("meta_paths_dict not found in graph object. Please ensure your dataset sets self.hg.meta_paths_dict.")

                                                           
                                                                                   
                                           
        if hasattr(self.hg, 'target_node_type') and self.hg.target_node_type is not None:
            self.args.target_node_type = self.hg.target_node_type
            self.logger.info(f"Successfully loaded target_node_type from graph: {self.args.target_node_type}")
        else:
                                                         
            self.logger.error("Error: self.hg.target_node_type is not found or is None.")
                                 
                                                                                   


        self.args.category = self.task.dataset.category
        self.category = self.args.category

                            
                                                      
                                                                                  
                                       
                                                                                      
                                                                                          
               
                                                                                          
                           

        self.num_classes = self.task.dataset.num_classes
        if not hasattr(args, 'batch_size'):
            args.batch_size = 256                     
            self.logger.info(f"DEBUG: batch_size not found in args, setting to default: {args.batch_size}")
        

        if not hasattr(self.task.dataset, 'out_dim') or args.out_dim != self.num_classes:
            self.logger.info('[NC Specific] Modify the out_dim with num_classes')
            args.out_dim = self.num_classes
        self.args.out_node_type = [self.category]

                                              
        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)
        print(f"DEBUG: args.mini_batch_flag = {self.args.mini_batch_flag}")

        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(),
                                                                   lr=args.lr, weight_decay=args.weight_decay)

        self.train_idx, self.valid_idx, self.test_idx = self.task.get_split()

        if self.args.prediction_flag:
            self.pred_idx = self.task.dataset.pred_idx

        self.labels = self.task.get_labels().to(self.device)

                                                      
        self.is_multi_label = getattr(args, 'multi_label', False)
        self.threshold = getattr(args, 'threshold', 0.5)

        if self.is_multi_label:
            self.logger.info(f'[Multi-label Mode] Using BCEWithLogitsLoss, threshold={self.threshold}')
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

        if self.args.mini_batch_flag:
                                                                   
            sampler = AOASampler(g=self.hg, seed_ntypes=[self.category], meta_paths_dict=self.args.meta_paths_dict,
                                 num_neighbors=20)
            if self.train_idx is not None:
                self.train_loader = dgl.dataloading.DataLoader(
                    self.hg, {self.category: self.train_idx.to(self.device)}, sampler,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)
            if self.valid_idx is not None:
                self.val_loader = dgl.dataloading.DataLoader(
                    self.hg, {self.category: self.valid_idx.to(self.device)}, sampler,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)
            if self.args.test_flag:
                self.test_loader = dgl.dataloading.DataLoader(
                    self.hg, {self.category: self.test_idx.to(self.device)}, sampler,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)
            if self.args.prediction_flag:
                self.pred_loader = dgl.dataloading.DataLoader(
                    self.hg, {self.category: self.pred_idx.to(self.device)}, sampler,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)

    def preprocess(self):

        super(AOANodeClassification, self).preprocess()

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
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
            return dict(metric=metric_dict, epoch=epoch)

    def _full_train_step(self):
        self.model.train()
                                                                              
        h_dict = self.model.input_feature()
                                                                               
        logits = self.model(self.hg, h_dict)[self.category]
                                                           
        loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
                                                                         
        self.optimizer.zero_grad()
                                                                   
        loss.backward()
                                                                    
        self.optimizer.step()
                                                               
        return loss.item()

    def _mini_train_step(self, ):
        self.model.train()
        loss_all = 0.0
        loader_tqdm = tqdm(self.train_loader, ncols=120)
        for i, (ntype_mp_name_input_nodes_dict, seeds, ntype_mp_name_block_dict) in enumerate(loader_tqdm):
            seeds = seeds[self.category]
            mp_name_input_nodes_dict = ntype_mp_name_input_nodes_dict[self.category]
            emb_dict = {}
            for meta_path_name, input_nodes in mp_name_input_nodes_dict.items():
                emb_dict[meta_path_name] = self.model.input_feature.forward_nodes({self.category: input_nodes})
            emb_dict = {self.category: emb_dict}
            logits = self.model(ntype_mp_name_block_dict, emb_dict)[self.category]
            lbl = self.labels[seeds].to(self.device)
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
                    masks[mode] = self.valid_idx
                elif mode == "test":
                    masks[mode] = self.test_idx

                               
            if self.is_multi_label:
                from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, accuracy_score

                metric_dict = {key: {} for key in masks}
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
                metric_dict = {key: self.task.evaluate(logits, mode=key) for key in masks}

            loss_dict = {key: self.loss_fn(logits[mask], self.labels[mask]).item() for key, mask in masks.items()}
            return metric_dict, loss_dict

    def _mini_test_step(self, modes):
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

                for i, (ntype_mp_name_input_nodes_dict, seeds, ntype_mp_name_block_dict) in enumerate(loader_tqdm):
                    seeds = seeds[self.category]
                    mp_name_input_nodes_dict = ntype_mp_name_input_nodes_dict[self.category]
                    emb_dict = {}
                    for meta_path_name, input_nodes in mp_name_input_nodes_dict.items():
                        emb_dict[meta_path_name] = self.model.input_feature.forward_nodes({self.category: input_nodes})
                    emb_dict = {self.category: emb_dict}
                    logits = self.model(ntype_mp_name_block_dict, emb_dict)[self.category]
                    lbl = self.labels[seeds].to(self.device)
                    loss = self.loss_fn(logits, lbl)
                    loss_all += loss.item()
                    y_trues.append(lbl.detach().cpu())
                    y_predicts.append(logits.detach().cpu())
                loss_all /= (i + 1)
                y_trues = torch.cat(y_trues, dim=0)
                y_predicts = torch.cat(y_predicts, dim=0)

                                   
                if self.is_multi_label:
                    from sklearn.metrics import f1_score

                    y_true_np = y_trues.numpy()
                    y_pred_prob = torch.sigmoid(y_predicts).numpy()
                    y_pred_np = (y_pred_prob > self.threshold).astype(int)

                    metric_dict[mode] = {
                        'micro_f1': f1_score(y_true_np, y_pred_np, average='micro', zero_division=0),
                        'macro_f1': f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
                    }
                                   
                else:
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
            for i, (input_nodes_dict, seeds, block_dict) in enumerate(loader_tqdm):
                seeds = seeds[self.category]
                emb_dict = {}
                for meta_path_name, input_nodes in input_nodes_dict.items():
                    emb_dict[meta_path_name] = self.model.input_feature.forward_nodes(input_nodes)
                logits = self.model(block_dict, emb_dict)[self.category]
                indices.append(seeds.detach().cpu())
                y_predicts.append(logits.detach().cpu())
            indices = torch.cat(indices, dim=0)
            y_predicts = torch.cat(y_predicts, dim=0)
        return indices, y_predicts
