import torch as th
from . import BaseDataset, register_dataset
import torch
from .your_dataset_filtered import YourDatasetFiltered
from .imdb_dataset import IMDBDataset
from .persona_dataset import PersonaDataset

@register_dataset('node_classification')
class NodeClassificationDataset(BaseDataset):
    r"""
    The class *NodeClassificationDataset* is a base class for datasets which can be used in task *node classification*.
    So its subclass should contain attributes such as graph, category, num_classes and so on.
    Besides, it should implement the functions *get_labels()* and *get_split()*.

    Attributes
    -------------
    g : dgl.DGLHeteroGraph
        The heterogeneous graph.
    category : str
        The category(or target) node type need to be predict. In general, we predict only one node type.
    num_classes : int
        The target node  will be classified into num_classes categories.
    has_feature : bool
        Whether the dataset has feature. Default ``False``.
    multi_label : bool
        Whether the node has multi label. Default ``False``. For now, only HGBn-IMDB has multi-label.
    """

    def __init__(self, *args, **kwargs):
        super(NodeClassificationDataset, self).__init__(*args, **kwargs)
        self.g = None
        self.category = None
        self.num_classes = None
        self.has_feature = False
        self.multi_label = False
        self.meta_paths_dict =None
                            

    def get_labels(self):
        r"""
        The subclass of dataset should overwrite the function. We can get labels of target nodes through it.

        Notes
        ------
        In general, the labels are th.LongTensor.
        But for multi-label dataset, they should be th.FloatTensor. Or it will raise
        RuntimeError: Expected object of scalar type Long but got scalar type Float for argument #2 target' in call to _thnn_nll_loss_forward

        return
        -------
        labels : torch.Tensor
        """
        if 'labels' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('labels')
                                             
            if labels.dtype == torch.float32 and labels.dim() == 2:
                pass                        
            else:
                labels = labels.long()                       
        elif 'label' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('label')
            if labels.dtype == torch.float32 and labels.dim() == 2:
                pass         
            else:
                labels = labels.long()         
        else:
            raise ValueError('Labels of nodes are not in the hg.nodes[category].data.')
        labels = labels.float() if self.multi_label else labels
        return labels

    def get_split(self, validation=True):
        r"""
        
        Parameters
        ----------
        validation : bool
            Whether to split dataset. Default ``True``. If it is False, val_idx will be same with train_idx.

        We can get idx of train, validation and test through it.

        return
        -------
        train_idx, val_idx, test_idx : torch.Tensor, torch.Tensor, torch.Tensor
        """
        if 'train_mask' not in self.g.nodes[self.category].data:
            self.logger.dataset_info("The dataset has no train mask. "
                  "So split the category nodes randomly. And the ratio of train/test is 8:2.")
            num_nodes = self.g.number_of_nodes(self.category)
            n_test = int(num_nodes * 0.2)
            n_train = num_nodes - n_test
    
            train, test = th.utils.data.random_split(range(num_nodes), [n_train, n_test])
            train_idx = th.tensor(train.indices)
            test_idx = th.tensor(test.indices)
            if validation:
                self.logger.dataset_info("Split train into train/valid with the ratio of 8:2 ")
                random_int = th.randperm(len(train_idx))
                valid_idx = train_idx[random_int[:len(train_idx) // 5]]
                train_idx = train_idx[random_int[len(train_idx) // 5:]]
            else:
                self.logger.dataset_info("Set valid set with train set.")
                valid_idx = train_idx
                train_idx = train_idx
        else:
            train_mask = self.g.nodes[self.category].data.pop('train_mask')
            test_mask = self.g.nodes[self.category].data.pop('test_mask')
            train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
            test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
            if validation:
                if 'val_mask' in self.g.nodes[self.category].data:
                    val_mask = self.g.nodes[self.category].data.pop('val_mask')
                    valid_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
                elif 'valid_mask' in self.g.nodes[self.category].data:
                    val_mask = self.g.nodes[self.category].data.pop('valid_mask').squeeze()
                    valid_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
                else:
                                                                        
                    self.logger.dataset_info("Split train into train/valid with the ratio of 8:2 ")
                    random_int = th.randperm(len(train_idx))
                    valid_idx = train_idx[random_int[:len(train_idx) // 5]]
                    train_idx = train_idx[random_int[len(train_idx) // 5:]]
            else:
                self.logger.dataset_info("Set valid set with train set.")
                valid_idx = train_idx
                train_idx = train_idx
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
                                                                          
        return self.train_idx, self.valid_idx, self.test_idx


@register_dataset('imdb_node_classification')
class IMDBNodeClassification(NodeClassificationDataset):
    """
    IMDB Dataset for Node Classification

    Usage:
        python main.py --model ROSA --task node_classification --dataset imdb_node_classification --gpu 0
    """
    def __init__(self, *args, **kwargs):
        super(IMDBNodeClassification, self).__init__(*args, **kwargs)

                                                                      
        data_path = kwargs.get('data_path', './openhgnn/dataset/data/imdb/')

                                                               
        dataset = IMDBDataset(name='imdb', data_path=data_path, logger=kwargs.get('logger'))
        self.g = dataset.graph
        self.category = 'movie'                                       
        self.num_classes = dataset.num_classes
        self.has_feature = True

                                                                             
                                                                                  
        args_obj = kwargs.get('args')
        if args_obj and hasattr(args_obj, 'multi_label'):
            self.multi_label = args_obj.multi_label
            print(f"[IMDB NC] multi_label from args: {self.multi_label}")
        else:
                                                                            
            self.multi_label = True
            print(f"[IMDB NC] multi_label default: {self.multi_label}")

                                                                              
        self.meta_paths_dict = {
            'MAM': [('movie', 'acted_by', 'actor'), ('actor', 'act_in', 'movie')],
            'MDM': [('movie', 'directed_by', 'director'), ('director', 'direct', 'movie')],
                                                                                  
        }

        print(f"[IMDB NC] Loaded graph with {self.g.num_nodes()} nodes")
        print(f"[IMDB NC] Target: {self.category}, Classes: {self.num_classes}")

    def get_labels(self):
        """Return labels for the target node type"""
        if 'label' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data['label']
            return labels.long() if not self.multi_label else labels.float()
        else:
            raise ValueError(f"No labels found for node type '{self.category}'")

    def get_split(self, validation=True):
        """
        Split dataset into train/val/test sets.
        If masks exist in the graph, use them; otherwise create random split.
        """
        if 'train_mask' in self.g.nodes[self.category].data:
                                
            train_mask = self.g.nodes[self.category].data['train_mask']
            val_mask = self.g.nodes[self.category].data['val_mask']
            test_mask = self.g.nodes[self.category].data['test_mask']

            train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
            val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
            test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()

            print(f"[IMDB NC] Using existing masks - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        else:
                                            
            num_nodes = self.g.number_of_nodes(self.category)
            indices = torch.randperm(num_nodes)

            n_train = int(num_nodes * 0.6)
            n_val = int(num_nodes * 0.2)

            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]

            print(f"[IMDB NC] Random split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        self.train_idx = train_idx
        self.valid_idx = val_idx
        self.test_idx = test_idx

        return train_idx, val_idx, test_idx


@register_dataset('persona_node_classification')
class PersonaNodeClassification(NodeClassificationDataset):
    """
    Persona Dataset for Node Classification

    Usage:
        python main.py --model ROSA --task node_classification --dataset persona_node_classification --gpu 0
    """
    def __init__(self, *args, **kwargs):
        super(PersonaNodeClassification, self).__init__(*args, **kwargs)

                                                                      
        data_path = kwargs.get('data_path', './openhgnn/dataset/data/persona/')

                      
        dataset = PersonaDataset(name='persona', data_path=data_path, logger=self.logger)
        self.g = dataset.graph

                                                             
        self.category = dataset.target_node_type if hasattr(dataset, 'target_node_type') else 'user'
        self.num_classes = dataset.num_classes
        self.has_feature = True
        self.multi_label = True                                                   

                                                                      
                                                                            
        if 'user' in self.g.ntypes and 'product' in self.g.ntypes and 'persona' in self.g.ntypes:
            self.meta_paths_dict = {
                                                 
                'UPU': [
                    ('user', 'interact', 'product'),
                    ('product', 'interacted_by', 'user')
                ],
                                                    
                'UPPU': [
                    ('user', 'interact', 'product'),
                    ('product', 'define', 'persona'),
                    ('persona', 'define_by', 'product'),
                    ('product', 'interacted_by', 'user')
                ],
            }


        print(f"[Persona NC] Loaded graph with {self.g.num_nodes()} nodes")
        print(f"[Persona NC] Target: {self.category}, Classes: {self.num_classes}")
        print(f"[Persona NC] Node types: {self.g.ntypes}")

    def get_labels(self):
        """Return labels for the target node type"""
        if 'label' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data['label']
            return labels.long() if not self.multi_label else labels.float()
        else:
            raise ValueError(f"No labels found for node type '{self.category}'")

    def get_split(self, validation=True):
        """
        Split dataset into train/val/test sets.
        If masks exist in the graph, use them; otherwise create random split.
        """
        if 'train_mask' in self.g.nodes[self.category].data:
                                
            train_mask = self.g.nodes[self.category].data['train_mask']
            val_mask = self.g.nodes[self.category].data['val_mask']
            test_mask = self.g.nodes[self.category].data['test_mask']

            train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
            val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
            test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()

            print(f"[Persona NC] Using existing masks - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        else:
                                            
            num_nodes = self.g.number_of_nodes(self.category)
            indices = torch.randperm(num_nodes)

            n_train = int(num_nodes * 0.6)
            n_val = int(num_nodes * 0.2)

            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]

            print(f"[Persona NC] Random split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        self.train_idx = train_idx
        self.valid_idx = val_idx
        self.test_idx = test_idx

        return train_idx, val_idx, test_idx


@register_dataset('my_custom_node_classification_filtered')
class ChemistryDatasetFiltered(NodeClassificationDataset):
    """
    Chemistry Dataset with Rare Label Filtering for Node Classification

    This dataset automatically filters out rare labels:
    - If a sample has both rare and common labels, keep the sample but remove rare labels
    - If a sample only has rare labels, remove the entire sample

    Usage:
        python main.py --model HAN --task node_classification --dataset my_custom_node_classification_filtered --gpu 0

    Parameters:
        min_label_samples: Minimum number of samples required for a label to be considered common (default: 10)
    """
    def __init__(self, *args, **kwargs):
        super(ChemistryDatasetFiltered, self).__init__(*args, **kwargs)

                                                     
        min_label_samples = 10           
        if 'args' in kwargs:
            args_obj = kwargs['args']
            if hasattr(args_obj, 'min_label_samples'):
                min_label_samples = args_obj.min_label_samples

                                   
        dataset = YourDatasetFiltered(
            name='your_dataset_filtered',
            raw_dir='',
            min_label_samples=min_label_samples
        )

        self.g = dataset.graph
        self.category = dataset.category           
        self.num_classes = dataset.num_classes       
        self.has_feature = True
        self.multi_label = True                              

                                          
        if hasattr(dataset.graph, 'meta_paths_dict'):
            self.meta_paths_dict = dataset.graph.meta_paths_dict

        print(f"[ChemistryDatasetFiltered] Initialized with min_label_samples={min_label_samples}")
        print(f"[ChemistryDatasetFiltered] Graph: {self.g.num_nodes('paper')} paper nodes")
        print(f"[ChemistryDatasetFiltered] Target: {self.category}, Classes: {self.num_classes}")
        print(f"[ChemistryDatasetFiltered] Multi-label: {self.multi_label}")

    def get_labels(self):
        """Return multi-label FloatTensor for the target node type"""
        if 'labels' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data['labels']
        elif 'label' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data['label']
        else:
            raise ValueError(f"No labels found for node type '{self.category}'")

                                             
        return labels.float() if self.multi_label else labels.long()

    def get_split(self, validation=True):
        """
        Split dataset into train/val/test sets.
        Uses existing masks from the filtered dataset.
        """
        if 'train_mask' in self.g.nodes[self.category].data:
            train_mask = self.g.nodes[self.category].data['train_mask']
            test_mask = self.g.nodes[self.category].data['test_mask']
            train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
            test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()

            if validation:
                if 'val_mask' in self.g.nodes[self.category].data:
                    val_mask = self.g.nodes[self.category].data['val_mask']
                    valid_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
                else:
                                                
                    random_int = torch.randperm(len(train_idx))
                    valid_idx = train_idx[random_int[:len(train_idx) // 5]]
                    train_idx = train_idx[random_int[len(train_idx) // 5:]]
            else:
                valid_idx = train_idx
        else:
                                                   
            num_nodes = self.g.number_of_nodes(self.category)
            indices = torch.randperm(num_nodes)

            n_train = int(num_nodes * 0.6)
            n_val = int(num_nodes * 0.2)

            train_idx = indices[:n_train]
            valid_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]

        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx

        print(f"[ChemistryDatasetFiltered] Split - Train: {len(train_idx)}, Val: {len(valid_idx)}, Test: {len(test_idx)}")

        return train_idx, valid_idx, test_idx
