              
import dgl
import torch
import pandas as pd
import numpy as np
from .base_dataset import BaseDataset


def parse_multi_labels(label_str, num_classes):
    """
    Convert a comma-separated label string to a multi-hot vector.

    Args:
        label_str: "0,2,5" or "3" or NaN
        num_classes: total number of labels

    Returns:
        multi_hot: numpy array [1,0,1,0,0,1,...] shape: (num_classes,)
    """
    multi_hot = np.zeros(num_classes, dtype=np.float32)
    if pd.notna(label_str):        
        try:
            label_str = str(label_str).strip()
                                           
            if ',' in label_str:
                label_ids = [int(x.strip()) for x in label_str.split(',')]
            else:
                label_ids = [int(label_str)]

                         
            valid_ids = [lid for lid in label_ids if 0 <= lid < num_classes]
            if len(valid_ids) != len(label_ids):
                print(f"  [Warning] Label '{label_str}' has IDs out of range [0, {num_classes})")
            multi_hot[valid_ids] = 1
        except (ValueError, TypeError) as e:
            print(f"  [Error] Cannot parse label: '{label_str}', error: {e}")
    return multi_hot


class PersonaDataset(BaseDataset):
    """
    Persona Dataset for heterogeneous graph node classification.

    Expected data structure:
    - nodes/: Contains CSV files for each node type
    - edges/: Contains CSV files for each edge type

    This is a generic template. Adjust node and edge types according to your actual data.
    """

    def __init__(self, name='persona', data_path=None, logger=None, *args, **kwargs):
        """
        Parameters
        ----------
        name : str
            Dataset name
        data_path : str
            Base path to the dataset directory containing nodes/ and edges/ folders
            If None, will use default path: './openhgnn/dataset/data/persona/'
        logger : optional
            Logger instance
        """
                                                     
        kwargs['logger'] = logger
        super(PersonaDataset, self).__init__(*args, **kwargs)
        self.name = name

                                               
        if data_path is None:
            self.data_path = './openhgnn/dataset/data/persona/'
        else:
            self.data_path = data_path

        self.graph = self.load_graph()                                           
                                                     

    def load_graph(self):
        """
        Load Persona heterogeneous graph from CSV files.

        Returns
        -------
        graph : dgl.DGLHeteroGraph
            The constructed heterogeneous graph
        """
        print(f"[Persona Dataset] Loading from {self.data_path}")
                 
        user_nodes = pd.read_csv(f"{self.data_path}nodes/user_embedding32.csv")
        product_nodes = pd.read_csv(f"{self.data_path}nodes/product-embedding.csv")
        persona_nodes = pd.read_csv(f"{self.data_path}nodes/persona_embedding_384.csv")
        label = pd.read_csv(f"{self.data_path}lab-all.csv")

        print(f"  [Nodes] User: {len(user_nodes)}, product: {len(product_nodes)},persona:{len(persona_nodes)}")
                                                     
        user_product_edges = pd.read_csv(f"{self.data_path}edges/user-product.csv")
        product_persona_edges = pd.read_csv(f"{self.data_path}edges/product-persona.csv")

                                                      
        data_dict = {
            ('user', 'interact', 'product'): (
                torch.tensor(user_product_edges['user_id'].values, dtype=torch.long),
                torch.tensor(user_product_edges['product_id'].values, dtype=torch.long)
            ),
                              
            ('product', 'interacted_by', 'user'): (
                torch.tensor(user_product_edges['product_id'].values, dtype=torch.long),
                torch.tensor(user_product_edges['user_id'].values, dtype=torch.long)
            ),
            ('product', 'define', 'persona'): (
                torch.tensor(product_persona_edges['product_id'].values, dtype=torch.long),
                torch.tensor(product_persona_edges['persona_id'].values, dtype=torch.long)
            ),
            ('persona', 'define_by', 'product'): (
                torch.tensor(product_persona_edges['persona_id'].values, dtype=torch.long),
                torch.tensor(product_persona_edges['product_id'].values, dtype=torch.long)
            ),
                                            
                                                           
                                                   
        }

        graph = dgl.heterograph(data_dict)
        print(f"  [Graph] Created with {graph.num_nodes()} nodes and {graph.num_edges()} edges")
        print(f"  [Graph] Node types: {graph.ntypes}")
        print(f"  [Graph] Edge types: {graph.etypes}")
                                                                                          
        user_feature_cols = [col for col in user_nodes.columns
                            if 'embedding' in col.lower() or 'feature' in col.lower()]
        product_feature_cols = [col for col in product_nodes.columns
                            if 'embedding' in col.lower() or 'feature' in col.lower()]
        persona_feature_cols = [col for col in persona_nodes.columns
                            if 'emb' in col.lower() or 'feature' in col.lower()]

                                                             
        feature_dim = 128                             

        if not user_feature_cols:
            print(f"  [Warning] No user features found, creating random {feature_dim}-dim features")
            user_features = torch.randn(len(user_nodes), feature_dim)
        else:
            user_features = torch.tensor(user_nodes[user_feature_cols].values, dtype=torch.float32)

        if not product_feature_cols:
            print(f"  [Warning] No product features found, creating random {feature_dim}-dim features")
            product_features = torch.randn(len(product_nodes), feature_dim)
        else:
            product_features = torch.tensor(product_nodes[product_feature_cols].values, dtype=torch.float32)
        
        if not persona_feature_cols:
            print(f"  [Warning] No persona features found, creating random {feature_dim}-dim features")
            persona_features = torch.randn(len(persona_nodes), feature_dim)
        else:
            persona_features = torch.tensor(persona_nodes[persona_feature_cols].values, dtype=torch.float32)

                               
        graph.nodes['user'].data['h'] = user_features
        graph.nodes['product'].data['h'] = product_features
        graph.nodes['persona'].data['h'] = persona_features
        graph.nodes['user'].data['feature'] = user_features
        graph.nodes['product'].data['feature'] = product_features
        graph.nodes['persona'].data['feature'] = persona_features
        

        print(f"  [Features] User: {user_features.shape}, product: {product_features.shape}, persona: {persona_features.shape}")

        target_node_type = 'user'

        print(f"  [Debug] Label column dtype: {label['labels'].dtype}")
        print(f"  [Debug] First 5 labels: {label['labels'].head().tolist()}")

                                                                    
        print("  [Info] Scanning labels to determine num_classes...")
        max_label_id = -1
        for label_str in label['labels']:
            if pd.notna(label_str):
                label_str = str(label_str).strip()
                if ',' in label_str:
                    label_ids = [int(x.strip()) for x in label_str.split(',')]
                else:
                    try:
                        label_ids = [int(label_str)]
                    except ValueError:
                        continue
                if label_ids:
                    max_label_id = max(max_label_id, max(label_ids))

        self.num_classes = max_label_id + 1
        print(f"  [Info] Detected num_classes: {self.num_classes}")

                                                               
        print("  [Info] Converting labels to multi-hot vectors...")
        multi_hot_labels = np.array([
            parse_multi_labels(label_str, self.num_classes)
            for label_str in label['labels']
        ])

                                                                       
        labels = torch.tensor(multi_hot_labels, dtype=torch.float32)
        graph.nodes['user'].data['label'] = labels

        print(f"  [Labels] User labels shape: {labels.shape}, Num classes: {self.num_classes}")
        print(f"  [Labels] Sample label vector: {labels[0][:20]}...")                      
                                               
        self.target_node_type = target_node_type
                                                     
        target_nodes_df = user_nodes 

        if 'train_mask' in target_nodes_df.columns:
            graph.nodes[target_node_type].data['train_mask'] = torch.tensor(
                target_nodes_df['train_mask'].values, dtype=torch.bool)
            graph.nodes[target_node_type].data['val_mask'] = torch.tensor(
                target_nodes_df['val_mask'].values, dtype=torch.bool)
            graph.nodes[target_node_type].data['test_mask'] = torch.tensor(
                target_nodes_df['test_mask'].values, dtype=torch.bool)
            print("  [Masks] Loaded train/val/test masks from CSV")

        print(f"[Persona Dataset] Loading completed!\n")
        return graph


if __name__ == '__main__':
                              
    dataset = PersonaDataset(name='persona', data_path='./openhgnn/dataset/data/persona/')
    print("Dataset loaded successfully!")
    print(f"Graph: {dataset.graph}")
    print(f"Num classes: {dataset.num_classes}")
    print(f"Target node type: {dataset.target_node_type}")
