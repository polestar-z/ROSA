                      
                     
                                  

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

    Examples:
        >>> parse_multi_labels("0,2,5", 10)
        array([1., 0., 1., 0., 0., 1., 0., 0., 0., 0.])
        >>> parse_multi_labels("3", 10)
        array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
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


class IMDBDataset(BaseDataset):
    """
    IMDB Dataset for heterogeneous graph node classification.

    Expected data structure:
    - nodes/: Contains CSV files for each node type (movie, actor, director, etc.)
    - edges/: Contains CSV files for each edge type

    Node files should contain:
    - id column (e.g., movie_id, actor_id)
    - feature columns (embeddings)
    - label column (for target node type)

    Edge files should contain:
    - source and target id columns
    """

    def __init__(self, name='imdb', data_path=None, logger=None, *args, **kwargs):
        """
        Parameters
        ----------
        name : str
            Dataset name
        data_path : str
            Base path to the dataset directory containing nodes/ and edges/ folders
            If None, will use default path: './openhgnn/dataset/data/imdb/'
        logger : optional
            Logger instance
        """
                                                     
        kwargs['logger'] = logger
        super(IMDBDataset, self).__init__(*args, **kwargs)
        self.name = name

                                               
        if data_path is None:
            self.data_path = './openhgnn/dataset/data/imdb/'
        else:
            self.data_path = data_path

        self.graph = self.load_graph()                                           

    def load_graph(self):
        """
        Load IMDB heterogeneous graph from CSV files.

        Returns
        -------
        graph : dgl.DGLHeteroGraph
            The constructed heterogeneous graph
        """
        print(f"[IMDB Dataset] Loading from {self.data_path}")

                                                      
                            
                                                      
                                                                   
        movie_nodes = pd.read_csv(f"{self.data_path}nodes/movie_embedding32.csv")
        actor_nodes = pd.read_csv(f"{self.data_path}nodes/actor_embedding32.csv")
        director_nodes = pd.read_csv(f"{self.data_path}nodes/director_embedding32.csv")
        keyword_nodes = pd.read_csv(f"{self.data_path}nodes/keyword_embedding32.csv")
        label = pd.read_csv(f"{self.data_path}label-final.csv")

        print(f"  [Nodes] Movie: {len(movie_nodes)}, Actor: {len(actor_nodes)}, Director: {len(director_nodes)}, Keyword: {len(keyword_nodes)}")

                                                      
                            
                                                      
        movie_actor_edges = pd.read_csv(f"{self.data_path}edges/movie-actor-final.csv")
        movie_director_edges = pd.read_csv(f"{self.data_path}edges/movie-director-final.csv")
        movie_keyword_edges = pd.read_csv(f"{self.data_path}edges/movie-keyword-final.csv")
                                       

        print(f"  [Edges] Movie-Actor: {len(movie_actor_edges)}, Movie-Director: {len(movie_director_edges)},Movie-Keyword: {len(movie_keyword_edges)}")

                                                      
                                      
                                                      
        data_dict = {
            ('movie', 'acted_by', 'actor'): (
                torch.tensor(movie_actor_edges['movie_id'].values, dtype=torch.long),
                torch.tensor(movie_actor_edges['actor_id'].values, dtype=torch.long)
            ),
            ('movie', 'directed_by', 'director'): (
                torch.tensor(movie_director_edges['movie_id'].values, dtype=torch.long),
                torch.tensor(movie_director_edges['director_id'].values, dtype=torch.long)
            ),
                                                          
            ('actor', 'act_in', 'movie'): (
                torch.tensor(movie_actor_edges['actor_id'].values, dtype=torch.long),
                torch.tensor(movie_actor_edges['movie_id'].values, dtype=torch.long)
            ),
            ('director', 'direct', 'movie'): (
                torch.tensor(movie_director_edges['director_id'].values, dtype=torch.long),
                torch.tensor(movie_director_edges['movie_id'].values, dtype=torch.long)
            ),
            ('keyword', 'in', 'movie'): (
                torch.tensor(movie_keyword_edges['keyword_id'].values, dtype=torch.long),
                torch.tensor(movie_keyword_edges['movie_id'].values, dtype=torch.long)
            ),
            ('movie', 'contain', 'keyword'): (                
                torch.tensor(movie_keyword_edges['movie_id'].values, dtype=torch.long),
                torch.tensor(movie_keyword_edges['keyword_id'].values, dtype=torch.long)
            ),
        }

        graph = dgl.heterograph(data_dict)
        print(f"  [Graph] Created with {graph.num_nodes()} nodes and {graph.num_edges()} edges")

                                                      
                              
                                                      
                                                                               
                                                      

                                                                                        
        movie_feature_cols = [col for col in movie_nodes.columns if 'embedding' in col or 'feature' in col]
        actor_feature_cols = [col for col in actor_nodes.columns if 'embedding' in col or 'feature' in col]
        director_feature_cols = [col for col in director_nodes.columns if 'embedding' in col or 'feature' in col]
        keyword_feature_cols = [col for col in keyword_nodes.columns if 'embedding' in col or 'feature' in col]

                                                             
        if not movie_feature_cols:
            print("  [Warning] No movie features found, creating random 128-dim features")
            movie_features = torch.randn(len(movie_nodes), 128)
        else:
            movie_features = torch.tensor(movie_nodes[movie_feature_cols].values, dtype=torch.float32)

        if not actor_feature_cols:
            print("  [Warning] No actor features found, creating random 128-dim features")
            actor_features = torch.randn(len(actor_nodes), 128)
        else:
            actor_features = torch.tensor(actor_nodes[actor_feature_cols].values, dtype=torch.float32)

        if not director_feature_cols:
            print("  [Warning] No director features found, creating random 128-dim features")
            director_features = torch.randn(len(director_nodes), 128)
        else:
            director_features = torch.tensor(director_nodes[director_feature_cols].values, dtype=torch.float32)

        if not keyword_feature_cols:
            print("  [Warning] No keyword features found, creating random 128-dim features")
            keyword_features = torch.randn(len(keyword_nodes), 128)
        else:
            keyword_features = torch.tensor(keyword_nodes[keyword_feature_cols].values, dtype=torch.float32)

        graph.nodes['movie'].data['h'] = movie_features
        graph.nodes['actor'].data['h'] = actor_features
        graph.nodes['director'].data['h'] = director_features
        graph.nodes['keyword'].data['h'] = keyword_features

                                                 
        graph.nodes['movie'].data['feature'] = movie_features
        graph.nodes['actor'].data['feature'] = actor_features
        graph.nodes['director'].data['feature'] = director_features
        graph.nodes['keyword'].data['feature'] = keyword_features

        print(f"  [Features] Movie: {movie_features.shape}, Actor: {actor_features.shape}, Director: {director_features.shape},keyword: {keyword_features.shape}")

                                                      
                                                                
                                                      
        if 'label' in label.columns:
            print(f"  [Debug] Label column dtype: {label['label'].dtype}")
            print(f"  [Debug] First 5 labels: {label['label'].head().tolist()}")

                                                                        
            print("  [Info] Scanning labels to determine num_classes...")
            max_label_id = -1
            for label_str in label['label']:
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
                for label_str in label['label']
            ])

                                                                           
            movie_labels = torch.tensor(multi_hot_labels, dtype=torch.float32)
            graph.nodes['movie'].data['label'] = movie_labels

            print(f"  [Labels] Movie labels shape: {movie_labels.shape}, Num classes: {self.num_classes}")
            print(f"  [Labels] Sample label vector: {movie_labels[0][:20]}...")                      
        else:
            print("  [Warning] No 'label' column found in label-final.csv")
            self.num_classes = 3                 

                                                      
                                                
                                                      
                                                                       
        if 'train_mask' in movie_nodes.columns:
            graph.nodes['movie'].data['train_mask'] = torch.tensor(movie_nodes['train_mask'].values, dtype=torch.bool)
            graph.nodes['movie'].data['val_mask'] = torch.tensor(movie_nodes['val_mask'].values, dtype=torch.bool)
            graph.nodes['movie'].data['test_mask'] = torch.tensor(movie_nodes['test_mask'].values, dtype=torch.bool)
            print("  [Masks] Loaded train/val/test masks from CSV")

        print(f"[IMDB Dataset] Loading completed!\n")
        return graph


if __name__ == '__main__':
                              
    dataset = IMDBDataset(name='imdb', data_path='./openhgnn/dataset/data/imdb/')
    print("Dataset loaded successfully!")
    print(f"Graph: {dataset.graph}")
    print(f"Num classes: {dataset.num_classes}")
