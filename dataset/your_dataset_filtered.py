import dgl
import torch as th
import torch
import pandas as pd
import numpy as np
from .utils import get_binary_mask


def parse_multi_labels(label_str, num_classes=107):
    """
    Convert a comma-separated label string to a multi-hot vector.

    Args:
        label_str: "0,2,5" or "3" or NaN
        num_classes: total number of labels (default: 107)

    Returns:
        multi_hot: numpy array [1,0,1,0,0,1,...] shape: (num_classes,)

    Examples:
        >>> parse_multi_labels("0,2,5", 107)
        array([1., 0., 1., 0., 0., 1., 0., ..., 0.])  107 dims
        >>> parse_multi_labels("3", 107)
        array([0., 0., 0., 1., 0., ..., 0.])  only index 3 is 1
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
                print(f"[Warning] Label '{label_str}' has IDs out of range [0, {num_classes})")
            multi_hot[valid_ids] = 1
        except (ValueError, TypeError) as e:
            print(f"[Error] Failed to parse label: '{label_str}', error: {e}")
    return multi_hot


def filter_rare_labels(labels_array, min_samples=1277, num_classes=107):
    """
    Filter rare labels and adjust samples.

    Strategy:
    1. If a sample has both rare and common labels -> keep the sample, drop rare labels
    2. If a sample has only rare labels -> remove the sample

    Args:
        labels_array: numpy array of shape [N, num_classes], multi-hot encoded
        min_samples: minimum samples per label
        num_classes: total number of labels

    Returns:
        filtered_labels: filtered label array
        valid_sample_indices: indices of kept samples
        rare_label_ids: ids of rare labels
    """
                      
    label_counts = labels_array.sum(axis=0)                

    print(f"\n{'='*60}")
    print(f"[Label Filter] Start filtering rare labels (threshold: {min_samples} samples)")
    print(f"{'='*60}")

               
    rare_label_mask = label_counts < min_samples
    rare_label_ids = np.where(rare_label_mask)[0]
    common_label_ids = np.where(~rare_label_mask)[0]

    print(f"[Stats] Total label classes: {num_classes}")
    print(f"[Stats] Rare labels: {len(rare_label_ids)} (samples < {min_samples})")
    print(f"[Stats] Common labels: {len(common_label_ids)} (samples >= {min_samples})")

    if len(rare_label_ids) > 0:
        print(f"\n[Rare Label Details] Labels to be processed:")
        for label_id in rare_label_ids:
            count = int(label_counts[label_id])
            print(f"  - Label {label_id}: {count} samples")

                  
    filtered_labels = labels_array.copy()
    valid_sample_mask = np.ones(len(labels_array), dtype=bool)

    samples_with_rare_only = 0
    samples_with_mixed = 0
    rare_labels_removed_count = 0

    for i in range(len(labels_array)):
        sample_label_ids = np.where(labels_array[i] > 0)[0]

        if len(sample_label_ids) == 0:
                               
            continue

                    
        has_rare = any(lid in rare_label_ids for lid in sample_label_ids)
        has_common = any(lid in common_label_ids for lid in sample_label_ids)

        if has_rare and not has_common:
                                
            valid_sample_mask[i] = False
            samples_with_rare_only += 1
        elif has_rare and has_common:
                                     
            for rare_id in rare_label_ids:
                if filtered_labels[i, rare_id] > 0:
                    filtered_labels[i, rare_id] = 0
                    rare_labels_removed_count += 1
            samples_with_mixed += 1

             
    valid_sample_indices = np.where(valid_sample_mask)[0]
    filtered_labels = filtered_labels[valid_sample_indices]

    print(f"\n[Results]")
    print(f"  - Removed samples (rare-only): {samples_with_rare_only}")
    print(f"  - Kept samples (mixed labels): {samples_with_mixed}")
    print(f"  - Rare labels removed from mixed samples: {rare_labels_removed_count}")
    print(f"  - Total kept samples: {len(valid_sample_indices)} / {len(labels_array)}")
    print(f"  - Filter ratio: {(1 - len(valid_sample_indices)/len(labels_array))*100:.2f}%")

                   
    new_label_counts = filtered_labels.sum(axis=0)
    print(f"\n[Validation] Label stats after filtering:")
    print(f"  - Avg labels per sample: {filtered_labels.sum(axis=1).mean():.2f}")
    print(f"  - Samples with 0 labels: {(filtered_labels.sum(axis=1) == 0).sum()}")

    remaining_rare = np.where(new_label_counts < min_samples)[0]
    if len(remaining_rare) > 0:
        print(f"  - [Warning] {len(remaining_rare)} labels still have samples < {min_samples}:")
        for label_id in remaining_rare:
            print(f"      Label {label_id}: {int(new_label_counts[label_id])} samples")
    else:
        print(f"  - [OK] All labels have samples >= {min_samples}")

    print(f"{'='*60}\n")

    return filtered_labels, valid_sample_indices, rare_label_ids


def remap_node_ids(valid_indices, edge_dataframes, id_column_names):
    """
    Remap node IDs.

    Args:
        valid_indices: kept paper node indices
        edge_dataframes: list of edge DataFrames containing paper_id columns
        id_column_names: list of column names for paper_id in each DataFrame

    Returns:
        id_mapping: dict {old_id: new_id}
        updated_edges: list of updated edge DataFrames
    """
                             
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(valid_indices)}

    print(f"[ID Remap] Mapping created: {len(valid_indices)} valid paper nodes")

    updated_edges = []
    for df, col_names in zip(edge_dataframes, id_column_names):
        df_updated = df.copy()

                    
        for col_name in col_names:
            if col_name in df_updated.columns:
                                
                before_len = len(df_updated)
                df_updated = df_updated[df_updated[col_name].isin(valid_indices)]
                after_len = len(df_updated)

                        
                df_updated[col_name] = df_updated[col_name].map(id_mapping)

                if before_len != after_len:
                    print(f"  - Edge type '{col_name}': {before_len} -> {after_len} (removed {before_len - after_len})")

        updated_edges.append(df_updated)

    return id_mapping, updated_edges


class YourDatasetFiltered:
    """
    Dataset variant with rare-label filtering.

    Strategy:
    1. Count samples per label
    2. For labels with samples < min_samples:
       - If a sample has both rare and common labels -> keep the sample, drop rare labels
       - If a sample has only rare labels -> remove the sample and its related edges
    """

    def __init__(self, name, raw_dir='', min_label_samples=1277):
        """
        Args:
            name: dataset name
            raw_dir: data directory
            min_label_samples: minimum samples per label (default: 10)
        """
        self.name = name
        self.raw_dir = raw_dir
        self.min_label_samples = min_label_samples
        self.graph, self.category, self.num_classes, self.in_dim = self.load_data()

    def load_data(self):
                
                
        paper_nodes = pd.read_csv("./openhgnn/dataset/data/chemistry/nodes/32/paper_embedding32.csv",encoding='latin-1')                                                                                                                                
                                                                                                                                             
        keywords_nodes = pd.read_csv("./openhgnn/dataset/data/chemistry/nodes/32/keywords_embedding32.csv")                     
                                                                                                                             
        paper_labels = pd.read_csv("./openhgnn/dataset/data/chemistry/nodes/128/id+multilabel.csv")

        substrate_nodes = pd.read_csv("./openhgnn/dataset/data/chemistry/nodes/32/substrate_cleaned_embedding32.csv")                     
                                                                                                                                         
                                                                                                                                                   
                                                                                                                                                     
        
               
                                                                                                                                 

                                                      
                                                          
                                                                                         
                                                           
                                                                                           

                                                                                                                              
        paper_keywords_edges = pd.read_csv("./openhgnn/dataset/data/chemistry/edges/paper_keywords.csv")                         
        paper_paper_edges = pd.read_csv("./openhgnn/dataset/data/chemistry/edges/paper_paper.csv")                      
        
                         
        paper_substrate_edges = pd.read_csv("./openhgnn/dataset/data/chemistry/edges/paper-substrate-final_cleaned.csv")
                                                                                                              
                                                                                                                  
                                                                                                                           

                                           
        paper_substrate_edges['substrate_id'] = pd.to_numeric(paper_substrate_edges['substrate_id'], errors='coerce')

                     
        paper_substrate_edges = paper_substrate_edges.dropna(subset=['substrate_id'])

                              
        paper_substrate_edges['substrate_id'].fillna(0, inplace=True)

                             
        paper_substrate_edges['substrate_id'] = paper_substrate_edges['substrate_id'].astype(np.int32)

                                       
                     
        num_papers = len(paper_nodes)
        num_keywords = len(keywords_nodes)
        num_substrates = len(substrate_nodes)

                                                 
                                                      
                                                         

                                      
        before_filter = len(paper_keywords_edges)
        paper_keywords_edges = paper_keywords_edges[
            (paper_keywords_edges['paper_id'] < num_papers) &
            (paper_keywords_edges['keywords_id'] < num_keywords)
        ]
                                                                                                                                              

                                   
        before_filter = len(paper_paper_edges)
        paper_paper_edges = paper_paper_edges[
            (paper_paper_edges['source'] < num_papers) &
            (paper_paper_edges['target'] < num_papers)
        ]
                                                                                                                                     

                                       
        before_filter = len(paper_substrate_edges)
        paper_substrate_edges = paper_substrate_edges[
            (paper_substrate_edges['paper_id'] < num_papers) &
            (paper_substrate_edges['substrate_id'] < num_substrates)
        ]
                                                                                                                                                 

                                           
        if len(paper_substrate_edges) > 0:
            substrate_degree = paper_substrate_edges['substrate_id'].value_counts().sort_values(ascending=False)
                                               
                                                            
                                                              
                                                       
                                                                     
                                                                     
                                                                       
                                                         
                                                                                         
                                                                                                            

                           
            total_pop_edges = (substrate_degree ** 2).sum()
                                                                  
            if total_pop_edges > 10_000_000:
                print(f"  [Critical Warning] Too many POP edges, may cause OOM. Filtering high-degree nodes...")

                                              
            MAX_substrate_DEGREE = 5000           
            high_degree_substrates = substrate_degree[substrate_degree > MAX_substrate_DEGREE].index.tolist()

            if len(high_degree_substrates) > 0:
                before_filter = len(paper_substrate_edges)
                paper_substrate_edges = paper_substrate_edges[
                    ~paper_substrate_edges['substrate_id'].isin(high_degree_substrates)
                ]
                                                                                                           
                                                                                                                                                             

                           
                substrate_degree_filtered = paper_substrate_edges['substrate_id'].value_counts()
                total_pop_edges_filtered = (substrate_degree_filtered ** 2).sum()
                                                                            

                                    
        print(f"\n[Raw Data Stats]")
        print(f"  - paper nodes: {num_papers}")
        print(f"  - keywords nodes: {num_keywords}")
        print(f"  - substrate nodes: {num_substrates}")
                                                      
        print(f"  - paper_keywords edges: {len(paper_keywords_edges)}")
        print(f"  - paper_paper edges: {len(paper_paper_edges)}")
        print(f"  - paper_substrate edges: {len(paper_substrate_edges)}")
                                                                 

                    
        sample_label = paper_labels['label'].dropna().iloc[0] if len(paper_labels['label'].dropna()) > 0 else "0"
        is_multi_label = ',' in str(sample_label)

        if is_multi_label:
            print("\n[Data Load] Detected multi-label format; parsing to multi-hot vectors")
            labels_array = np.array([
                parse_multi_labels(x, num_classes=107)
                for x in paper_labels['label'].values
            ])
            print(f"[Data Load] Raw label shape: {labels_array.shape}, avg labels per sample: {labels_array.mean(axis=0).sum():.2f}")
        else:
            print("\n[Data Load] Detected single-label format; using LongTensor")
                                   
            labels_array = np.zeros((len(paper_labels), 107), dtype=np.float32)
            for i, label in enumerate(paper_labels['label'].values):
                if pd.notna(label):
                    labels_array[i, int(label)] = 1
            print(f"[Data Load] Raw label shape: {labels_array.shape}")

                                    
        filtered_labels, valid_sample_indices, rare_label_ids = filter_rare_labels(
            labels_array,
            min_samples=self.min_label_samples,
            num_classes=107
        )

                                         
        paper_nodes = paper_nodes.iloc[valid_sample_indices].reset_index(drop=True)
        print(f"\n[Node Filter] paper nodes: {num_papers} -> {len(paper_nodes)}")

                                                            
                               
                                                                          
                                                                                                 

                                   
                                                                                    

                                         
                                                                                                         

                                               
                                                                                                          

                                                          

                                                
        print(f"\n[Edge Update] Start remapping paper IDs...")
        id_mapping, updated_edges = remap_node_ids(
            valid_sample_indices,
            [paper_keywords_edges, paper_paper_edges, paper_substrate_edges],                     
            [['paper_id'], ['source', 'target'], ['paper_id'], ['paper_new_id']]
        )

        paper_keywords_edges, paper_paper_edges, paper_substrate_edges = updated_edges                      

        print(f"\n[Post-filter Stats]")
        print(f"  - paper nodes: {len(paper_nodes)}")
        print(f"  - paper_keywords edges: {len(paper_keywords_edges)}")
        print(f"  - paper_paper edges: {len(paper_paper_edges)}")
        print(f"  - paper_substrate edges: {len(paper_substrate_edges)}")
                                                                 
        print(f"  - labels kept: {(filtered_labels.sum(axis=0) > 0).sum()} / 107")

                                     
        g = dgl.heterograph({
            ('paper', 'pk', 'keywords'): (paper_keywords_edges['paper_id'].values, paper_keywords_edges['keywords_id'].values),
            ('keywords', 'kp', 'paper'): (paper_keywords_edges['keywords_id'].values, paper_keywords_edges['paper_id'].values),
            ('paper', 'pp', 'paper'): (paper_paper_edges['source'].values, paper_paper_edges['target'].values),
            ('paper', 'ps', 'substrate'): (paper_substrate_edges['paper_id'].values, paper_substrate_edges['substrate_id'].values),
            ('substrate', 'sp', 'paper'): (paper_substrate_edges['substrate_id'].values, paper_substrate_edges['paper_id'].values),
                                                                                                                                 
                                                                                                                                 
        })

        print(f"\n[Graph Build Complete]")
        print(f"  - node types: {g.ntypes}")
        print(f"  - edge types: {g.canonical_etypes}")
        for ntype in g.ntypes:
            print(f"  - {ntype} nodes: {g.number_of_nodes(ntype)}")

                
        paper_columns = [f'title+abstract_embedding_{i}' for i in range(32)]
        keywords_columns = [f'term_embedding_{i}' for i in range(32)]
        substrate_columns = [f'substrate_embedding_{i}' for i in range(32)]
                                                                     

        g.nodes['paper'].data['h'] = torch.tensor(paper_nodes[paper_columns].values, dtype=torch.float32)
        g.nodes['keywords'].data['h'] = torch.tensor(keywords_nodes[keywords_columns].values, dtype=torch.float32)
        g.nodes['substrate'].data['h'] = torch.tensor(substrate_nodes[substrate_columns].values, dtype=torch.float32)
                                                                                                              

               
        g.meta_paths_dict = {
            'PSP': [('paper', 'ps', 'substrate'), ('substrate', 'sp', 'paper')],
                                                                            
        }
        g.target_node_type = 'paper'

                     
        labels = th.FloatTensor(filtered_labels)

                     
        num_nodes = g.number_of_nodes('paper')
        train_idx, val_idx, test_idx = split_indices(num_nodes)
        train_mask = get_binary_mask(num_nodes, train_idx)
        val_mask = get_binary_mask(num_nodes, val_idx)
        test_mask = get_binary_mask(num_nodes, test_idx)

        g.nodes['paper'].data['labels'] = labels
        g.nodes['paper'].data['train_mask'] = train_mask
        g.nodes['paper'].data['val_mask'] = val_mask
        g.nodes['paper'].data['test_mask'] = test_mask

        print(f"\n[Dataset Ready]")
        print(f"  - feature dim: {g.nodes['paper'].data['h'].shape[1]}")
        print(f"  - label dim: {labels.shape[1]}")
        print(f"  - train: {train_mask.sum().item()} samples")
        print(f"  - val: {val_mask.sum().item()} samples")
        print(f"  - test: {test_mask.sum().item()} samples")

        return g, 'paper', 107, g.nodes['paper'].data['h'].shape[1]


def split_indices(num_nodes, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """Randomly split train/val/test."""
    assert train_ratio + val_ratio + test_ratio == 1.0, "The sum of ratios must be 1"

    all_indices = np.random.permutation(num_nodes)

    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    train_idx = all_indices[:train_size]
    val_idx = all_indices[train_size:train_size+val_size]
    test_idx = all_indices[train_size+val_size:]

    return train_idx, val_idx, test_idx
