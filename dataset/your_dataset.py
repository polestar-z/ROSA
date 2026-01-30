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


class YourDataset:
    def __init__(self, name, raw_dir=''):
        self.name = name
        self.raw_dir = raw_dir
        self.graph, self.category, self.num_classes, self.in_dim = self.load_data()

    def load_data(self):
                
                
        paper_nodes = pd.read_csv("./openhgnn/dataset/data/chemistry/nodes/32/word2vec/title+summary_word2vec_embedding16+16.csv",encoding='latin-1')                                                                                                                                
                                                                                                                                                               
        keywords_nodes = pd.read_csv("./openhgnn/dataset/data/chemistry/nodes/32/word2vec/keywords_word2vec_embedding32.csv")                     
                                                                                                                             

                
        label_file_path = "./openhgnn/dataset/data/chemistry/nodes/128/id+multilabel.csv"
        paper_labels = pd.read_csv(label_file_path)

        substrate_nodes = pd.read_csv("./openhgnn/dataset/data/chemistry/nodes/32/word2vec/substrate_cleaned_embedding32.csv")                     
                                                                                                                                         
                                                                                                                                                   
                                                                                                                                                     
        
               
                                                                                                                                 
                                                                                                                              
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

        print(f"[Data Load] Paper nodes: {num_papers}")
        print(f"[Data Load] Keywords nodes: {num_keywords}")
        print(f"[Data Load] Substrate nodes: {num_substrates}")

                                      
        before_filter = len(paper_keywords_edges)
        paper_keywords_edges = paper_keywords_edges[
            (paper_keywords_edges['paper_id'] < num_papers) &
            (paper_keywords_edges['keywords_id'] < num_keywords)
        ]
        print(f"[Data Filter] paper_keywords edges: {before_filter} -> {len(paper_keywords_edges)} (filtered {before_filter - len(paper_keywords_edges)})")

                                   
        before_filter = len(paper_paper_edges)
        paper_paper_edges = paper_paper_edges[
            (paper_paper_edges['source'] < num_papers) &
            (paper_paper_edges['target'] < num_papers)
        ]
        print(f"[Data Filter] paper_paper edges: {before_filter} -> {len(paper_paper_edges)} (filtered {before_filter - len(paper_paper_edges)})")

                                       
        before_filter = len(paper_substrate_edges)
        paper_substrate_edges = paper_substrate_edges[
            (paper_substrate_edges['paper_id'] < num_papers) &
            (paper_substrate_edges['substrate_id'] < num_substrates)
        ]
        print(f"[Data Filter] paper_substrate edges: {before_filter} -> {len(paper_substrate_edges)} (filtered {before_filter - len(paper_substrate_edges)})")

                                           
        if len(paper_substrate_edges) > 0:
            substrate_degree = paper_substrate_edges['substrate_id'].value_counts().sort_values(ascending=False)
            print(f"\n[Degree Distribution] Substrate node degree stats:")
            print(f"  Mean degree: {substrate_degree.mean():.2f}")
            print(f"  Median degree: {substrate_degree.median():.2f}")
            print(f"  Max degree: {substrate_degree.max()}")
            print(f"  Nodes with degree > 100: {(substrate_degree > 100).sum()}")
            print(f"  Nodes with degree > 500: {(substrate_degree > 500).sum()}")
            print(f"  Nodes with degree > 1000: {(substrate_degree > 1000).sum()}")
            print(f"\n  Top 10 high-degree substrate nodes (ID: degree):")
            for idx, (prod_id, degree) in enumerate(substrate_degree.head(10).items()):
                print(f"    {idx+1}. substrate_{prod_id}: {degree} papers (potentially {degree*degree:,} POP edges)")

                           
            total_pop_edges = (substrate_degree ** 2).sum()
            print(f"\n  [Warning] Theoretical POP meta-path edges: {total_pop_edges:,}")
            if total_pop_edges > 10_000_000:
                print(f"  [Critical Warning] Too many POP edges, may cause OOM. Filtering high-degree nodes...")

                                              
            MAX_substrate_DEGREE = 5000           
            high_degree_substrates = substrate_degree[substrate_degree > MAX_substrate_DEGREE].index.tolist()

            if len(high_degree_substrates) > 0:
                before_filter = len(paper_substrate_edges)
                paper_substrate_edges = paper_substrate_edges[
                    ~paper_substrate_edges['substrate_id'].isin(high_degree_substrates)
                ]
                print(f"\n  [Filter High Degree] Removed {len(high_degree_substrates)} substrates with degree > {MAX_substrate_DEGREE}")
                print(f"  [Filter High Degree] paper_substrate edges: {before_filter} -> {len(paper_substrate_edges)} (filtered {before_filter - len(paper_substrate_edges)})")

                           
                substrate_degree_filtered = paper_substrate_edges['substrate_id'].value_counts()
                total_pop_edges_filtered = (substrate_degree_filtered ** 2).sum()
                print(f"  [After Filter] POP meta-path edges: {total_pop_edges_filtered:,}")

               
        g = dgl.heterograph({
                                                                                                                                 
                                                                                                                                 
            ('paper', 'pk', 'keywords'): (paper_keywords_edges['paper_id'].values, paper_keywords_edges['keywords_id'].values),
            ('keywords', 'kp', 'paper'): (paper_keywords_edges['keywords_id'].values, paper_keywords_edges['paper_id'].values),
                                                                                                                             
                                                                                                                             
            ('paper', 'pp', 'paper'): (paper_paper_edges['source'].values, paper_paper_edges['target'].values),
            ('paper', 'pp', 'paper'): (paper_paper_edges['source'].values, paper_paper_edges['target'].values),           
                                             
            ('paper', 'ps', 'substrate'): (paper_substrate_edges['paper_id'].values, paper_substrate_edges['substrate_id'].values),
            ('substrate', 'sp', 'paper'): (paper_substrate_edges['substrate_id'].values, paper_substrate_edges['paper_id'].values),

                                                                                                                             
                                                                                                                             
            
                                                                                                                                   
                                                                                                                                   
            
                                                                                                                                           
                                                                                                                                          
        
        })
        

                
                                 
                                                                     
        paper_columns = [f'title+summary_embedding_{i}' for i in range(32)]
                                                                              
                                                                     
        keywords_columns = [f'term_embedding_{i}' for i in range(32)]
                      
        substrate_columns = [f'substrate_embedding_{i}' for i in range(32)]        
                                                                         
                                                                           
                                                                                     

                                                                         


                                                                      

                     
                                                                                                              
        g.nodes['paper'].data['h'] = torch.tensor(paper_nodes[paper_columns].values, dtype=torch.float32)
        g.nodes['keywords'].data['h'] = torch.tensor(keywords_nodes[keywords_columns].values, dtype=torch.float32)
                                                                                                                 
                               
        g.nodes['substrate'].data['h'] = torch.tensor(substrate_nodes[substrate_columns].values, dtype=torch.float32)
                                                                                                                 
                                                                                                                    
                                                                                                                    


                                          
                                          
                                   
        g.meta_paths_dict = {
                                    
                                                                            
                                      
                                                                                
            'PSP': [('paper', 'ps', 'substrate'), ('substrate', 'sp', 'paper')],
                                                                              
                                                                                
                                                                                
                                     
                                                                              
                                                                                                  
                                               
                                   
                                       
                                                          
                                                               
                                   
                                                                      
                             
                                                                                              
                                                                                                 
                                            
                                               
                                               
                                                         
        }

                                           
                               
        g.target_node_type = 'paper'                          

                       

              
                                       
        import time
        start_time = time.time()

        sample_label = paper_labels['label'].dropna().iloc[0] if len(paper_labels['label'].dropna()) > 0 else "0"
        is_multi_label = ',' in str(sample_label)

                                             
        if 'multilabel' in label_file_path and not is_multi_label:
            print("[Warning] Filename contains 'multilabel' but no comma-separated format detected; forcing multi-label parsing")
            is_multi_label = True

        if is_multi_label:
            print("[Data Load] Detected multi-label format; parsing to multi-hot vectors")
            labels_array = np.array([
                parse_multi_labels(x, num_classes=107)
                for x in paper_labels['label'].values
            ])
            labels = th.FloatTensor(labels_array)                   
            parse_time = time.time() - start_time
            print(f"[Data Load] Label shape: {labels.shape}, avg labels per sample: {labels.sum(dim=1).mean():.2f}")
            print(f"[Timing] Label parsing time: {parse_time:.2f} seconds ({len(paper_labels)} samples)")
        else:
            print("[Data Load] Detected single-label format; using LongTensor")
            labels = th.LongTensor(paper_labels['label'].values)              
            print(f"[Data Load] Label shape: {labels.shape}")


                     
        num_nodes = g.number_of_nodes('paper') 
        train_idx, val_idx, test_idx = split_indices(num_nodes)
        train_mask = get_binary_mask(num_nodes, train_idx)
        val_mask = get_binary_mask(num_nodes, val_idx)
        test_mask = get_binary_mask(num_nodes, test_idx)
        
        g.nodes['paper'].data['labels'] = labels
        g.nodes['paper'].data['train_mask'] = train_mask
        g.nodes['paper'].data['val_mask'] = val_mask
        g.nodes['paper'].data['test_mask'] = test_mask
        
        return g, 'paper', 107, g.nodes['paper'].data['h'].shape[1]
    
def split_indices(num_nodes, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
                           
        assert train_ratio + val_ratio + test_ratio == 1.0, "The sum of ratios must be 1"

        all_indices = np.random.permutation(num_nodes)
        
        train_size = int(num_nodes * train_ratio)
        val_size = int(num_nodes * val_ratio)
        
        train_idx = all_indices[:train_size]
        val_idx = all_indices[train_size:train_size+val_size]
        test_idx = all_indices[train_size+val_size:]
        
        return train_idx, val_idx, test_idx
