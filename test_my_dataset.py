                                            
from dataset import build_dataset
import logging

def test_my_dataset():
             
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
                               
    dataset = build_dataset('my_custom_node_classification', task='node_classification',logger=logger)

              
    print("Graph:", dataset.g)
    print("Target node type:", dataset.category)
    print("Number of classes:", dataset.num_classes)
    print("Has features:", dataset.has_feature)
    print("Feature shape:", dataset.g.nodes[dataset.category].data['feature'].shape)
    print("Train/val/test:")
    print(dataset.graph)
                                                       
                                                       
                                                     

if __name__ == "__main__":
    test_my_dataset()
