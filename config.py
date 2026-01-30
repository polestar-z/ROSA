import configparser
import warnings

import torch as th


class Config(object):
    def __init__(self, file_path, model, dataset, task, gpu):
        conf = configparser.ConfigParser()
        conf.read(file_path)

        self.seed = 0
        self.patience = 1
        self.max_epoch = 1
        self.task = task
        self.model = model
        self.dataset = dataset

        if isinstance(dataset, str):
            self.dataset_name = dataset
        else:
            self.dataset_name = self.dataset.name
        if isinstance(model, str):
            self.model_name = model
        else:
            self.model_name = type(self.model).__name__

        self.optimizer = "Adam"

                          
        self.lr = conf.getfloat("General", "learning_rate", fallback=0.01)
        self.weight_decay = conf.getfloat("General", "weight_decay", fallback=0.0)
        self.dropout = conf.getfloat("General", "dropout", fallback=0.0)
        self.hidden_dim = conf.getint("General", "hidden_dim", fallback=64)
        self.max_epoch = conf.getint("General", "max_epoch", fallback=self.max_epoch)
        self.patience = conf.getint("General", "patience", fallback=self.patience)
        self.mini_batch_flag = conf.getboolean("General", "mini_batch_flag", fallback=False)
        self.batch_size = conf.getint("General", "batch_size", fallback=64)
        self.num_workers = conf.getint("General", "num_workers", fallback=0)
        self.fanout = conf.getint("General", "fanout", fallback=-1)
        self.use_uva = conf.getboolean("General", "use_uva", fallback=False)
        self.norm = conf.getboolean("General", "norm", fallback=False)
        self.multi_label = conf.getboolean("General", "multi_label", fallback=True)
        self.threshold = conf.getfloat("General", "threshold", fallback=0.5)

        self.graphbolt = False
        self.test_flag = True
        self.prediction_flag = False

        if isinstance(model, th.nn.Module):
            pass
        elif self.model_name == "HAN":
            self.lr = conf.getfloat("HAN", "learning_rate")
            self.weight_decay = conf.getfloat("HAN", "weight_decay")
            self.seed = conf.getint("HAN", "seed")
            self.dropout = conf.getfloat("HAN", "dropout")
            self.hidden_dim = conf.getint("HAN", "hidden_dim")
            self.out_dim = conf.getint("HAN", "out_dim")
            num_heads = conf.get("HAN", "num_heads").split("-")
            self.num_heads = [int(i) for i in num_heads]
            self.patience = conf.getint("HAN", "patience")
            self.max_epoch = conf.getint("HAN", "max_epoch")
            self.mini_batch_flag = conf.getboolean("HAN", "mini_batch_flag")

        elif self.model_name == "MAGNN_Multi":
            self.lr = conf.getfloat("MAGNN_Multi", "learning_rate")
            self.weight_decay = conf.getfloat("MAGNN_Multi", "weight_decay")
            self.seed = conf.getint("MAGNN_Multi", "seed")
            self.dropout = conf.getfloat("MAGNN_Multi", "dropout")
            self.hidden_dim = conf.getint("MAGNN_Multi", "hidden_dim")
            self.out_dim = conf.getint("MAGNN_Multi", "out_dim")
            self.inter_attn_feats = conf.getint("MAGNN_Multi", "inter_attn_feats")
            self.num_heads = conf.getint("MAGNN_Multi", "num_heads")
            self.num_layers = conf.getint("MAGNN_Multi", "num_layers")
            self.patience = conf.getint("MAGNN_Multi", "patience")
            self.max_epoch = conf.getint("MAGNN_Multi", "max_epoch")
            self.encoder_type = conf.get("MAGNN_Multi", "encoder_type")
            self.mini_batch_flag = conf.getboolean("MAGNN_Multi", "mini_batch_flag")
            self.batch_size = conf.getint("MAGNN_Multi", "batch_size")
            self.num_samples = conf.getint("MAGNN_Multi", "num_samples")
            self.num_workers = conf.getint("MAGNN_Multi", "num_workers")
            self.multi_label = conf.getboolean("MAGNN_Multi", "multi_label")
            self.threshold = conf.getfloat("MAGNN_Multi", "threshold")

        elif self.model_name == "HGT":
            self.lr = conf.getfloat("HGT", "learning_rate")
            self.weight_decay = conf.getfloat("HGT", "weight_decay")
            self.seed = conf.getint("HGT", "seed")
            self.dropout = conf.getfloat("HGT", "dropout")
            self.batch_size = conf.getint("HGT", "batch_size")
            self.hidden_dim = conf.getint("HGT", "hidden_dim")
            self.out_dim = conf.getint("HGT", "out_dim")
            self.num_heads = conf.getint("HGT", "num_heads")
            self.patience = conf.getint("HGT", "patience")
            self.max_epoch = conf.getint("HGT", "max_epoch")
            self.num_workers = conf.getint("HGT", "num_workers")
            self.mini_batch_flag = conf.getboolean("HGT", "mini_batch_flag")
            self.fanout = conf.getint("HGT", "fanout")
            self.norm = conf.getboolean("HGT", "norm")
            self.num_layers = conf.getint("HGT", "num_layers")
            self.use_uva = conf.getboolean("HGT", "use_uva")

        elif self.model_name == "ROSA":
            self.lr = conf.getfloat("ROSA", "learning_rate")
            self.weight_decay = conf.getfloat("ROSA", "weight_decay")
            self.seed = conf.getint("ROSA", "seed")
            self.dropout = conf.getfloat("ROSA", "dropout")
            self.batch_size = conf.getint("ROSA", "batch_size")
            self.hidden_dim = conf.getint("ROSA", "hidden_dim")
            self.out_dim = conf.getint("ROSA", "out_dim")
            self.hgt_num_heads = conf.getint("ROSA", "hgt_num_heads")
            self.patience = conf.getint("ROSA", "patience")
            self.max_epoch = conf.getint("ROSA", "max_epoch")
            self.num_workers = conf.getint("ROSA", "num_workers")
            self.mini_batch_flag = conf.getboolean("ROSA", "mini_batch_flag")
            self.fanout = conf.getint("ROSA", "fanout")
            self.norm = conf.getboolean("ROSA", "norm")
            self.num_layers = conf.getint("ROSA", "num_layers")
            self.han_num_heads = conf.getint("ROSA", "han_num_heads")
            self.use_uva = conf.getboolean("ROSA", "use_uva")

            self.fusion_type = conf.get("ROSA", "fusion_type")
            self.gate_type = conf.get("ROSA", "gate_type")
            self.residual_type = conf.get("ROSA", "residual_type")
            self.use_consistency_loss = conf.getboolean("ROSA", "use_consistency_loss")
            self.consistency_loss_type = conf.get("ROSA", "consistency_loss_type")
            self.consistency_loss_weight = conf.getfloat("ROSA", "consistency_loss_weight")
            self.consistency_temperature = conf.getfloat("ROSA", "consistency_temperature")
            self.lambda_head_l1 = conf.getfloat("ROSA", "lambda_head_l1")
            self.asl_gamma_neg = conf.getfloat("ROSA", "asl_gamma_neg")
            self.asl_gamma_pos = conf.getfloat("ROSA", "asl_gamma_pos")
            self.asl_clip = conf.getfloat("ROSA", "asl_clip")

        elif self.model_name == "HGT_Multi":
            self.lr = conf.getfloat("HGT_Multi", "learning_rate")
            self.weight_decay = conf.getfloat("HGT_Multi", "weight_decay")
            self.seed = conf.getint("HGT_Multi", "seed")
            self.dropout = conf.getfloat("HGT_Multi", "dropout")
            self.batch_size = conf.getint("HGT_Multi", "batch_size")
            self.hidden_dim = conf.getint("HGT_Multi", "hidden_dim")
            self.out_dim = conf.getint("HGT_Multi", "out_dim")
            self.num_heads = conf.getint("HGT_Multi", "num_heads")
            self.patience = conf.getint("HGT_Multi", "patience")
            self.max_epoch = conf.getint("HGT_Multi", "max_epoch")
            self.num_workers = conf.getint("HGT_Multi", "num_workers")
            self.mini_batch_flag = conf.getboolean("HGT_Multi", "mini_batch_flag")
            self.fanout = conf.getint("HGT_Multi", "fanout")
            self.norm = conf.getboolean("HGT_Multi", "norm")
            self.num_layers = conf.getint("HGT_Multi", "num_layers")
            self.use_uva = conf.getboolean("HGT_Multi", "use_uva")
            self.multi_label = conf.getboolean("HGT_Multi", "multi_label")
            self.threshold = conf.getfloat("HGT_Multi", "threshold")

        elif self.model_name == "HAN_Multi":
            self.lr = conf.getfloat("HAN_Multi", "learning_rate")
            self.weight_decay = conf.getfloat("HAN_Multi", "weight_decay")
            self.seed = conf.getint("HAN_Multi", "seed")
            self.dropout = conf.getfloat("HAN_Multi", "dropout")
            self.batch_size = conf.getint("HAN_Multi", "batch_size")
            self.hidden_dim = conf.getint("HAN_Multi", "hidden_dim")
            self.out_dim = conf.getint("HAN_Multi", "out_dim")
            num_heads = conf.get("HAN_Multi", "num_heads").split("-")
            self.num_heads = [int(i) for i in num_heads]
            self.patience = conf.getint("HAN_Multi", "patience")
            self.max_epoch = conf.getint("HAN_Multi", "max_epoch")
            self.num_workers = conf.getint("HAN_Multi", "num_workers")
            self.mini_batch_flag = conf.getboolean("HAN_Multi", "mini_batch_flag")
            self.fanout = conf.getint("HAN_Multi", "fanout")
            self.use_uva = conf.getboolean("HAN_Multi", "use_uva")
            self.multi_label = conf.getboolean("HAN_Multi", "multi_label")
            self.threshold = conf.getfloat("HAN_Multi", "threshold")

        elif self.model_name == "RGAT_Multi":
            self.lr = conf.getfloat("RGAT_Multi", "learning_rate")
            self.weight_decay = conf.getfloat("RGAT_Multi", "weight_decay")
            self.seed = conf.getint("RGAT_Multi", "seed")
            self.dropout = conf.getfloat("RGAT_Multi", "dropout")
            self.batch_size = conf.getint("RGAT_Multi", "batch_size")
            self.hidden_dim = conf.getint("RGAT_Multi", "hidden_dim")
            self.in_dim = conf.getint("RGAT_Multi", "in_dim")
            self.out_dim = conf.getint("RGAT_Multi", "out_dim")
            self.num_layers = conf.getint("RGAT_Multi", "num_layers")
            self.num_heads = conf.getint("RGAT_Multi", "num_heads")
            self.patience = conf.getint("RGAT_Multi", "patience")
            self.max_epoch = conf.getint("RGAT_Multi", "max_epoch")
            self.num_workers = conf.getint("RGAT_Multi", "num_workers")
            self.mini_batch_flag = conf.getboolean("RGAT_Multi", "mini_batch_flag")
            self.fanout = conf.getint("RGAT_Multi", "fanout")
            self.use_uva = conf.getboolean("RGAT_Multi", "use_uva")
            self.multi_label = conf.getboolean("RGAT_Multi", "multi_label")
            self.threshold = conf.getfloat("RGAT_Multi", "threshold")
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        if hasattr(self, "device"):
            self.device = th.device(self.device)
        elif gpu == -1:
            self.device = th.device("cpu")
        elif gpu >= 0:
            if not th.cuda.is_available():
                self.device = th.device("cpu")
                warnings.warn(
                    "cuda is unavailable, the program will use cpu instead. please set 'gpu' to -1."
                )
            else:
                self.device = th.device("cuda", int(gpu))

        if getattr(self, "use_uva", None):
            self.use_uva = False
            warnings.warn(
                "'use_uva' is only available when using cuda. please set 'use_uva' to False."
            )

    def __repr__(self):
        return "[Config Info]\tModel: {},\tTask: {},\tDataset: {}".format(
            self.model_name, self.task, self.dataset
        )
