from utils.dataset import *
from torch.utils.data import DataLoader


class DataloaderGetter(object):
    def __init__(self, config, data_type=None):
        if data_type is None:
            self.data_type = ["train", "test"]
        else:
            self.data_type = data_type
        self.datasets = list()
        for t in self.data_type:
            tmp_set = MyDataset(config.base_path, config.code_word2id, config.comment_word2id, config.dataset,
                                config.max_code_len, config.max_comment_len, config.max_keywords_len, t)
            self.datasets.append(tmp_set)
        self.batch_size = config.batch_size
        self.num_workers = 0
        self.pin_memory = False

    def get(self):
        data_loaders = list()
        for i in range(len(self.data_type)):
            shuffle_flag = True if i == 0 else False
            tmp_loader = DataLoader(self.datasets[i], batch_size=self.batch_size, shuffle=shuffle_flag,
                                    collate_fn=self.datasets[i].collate_fn, num_workers=self.num_workers,
                                    pin_memory=self.pin_memory)
            data_loaders.append(tmp_loader)
        return data_loaders
