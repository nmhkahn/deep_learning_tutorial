import os
import urllib.request
import tarfile
import pickle
import numpy as np

data_dir = "data/CIFAR-10/"
tar_path = "data/CIFAR-10/cifar-10-batches-py.tar.gz"
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

train_file = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
test_file = ["test_batch"]

def _maybe_download_and_extract():
    # 데이터 폴더가 있는 경우 받을 필요 없다고 가정
    if os.path.exists(data_dir):
        return
   
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)    
        
    file_path, _ = urllib.request.urlretrieve(url=data_url, filename=tar_path)
    tarfile.open(name=tar_path, mode="r:gz").extractall(data_dir)
    
def _convert_data(raw):
    images = raw["data"]
    labels = raw["labels"]
    
    images = np.array(images, dtype=float) / 255.0
    images = images.reshape([-1, 3, 32, 32])
    # TensorFlow에 맞게 reshape ([배치, height, width, channel])
    images = images.transpose([0, 2, 3, 1])
    
    return images, np.array(labels)


class CIFAR10():
    def __init__(self, flatten=False, shuffle=True, is_train=True):
        # 자동으로 데이터를 다운받도록 (만약 데이터가 준비되어있다면 이러한 작업은 필요없겠죠)
        _maybe_download_and_extract()
        
        if is_train:
            file_name = train_file
            data_size = 50000
        else:
            file_name = test_file
            data_size = 10000
            
        self.shuffle = shuffle
        self.X = np.zeros((data_size, 32, 32, 3))
        self.y = np.zeros((data_size), dtype=np.int64)
        
        # 이미지 및 라벨 불러오기
        for i, name in enumerate(file_name):
            with open(os.path.join(data_dir, "cifar-10-batches-py", name), "rb") as _file:
                raw = pickle.load(_file, encoding="latin1")
                self.X[i*10000:(i+1)*10000], self.y[i*10000:(i+1)*10000] = _convert_data(raw)
                
        # 클래스 이름 불러오기
        with open(os.path.join(data_dir, "cifar-10-batches-py", "batches.meta"), "rb") as _file:
            raw = pickle.load(_file, encoding="latin1")["label_names"]
        self.class_names = np.array(raw)
        
        # MLP인 경우를 위한 flatten
        if flatten:
            self.X = self.X.reshape(data_size, -1)
        
        if self.shuffle:
            self._shuffle()
        self.start, self.end = 0, 0
        self.epoch_done = False

    def next_batch(self, batch_size):
        """이 함수가 중요합니다"""
        self.start = self.end
        self.epoch_done = False
        # 마지막 배치 크기는 배치 사이즈보다 작을 수 있음
        self.end = min(len(self.X), self.start+batch_size)
        
        X_to_return = self.X[self.start:self.end]
        y_to_return = self.y[self.start:self.end]
        
        # 데이터 전체를 다 본 경우(1 epoch) 다시 셔플하기
        if self.end == len(self.X):
            self.epoch_done = True
            self.start, self.end = 0, 0
            if self.shuffle:
                self._shuffle()

        return X_to_return, y_to_return
        
    def _shuffle(self):
        perm = np.arange(len(self.X))
        np.random.shuffle(perm)
        self.X = self.X[perm]
        self.y = self.y[perm]