from torch.utils.data import Dataset
import json

class PRM_PDataset(Dataset):
    def __init__(self,data_path):
        super().__init__()
        with open(data_path,"r") as f:
            self.data_json = json.load(f)
    
    def __len__(self):
        return len(self.data_json)
    
    def __getitem__(self,idx):
        return self.data_json[idx].strip()