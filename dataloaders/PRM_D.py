from torch.utils.data import Dataset
import json

class PRM_DDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_json = []
        with open(data_path, "r") as f:
            for line in f:
                if line.strip():
                    self.data_json.append(json.loads(line.strip()))
    
    def __len__(self):
        return len(self.data_json)
    
    def __getitem__(self, idx):
        item = self.data_json[idx]
        problem = item.get("problem", "")
        steps = item.get("steps", [])
        corrects = item.get("is_correct", False)
        answers = item.get("answer", "")
        subjects = item.get("subject", "")
        levels = item.get("level", 1)
        unique_ids = item.get("unique_id", "")
        ground_truth_answers = item.get("ground_truth_answer", "")
        return problem, steps, answers, corrects, subjects, levels, unique_ids, ground_truth_answers
