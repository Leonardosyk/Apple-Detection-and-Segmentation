import torch
from model import SimpleCNN
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

model_path = 'S:\\Mv_py_ass\\model.pth'


def load_dataset(file_name):
    data = torch.load(file_name)
    return data['images'], data['labels']


test_images, test_labels = load_dataset('test_dataset.pth')
test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=1, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN(num_classes=7).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 初始化预测和标签列表
all_preds = []
all_targets = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        if labels.dim() > 1 and labels.size(1) > 1:
            actual = torch.argmax(labels, dim=1)

        # 收集所有的预测和实际标签
        all_preds.extend(predicted.view(-1).cpu().numpy())
        all_targets.extend(actual.view(-1).cpu().numpy())

        # 打印每个样本的预测和实际标签
        for i in range(len(predicted)):
            print(f'Predicted: {predicted[i].item()}, Actual: {actual[i].item()}')

# 计算并打印测试集的整体精确率、召回率和F1分数
precision = precision_score(all_targets, all_preds, average='weighted')
recall = recall_score(all_targets, all_preds, average='weighted')
f1 = f1_score(all_targets, all_preds, average='weighted')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
