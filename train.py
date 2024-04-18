import torch
from torch.utils.data import DataLoader, TensorDataset
from model import SimpleCNN  # Make sure this is the updated SimpleCNN class with multi-class output


# Function to load the dataset from .pth files
def load_dataset(file_name):
    data = torch.load(file_name)
    return data['images'], data['labels']


# Load the datasets
train_images, train_labels = load_dataset('train_dataset.pth')
val_images, val_labels = load_dataset('val_dataset.pth')

# Create DataLoaders directly from the loaded data
batch_size = 64
train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_images, val_labels), batch_size=batch_size, shuffle=False)

# Check GPU availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model instance and move it to the device
model = SimpleCNN(num_classes=7).to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the number of epochs and logging interval
num_epochs = 100
log_interval = 10
# 追踪损失和准确度
train_losses = []
train_accuracies = []


def calculate_accuracy(output, target):
    _, predicted = torch.max(output.data, 1)  # 获取最大概率的类别
    if target.dim() > 1:  # 检查target是否为one-hot编码
        target = torch.argmax(target, dim=1)  # 如果是，将其转换为类别索引
    correct = (predicted == target).sum().item()
    total = target.size(0)
    accuracy = correct / total
    return accuracy


# 模型训练
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_losses = []
    epoch_accuracies = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        if target.dim() > 1:  # 检查target是否为one-hot编码
            target = torch.argmax(target, dim=1)  # 如果是，将其转换为类别索引

        # 模型预测
        output = model(data)
        loss = criterion(output, target)  # 确保target是类别索引

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算损失和准确度
        epoch_losses.append(loss.item())
        accuracy = calculate_accuracy(output, target)
        epoch_accuracies.append(accuracy)

        # 打印状态信息
        if batch_idx % log_interval == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Train_Loss: {loss.item()}, Train_Accuracy: {accuracy:.2f}')

    # 计算平均损失和准确度
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
    train_losses.append(avg_loss)
    train_accuracies.append(avg_accuracy)
    print(f'Epoch {epoch} average loss: {avg_loss}, average accuracy: {avg_accuracy:.2f}')

    # 在验证集上评估模型性能
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_losses = []
        val_accuracies = []
        for val_data, val_target in val_loader:
            val_data, val_target = val_data.to(device), val_target.to(device)  # Move data and targets to the device

            # 模型预测
            val_output = model(val_data)  # No need for unsqueeze if the data is already in the correct shape

            # 计算验证集上的损失
            val_loss = criterion(val_output, val_target)
            val_losses.append(val_loss.item())

            # 计算验证集上的准确度或其他性能指标
            val_accuracy = calculate_accuracy(val_output, val_target)
            val_accuracies.append(val_accuracy)

        # 计算平均验证集损失和准确度
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)

        # 记录验证集性能指标
        print(f'Epoch {epoch} Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}')

    import pickle

    # 保存训练和验证数据到文件
    with open('data.pkl', 'wb') as file:
        data = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'Validation_loss': avg_val_loss,
            'Validation_accuracy': avg_val_accuracy
        }
        pickle.dump(data, file)

# 保存模型状态字典到文件
torch.save(model.state_dict(), 'model.pth')
