import pickle
import matplotlib.pyplot as plt

# 加载训练和验证数据
with open('data.pkl', 'rb') as file:
    data = pickle.load(file)
    train_losses = data['train_losses']
    train_accuracies = data['train_accuracies']

# 加载验证数据
with open('data.pkl', 'rb') as file:
    val_data = pickle.load(file)
    val_loss = val_data['Validation_loss']
    val_accuracy = val_data['Validation_accuracy']

# 绘制训练损失和准确度
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()

plt.show()


print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)
