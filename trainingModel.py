import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt


#prework on data to scale, separate label and features, separate test and training

fileContent = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

features = fileContent.drop('Diabetes_binary', axis=1)
label = fileContent['Diabetes_binary']

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

joblib.dump(scaler, 'full_feature_scaler.pkl')

features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)


features_train, features_test, label_train, label_test = train_test_split(
    features_scaled_df, label,
    test_size=0.1,         # 20% test data, 80% train data
    random_state=42,       # For reproducibility
    stratify=label           # Ensures class balance in both sets (important for classification)
)

features_val, features_test, label_val, label_test = train_test_split(
    features_test, label_test, test_size=0.5, stratify=label_test, random_state=42
)



#convert pandas df to pytorch tensors then to tensor datasets and then to data loaders


features_train_tensor = torch.tensor(features_train.values, dtype=torch.float32)
label_train_tensor = torch.tensor(label_train.values, dtype=torch.long)
features_test_tensor = torch.tensor(features_test.values, dtype=torch.float32)
label_test_tensor = torch.tensor(label_test.values, dtype=torch.long)

train_dataset = TensorDataset(features_train_tensor, label_train_tensor)
test_dataset = TensorDataset(features_test_tensor, label_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

features_val_tensor = torch.tensor(features_val.values, dtype=torch.float32)
label_val_tensor = torch.tensor(label_val.values, dtype=torch.long)
val_loader = DataLoader(TensorDataset(features_val_tensor, label_val_tensor), batch_size=64, shuffle=False)


#neural network model


class DiabetesModel(nn.Module):
    def __init__(self, input_size):
        super(DiabetesModel, self).__init__()
        
        # First fully connected layer (input to 128 hidden units)
        self.fc1 = nn.Linear(input_size, 128)
        
        # Dropout to reduce overfitting after first hidden layer
        self.dropout1 = nn.Dropout(0.3)

        # Second fully connected layer (128 -> 82)
        self.fc2 = nn.Linear(128, 64)

        # Another dropout layer
        self.dropout2 = nn.Dropout(0.3)

        # third fully connected layer (82 -> 54)
        self.fc3 = nn.Linear(64, 32)

        # Another dropout layer
        self.dropout3 = nn.Dropout(0.3)

        # Final layer outputs raw scores (logits) for 3 classes
        self.output = nn.Linear(32, 2)

    def forward(self, x):
        # Pass input through first layer and apply ReLU
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        # Pass through second layer + ReLU + Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # Pass through third layer + ReLU + Dropout
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)


        # Output layer (no activation — handled by loss function)
        return self.output(x)
    

#creating model
model = DiabetesModel(input_size=features_train_tensor.shape[1])

#loss function
criterion = nn.CrossEntropyLoss()

#optimiser
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)



#train model and eval model

# best_val_accuracy = 0
# patience = 15  # stop if val accuracy doesn't improve after 5 epochs
# epochs_without_improvement = 0
# num_epochs = 50
# val_accuracy_history = []

# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0

#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     # 🔍 Evaluate on validation set
#     model.eval()
#     val_correct = 0
#     val_total = 0

#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             val_correct += (predicted == labels).sum().item()
#             val_total += labels.size(0)

#     val_accuracy = 100 * val_correct / val_total
#     val_accuracy_history.append(val_accuracy)

#     print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Validation Accuracy={val_accuracy:.2f}%")

#     # 🔁 Early Stopping logic
#     if val_accuracy > best_val_accuracy:
#         best_val_accuracy = val_accuracy
#         epochs_without_improvement = 0
#         torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
#     else:
#         epochs_without_improvement += 1
#         if epochs_without_improvement >= patience:
#             print("Early stopping triggered 🚨")
#             break



# #final testing
# model.load_state_dict(torch.load('best_model.pth'))
# model.eval()

# test_correct = 0
# test_total = 0

# with torch.no_grad():
#     for inputs, labels in test_loader:
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         test_correct += (predicted == labels).sum().item()
#         test_total += labels.size(0)

# final_test_accuracy = 100 * test_correct / test_total
# print(f"\n✅ Final Test Accuracy: {final_test_accuracy:.2f}%")


# #ploting
# plt.plot(range(1, len(val_accuracy_history) + 1), val_accuracy_history, marker='o')
# plt.title("Validation Accuracy Over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Validation Accuracy (%)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


