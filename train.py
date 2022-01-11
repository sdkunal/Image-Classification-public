import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

Im = np.load('Train_Images.npy', allow_pickle=True)
Lb = np.load('Train_Labels.npy', allow_pickle=True)

data_size = 23502
train_size = 17000
val_size = 3000
test_size = 3502

Images = np.empty((data_size, 150, 150))
Labels = np.empty((data_size, ))

for i in range(data_size):
    Images[i, :, :] = Im[i, :, :]
    Labels[i, ] = Lb[i, ]

X_train, X_temp, y_train, y_temp = train_test_split(Images, Labels, test_size=.276657, stratify=Labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=.5386, stratify=y_temp)

X_tr = np.empty((train_size, 1, 150, 150))

for i in range(train_size):
    X_tr[i, :, :, :] = X_train[i, :, :]

tensor_xtrain = torch.Tensor(X_tr)
tensor_ytrain = torch.Tensor(y_train)

X_v = np.empty((val_size, 1, 150, 150))

for i in range(val_size):
    X_v[i, :, :, :] = X_val[i, :, :]

tensor_xval = torch.Tensor(X_v)
tensor_yval = torch.Tensor(y_val)

X_tee = np.empty((test_size, 1, 150, 150))

for i in range(test_size):
    X_tee[i, :, :, :] = X_test[i, :, :]

tensor_xtest = torch.Tensor(X_tee)
tensor_ytest = torch.Tensor(y_test)

train_dataset = TensorDataset(tensor_xtrain, tensor_ytrain)
train_dataloader = DataLoader(train_dataset, batch_size=320, shuffle=True)

val_dataset = TensorDataset(tensor_xval, tensor_yval)
val_dataloader = DataLoader(val_dataset, batch_size=40, shuffle=True)

test_dataset = TensorDataset(tensor_xtest, tensor_ytest)
test_dataloader = DataLoader(test_dataset, batch_size=40, shuffle=True)


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layer_1 = torch.nn.Conv2d(in_channels=1, out_channels=15, kernel_size=3, padding=1)

        self.batch_norm_1 = torch.nn.BatchNorm2d(num_features=15)

        self.relu_1 = torch.nn.ReLU()

        self.pool_layer_1 = torch.nn.MaxPool2d(kernel_size=3)

        self.conv_layer_2 = torch.nn.Conv2d(in_channels=15, out_channels=30, kernel_size=3, padding=1)

        self.relu_2 = torch.nn.ReLU()

        self.conv_layer_3 = torch.nn.Conv2d(in_channels=30, out_channels=45, kernel_size=3, padding=1)

        self.batch_norm_2 = torch.nn.BatchNorm2d(num_features=45)

        self.relu_3 = torch.nn.ReLU()

        self.conv_layer_4 = torch.nn.Conv2d(in_channels=45, out_channels=60, kernel_size=3, padding=1)

        self.relu_4 = torch.nn.ReLU()

        self.flatten = torch.nn.Flatten()

        self.fully_connected = torch.nn.Linear(in_features=50 * 50 * 60, out_features=25)

    def forward(self, input):
        conv_layer_1_output = self.conv_layer_1(input)

        batch_norm_1_output = self.batch_norm_1(conv_layer_1_output)

        relu_1_output = self.relu_1(batch_norm_1_output)

        pool_layer_1_output = self.pool_layer_1(relu_1_output)

        conv_layer_2_output = self.conv_layer_2(pool_layer_1_output)

        relu_2_output = self.relu_2(conv_layer_2_output)

        conv_layer_3_output = self.conv_layer_3(relu_2_output)

        batch_norm_2_output = self.batch_norm_2(conv_layer_3_output)

        relu_3_output = self.relu_3(batch_norm_2_output)

        conv_layer_4_output = self.conv_layer_4(relu_3_output)

        relu_4_output = self.relu_4(conv_layer_4_output)

        flatten_output = self.flatten(relu_4_output)

        fully_connected_output = self.fully_connected(flatten_output)

        return fully_connected_output


model = CNN()

optimizer = torch.optim.NAdam(model.parameters(), lr=0.001, weight_decay=0.0001)

loss_function = torch.nn.CrossEntropyLoss()

num_epochs = 10
best_accuracy = 0

for epoch in range(num_epochs):
    model.train()

    train_accuracy = 0
    train_loss = 0

    for images, labels in iter(train_dataloader):
        optimizer.zero_grad()
        outputs = model(images)

        labels = labels.type(torch.LongTensor)
        loss = loss_function(outputs, labels)
        loss.backward()

        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)
        train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_size
    train_loss = train_loss / train_size

    model.eval()

    val_accuracy = 0

    with torch.no_grad():
        for images, labels in iter(val_dataloader):
            outputs = model(images)

            _, prediction = torch.max(outputs.data, 1)

            val_accuracy += int(torch.sum(prediction == labels.data))

        val_accuracy = val_accuracy / val_size

    print('Epoch number: ' + str(epoch))
    print('Train Accuracy: ' + str(train_accuracy * 100.0) + '%')
    print('Validation Accuracy: ' + str(val_accuracy * 100.0) + '%')

    if val_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_network.model')
        best_accuracy = val_accuracy

test_accuracy = 0

with torch.no_grad():
    for images,labels in iter(test_dataloader):
        outputs = model(images)

        _,prediction = torch.max(outputs.data, 1)

        test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy = test_accuracy/test_size

print('Test accuracy: ' + str(test_accuracy * 100.0) + '%')
