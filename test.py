import numpy as np
import torch

# Im = np.load('Train_Images.npy', allow_pickle=True)
# Lb = np.load('Train_Labels.npy', allow_pickle=True)
#
# test_images = np.empty((100, 150, 150))
# true_val = np.empty((100, ))
#
# j = 0
# for i in range(20639, 20739):
#     test_images[j, :, :] = Im[i, :, :]
#     true_val[j, ] = Lb[i, ]
#     j += 1


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=15, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(num_features=15)
        self.relu1 = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=3)
        self.conv2 = torch.nn.Conv2d(in_channels=15, out_channels=30, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(in_channels=30, out_channels=45, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(num_features=45)
        self.relu3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(in_channels=45, out_channels=60, kernel_size=3, stride=1, padding=1)
        self.relu4 = torch.nn.ReLU()
        self.flat = torch.nn.Flatten()
        self.fc = torch.nn.Linear(in_features=50 * 50 * 60, out_features=25)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = self.conv4(output)
        output = self.relu4(output)
        output = self.flat(output)
        output = self.fc(output)

        return output


def test_func(X=test_images):
    X_tensor = np.empty((X.shape[0], 1, 150, 150))
    for i in range(X.shape[0]):
        X_tensor[i, :, :, :] = X[i, :, :]

    tensor_xtest = torch.Tensor(X_tensor)

    predicted_labels = np.empty((X.shape[0],))

    model = CNN()

    checkpoint = torch.load('Final_Model.model')
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        outputs = model(tensor_xtest)
        _, predicted_labels = torch.max(outputs.data, 1)

    predicted_labels = predicted_labels.cpu().detach().numpy()

    return predicted_labels


# Insert the images to be tested into test_images
pred_labels = test_func(test_images)

correct_count = 0

# Insert the true labels into true_val
# It should be of shape [test_images.shape[0], ]
for i in range(len(true_val)):
    if pred_labels[i] == true_val[i]:
        correct_count += 1
accuracy = (correct_count / len(true_val))*100

print(accuracy)
