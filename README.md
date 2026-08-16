# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
## STEP 1:

Import required libraries such as PyTorch, NumPy, Matplotlib and Scikit-Learn.

## STEP 2:

Load and preprocess the FashionMNIST dataset using normalization and tensor transformation.

## STEP 3:

Create DataLoader objects for training and testing datasets.

## STEP 4:

Design the CNN architecture using convolution, pooling and fully connected layers.

## STEP 5:

Train the CNN model using Cross Entropy Loss and Adam optimizer.

## STEP 6:

Evaluate the model using accuracy, confusion matrix, classification report and predict new images.



## PROGRAM

### Name:Yashwanth asv

### Register Number: 212224230309

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
trl = DataLoader(train_set, batch_size=64, shuffle=True)
tstl = DataLoader(test_set, batch_size=64, shuffle=False)

print("Train Samples:", len(train_set))
print("Test Samples:", len(test_set))
class CNNclassifier1(nn.Module):

    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.l1 = nn.Linear(128 * 3 * 3, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 10)

    def forward(self, x):

        x = self.pool(torch.relu(self.c1(x)))
        x = self.pool(torch.relu(self.c2(x)))
        x = self.pool(torch.relu(self.c3(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))

        x = self.l3(x)

        return x
class CNNclassifier1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.c2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.c3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.l1=nn.Linear(128*3*3,64)
        self.l2=nn.Linear(64,32)
        self.l3=nn.Linear(32,10)
    def forward(self,x):
        x=self.pool(torch.relu(self.c1(x)))
        x=self.pool(torch.relu(self.c2(x)))
        x=self.pool(torch.relu(self.c3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.l1(x))
        x=torch.relu(self.l2(x))
        x=self.l3(x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNclassifier1().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
summary(model,input_size=(1,28,28))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

epochs = 3
epochs = 3

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in trl:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss/len(trl):.4f}"
    )
model.eval()

correct = 0
total = 0

actual = []
predicted_list = []

with torch.no_grad():

    for images, labels in tstl:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        actual.extend(labels.cpu().numpy())
        predicted_list.extend(predicted.cpu().numpy())

accuracy = 100 * correct / total

print(f"\nAccuracy = {accuracy:.2f}%")

print("\nClassification Report\n")

print(
    classification_report(
        actual,
        predicted_list,
        target_names=test_set.classes
    )
)
cm = confusion_matrix(actual, predicted_list)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=test_set.classes,
    yticklabels=test_set.classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
with torch.no_grad():

    img, label = test_set[0]

    output = model(
        img.unsqueeze(0).to(device)
    )

    _, pred = torch.max(output, 1)

    display_img = img * 0.5 + 0.5

    plt.imshow(display_img.squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()

    print("Actual :", test_set.classes[label])
    print("Predicted :", test_set.classes[pred.item()])

```

### OUTPUT

## Training Loss per Epoch
<img width="969" height="112" alt="image" src="https://github.com/user-attachments/assets/ea7bd510-b3ac-4d6c-af08-9609fcab2bd0" />


## Confusion Matrix
<img width="747" height="665" alt="image" src="https://github.com/user-attachments/assets/b28f4e14-de07-4bfc-962a-b476c420e4b0" />

## Classification Report
<img width="692" height="335" alt="image" src="https://github.com/user-attachments/assets/ff0b3a20-482b-4a55-9f43-8ec24496cb83" />



### New Sample Data Prediction
<img width="369" height="400" alt="image" src="https://github.com/user-attachments/assets/a2121e55-da17-4228-8990-57de5eec71b2" />


## RESULT
Thus, a Convolutional Neural Network (CNN) was successfully developed and trained using the FashionMNIST dataset for image classification.
