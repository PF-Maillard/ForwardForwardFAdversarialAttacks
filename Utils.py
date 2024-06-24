import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.utils.data import DataLoader
import contextlib
from torchvision.datasets import CIFAR10

def MNIST_loaders(train_batch_size=60000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def CIFAR10_loaders(train_batch_size=50000, test_batch_size=10000):
    transform = Compose([
        ToTensor(),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        CIFAR10('./data/', train=True,
                download=True,
                transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        CIFAR10('./data/', train=False,
                download=True,
                transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = 1.0 #x.max()
    return x_

def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    
def Train(model, train_loader, device, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        
def FFTrain(model, train_loader, epochs):
    x_train, y_train = next(iter(train_loader))
    x_train, y_train = x_train.cuda(), y_train.cuda()
    for i in range(epochs):
        print("Epoch ", i)
        x_pos = overlay_y_on_x(x_train, y_train)
        rnd = torch.randperm(y_train.size(0))
        x_neg = overlay_y_on_x(x_train, y_train[rnd])
        model.train(x_pos, x_neg)
        
def Evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100}%')
    return accuracy

def FFEvaluate(model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predicted = model.predict(images)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Accuracy NET: {accuracy * 100}%')
    return accuracy   

def generate_adversarial_examples(model, test_loader, attack):
    model.eval()
    
    adversarial_images = []
    labels = []
    for images, labels_batch in test_loader:
        adv_images = attack(images, labels_batch)
        adversarial_images.append(adv_images)
        labels.append(labels_batch)
    adversarial_images = torch.cat(adversarial_images)
    labels = torch.cat(labels)
    return adversarial_images, labels

def CW_attack(model, X, y, target, Iterations, learningRate, c):
    ListSavedX = []
    for Index in range(len(X)):
        x = X[Index].unsqueeze(0).clone()
        x = x.requires_grad_(True)
        MyOptimizer = Adam([x], lr=learningRate)
        
        SavedX = x
        PastLoss = 1000 
        for i in range(Iterations):
    
            Pred = model(x)[0]
            _, Indexs = torch.topk(Pred, k=2)

            Current = Indexs[0]
            if target != -1:
                if Current == target:
                    Current = Indexs[1]       
                PredLoss = torch.max(Pred[Current] - Pred[target],torch.tensor(-0.0001))
            else:
                if Current == y[Index]:
                    Current = Indexs[1]   
                PredLoss = torch.max(Pred[y[Index]] - Pred[Current], torch.tensor(-0.0001))

            Loss = PredLoss*c + L2Distance(x, X[Index])
            
            if Loss < PastLoss and PredLoss < 0:
                SavedX = x.clone()
            
            MyOptimizer.zero_grad()
            Loss.backward()
            MyOptimizer.step()
            
            with torch.no_grad():
                x.clamp_(0,1)
        ListSavedX.append(SavedX.detach()[0])
        
    return torch.stack(ListSavedX)

def IterativeCW(model, X, y, target, Iterations, learningRate):
    c = 1000
    AdversarialImages = CW_attack(model, X, y, target, 0, learningRate, c)
    SavedVectors = torch.zeros_like(AdversarialImages)
    while c >= 0.01:
        print("CW Adversarial with c = ", c)
        AdversarialImages  = CW_attack(model, X, y, -1, Iterations, learningRate, c)
        AdversarialImages = torch.clamp(AdversarialImages, 0, 1)
        AdversarialLabels = model(AdversarialImages)
        Predictions = torch.argmax(AdversarialLabels, dim=1)
        mask = (y.cpu() != Predictions.cpu())
        SavedVectors[mask] = AdversarialImages[mask]     
        c/=2
    return SavedVectors


def DisplayResult(ClassicalModel, NewNet, WitnessModel, device, x_te, y_te, X_advFF, X_advPGD, X_advCW, X_advDeep):
    
    print("RESULTS: Adversarial and transferability")
    print()

    AvL2 = GetAvergeL2(x_te, X_advFF)
    print("FF average L2 distance: ", AvL2.item())
    AvL2 = GetAvergeL2(x_te, X_advPGD)
    print("PGD average L2 distance: ", AvL2.item())
    AvL2 = GetAvergeL2(x_te, X_advCW)
    print("CW average L2 distance: ", AvL2.item())
    AvL2 = GetAvergeL2(x_te, X_advDeep)
    print("DeepFool average L2 distance: ", AvL2.item())
    print()
    
    BasedDataloader = DataLoader(list(zip(x_te, y_te)), batch_size=64, shuffle=False)
    
    Evaluate(WitnessModel, BasedDataloader, device)
    MyDataLoader = SetDataLoader (WitnessModel, X_advFF, x_te, y_te, device)
    Evaluate(WitnessModel, MyDataLoader, device)  
    MyDataLoader = SetDataLoader (WitnessModel, X_advPGD, x_te, y_te, device)
    Evaluate(WitnessModel, MyDataLoader, device)
    MyDataLoader = SetDataLoader (WitnessModel, X_advCW, x_te, y_te, device)
    Evaluate(WitnessModel, MyDataLoader, device)
    MyDataLoader = SetDataLoader (WitnessModel, X_advDeep, x_te, y_te, device)
    Evaluate(WitnessModel, MyDataLoader, device)
    print()

    Evaluate(ClassicalModel, BasedDataloader, device)
    MyDataLoader = SetDataLoader (ClassicalModel, X_advFF, x_te, y_te, device)
    Evaluate(ClassicalModel, MyDataLoader, device)  
    MyDataLoader = SetDataLoader (ClassicalModel, X_advPGD, x_te, y_te, device)
    Evaluate(ClassicalModel, MyDataLoader, device)
    MyDataLoader = SetDataLoader (ClassicalModel, X_advCW, x_te, y_te, device)
    Evaluate(ClassicalModel, MyDataLoader, device)
    MyDataLoader = SetDataLoader (ClassicalModel, X_advDeep, x_te, y_te, device)
    Evaluate(ClassicalModel, MyDataLoader, device)
    print()
    
    FFEvaluate(NewNet, BasedDataloader, device)
    MyDataLoader = SetDataLoaderFF (NewNet, X_advFF, x_te, y_te, device)
    FFEvaluate(NewNet, MyDataLoader, device)
    MyDataLoader = SetDataLoaderFF (NewNet, X_advPGD, x_te, y_te, device)
    FFEvaluate(NewNet, MyDataLoader, device)
    MyDataLoader = SetDataLoaderFF (NewNet, X_advCW, x_te, y_te, device)
    FFEvaluate(NewNet, MyDataLoader, device)
    MyDataLoader = SetDataLoaderFF (NewNet, X_advDeep, x_te, y_te, device)
    FFEvaluate(NewNet, MyDataLoader, device)
    print()
    
def WriteResults(FileName, ClassicalModel, NewNet, WitnessModel, device, x_te, y_te, X_advFF, X_advPGD, X_advCW, X_advDeep):
    with open(FileName, 'w') as f:
        with contextlib.redirect_stdout(f):
            DisplayResult(ClassicalModel, NewNet, WitnessModel, device, x_te, y_te, X_advFF, X_advPGD, X_advCW, X_advDeep)
            
def L2Distance(tensor1, tensor2):
    dist = (tensor1 - tensor2)**2
    dist=dist.sum()
    return dist

def AverageL2Distance(ListImages1, ListImages2):
    AvL2=0
    for i in range(len(ListImages1)):
        AvL2 += L2Distance(ListImages1[i], ListImages2[i])
    AvL2/= len(ListImages1)
    return AvL2

def GetAvergeL2(X, AdvX):  
    mask = ~(X == AdvX).all(dim=1)
    X = X[mask]
    AdvX = AdvX[mask]
    L2 = AverageL2Distance(X, AdvX)
    return L2

def SetDataLoaderFF (Model, Xadv, X, y, device):
    Pred = Model.predict(X).to(device)
    mask = Pred == y
    y = y[mask]
    Xadv = Xadv[mask]
    MyDataLoader = DataLoader(list(zip(Xadv, y)), batch_size=64, shuffle=False)
    return MyDataLoader

def SetDataLoader (Model, Xadv, X, y, device):
    Pred = torch.argmax(Model(X).to(device), dim=1)
    mask = Pred == y
    y = y[mask]
    Xadv = Xadv[mask]
    MyDataLoader = DataLoader(list(zip(Xadv, y)), batch_size=64, shuffle=False)
    return MyDataLoader