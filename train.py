from transformer import my_transforms
from my_model import create_nn_model


import torch
import numpy as np
from tqdm import tqdm
import argparse
import os


def train_my_model(data_dir, save_dir, my_arch, hidden, learn_rate, epochs, gpu=False):

    print("Model: ",my_arch, "\n",
          "Hidden Layers: ", hidden, "\n",
          "Learn Rate: ", learn_rate, "\n",
          "epochs: ", epochs, "\n")
    
    if gpu == True:
        if torch.cuda.is_available() == True:
            print("GPU in use.")
            device = "cuda"
        else:
            print("GPU not available.")
            gpu = False
            device = "cpu"
    else:
        if torch.cuda.is_available() == True:
            print("GPU available but not selected. Set --gpu flag, to use.")
        print("CPU in use")
        device = "cpu"
    
    train_dir = f"{data_dir}/train"
    valid_dir = f"{data_dir}/valid"
    test_dir = f"{data_dir}/test"


    train_loader, valid_loader, test_loader, train_data = my_transforms(train_dir, valid_dir, test_dir)

    model, criterion, optimizer = create_nn_model(my_arch, hidden, learn_rate, gpu)

    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, total=len(train_loader), desc=f"Training Epoch {epoch+1}/{epochs}")

        for inputs, labels in train_loader_tqdm:
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=running_loss / len(train_loader))

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        running_loss = 0.0
        correct, total = 0, 0
        
        valid_loader_tqdm = tqdm(valid_loader, total=len(valid_loader), desc="Validating")

        with torch.no_grad():
            for inputs, labels in valid_loader_tqdm:
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                valid_loader_tqdm.set_postfix(loss=running_loss / len(valid_loader))

        valid_loss = running_loss / len(valid_loader)
        valid_losses.append(valid_loss)
        accuracy = 100 * correct / total

        print(f'Epoch [{epoch+1}/{epochs}], '
            f'Train Loss: {train_loss:.4f}, '
            f'Valid Loss: {valid_loss:.4f}, '
            f'Accuracy: {accuracy:.2f}%')
        
    #Test Network

    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0


    with torch.no_grad():
        for inputs, labels in test_loader:
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            preds = model(inputs)
            
            loss = criterion(preds, labels)
            
            test_loss+=loss.item()
            
            _, predicted = preds.max(1)
            
            total+=labels.size(0)
            
            correct += (predicted == labels).sum().item()
            
            
    test_loss /= len(test_loader)
    test_accuracy = 100*correct/total

    print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.2f}%")

    #Create checkpoint
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
        'architecture': my_arch,  
        'model_state_dict': model.state_dict(),  
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,  
        'classifier': model.classifier,  
        'hidden': hidden,
        'epochs': epochs,
        'learning_rate': 0.001
    }


    if not (os.path.exists(save_dir)):
        os.makedirs(save_dir)
    
    torch.save(checkpoint, f"{save_dir}/checkpoint-{my_arch}.pth" )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train network and save as checkpoint.")
    parser.add_argument('data_dir', type = str, help = "Specify the path to where the datasets are stored.")
    parser.add_argument('--save_dir', type=str, default="my_checkpoints", help="Specify path to where models are saved.")
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help="Specify which model architecture.")
    parser.add_argument('--hidden', type=int, default=512, help="Specify the hidden layer neurons.")
    parser.add_argument('--learn_rate', type = float, default=0.001, help="Specify the learning rate for the model.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of iterations to train the model.")
    parser.add_argument('--gpu', action='store_true',help="Specify whether to use the gpu or not.")

    args = parser.parse_args()

    print("Arguements Collected")

    train_my_model(args.data_dir, args.save_dir, args.arch, args.hidden, args.learn_rate, args.epochs, args.gpu)
