import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision


def print_image(image):
    plt.figure(figsize=(5, 5))
    plt.imshow(np.transpose(torchvision.utils.make_grid(image.cpu(), padding=2, normalize=True).cpu(),(1,2,0)));
    plt.axis('off');
    plt.show();    


def print_train_image(model, image, true_image):
    model.eval()
    test1 = image.to(device).view(1, 1, image.shape[1], image.shape[2])
    with torch.no_grad():
        pred = model(test1)
    MAE_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss()
    print('MAE:', float(MAE_loss(pred, true_image.to(device).view(1, 3, true_image.shape[1], true_image.shape[2]))))
    print('RMSE:', float(torch.sqrt(MSE_loss(pred, true_image.to(device).view(1, 3, true_image.shape[1], true_image.shape[2])))))
    
    print('Predicted Image')
    print_image(pred)
    
    print('Input Image')
    print_image(test1)
    
    print('Original Image')
    print_image(true_image)


def print_test_image(model, image):
    model.eval()
    test_img = image.to(device).view(1, 1, test_img.shape[1], test_img.shape[2])
    with torch.no_grad():
        pred = model(test_img)

    print('Result Image')
    print_image(pred)
    print('Input Image')
    print_image(test_img)


