# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:35:42 2024

@author: micha
"""
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from trainLibTorch import *

# select the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# get data
BATCH_SIZE = 16
train_dataloader, class_names, targets = create_dataset(
                                    path=PATH_TRAIN,
                                    batchsize=BATCH_SIZE,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]
                                )
test_dataloader, _, _ = create_dataset(
                            path=PATH_TEST,
                            batchsize=BATCH_SIZE,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]
                                )

# get model
lodaed_model = resnet50(weights=ResNet50_Weights.DEFAULT)
lodaed_model.fc = nn.Linear(2048, len(class_names))

# load model
model_path = "restnet_test0.pth"
lodaed_model.load_state_dict(torch.load(f=model_path))

visualize_preds(model=lodaed_model,
                dataloader=test_dataloader,
                class_names=class_names,
                batchsize=BATCH_SIZE)

preds, targets, accuracy = make_predictions_dataloader(lodaed_model, 
                                             train_dataloader,
                                             device,
                                             class_names)
print("Train subset accurafe = ", accuracy)

plot_decision_matrix(class_names=class_names,
                      y_pred_tensor=torch.tensor(preds),
                      targets=torch.tensor(targets))


preds, targets, accuracy  = make_predictions_dataloader(lodaed_model, 
                                             test_dataloader, 
                                             device,
                                             class_names)
print("Test subset accurafe = ", accuracy)
plot_decision_matrix(class_names=class_names,
                      y_pred_tensor=torch.tensor(preds),
                      targets=torch.tensor(targets))

plt.show()