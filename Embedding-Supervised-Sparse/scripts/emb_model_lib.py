import json
import os
import torch
import timm
import numpy as np
import math
import torch.nn as nn

from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision.models.resnet import resnet50, resnet18

import scripts.data_loading as prep

from scripts.wideresnet import WideResNet
from scripts.utils import get_train_dir, printProgressBar
from scripts.metrics import get_confusion_matrix


class EmbeddingModel:
    """Class representing the embedding model

    :param args: Training arguments for the embedding model
    :param wkdir: Working directory
    :param writer: Tensorboard writer

    :ivar global_step: Global setp
    :ivar args: Training arguments
    :ivar writer: Tensorboard writer
    :ivar expert_pred: e; 用于(x, e)-->g-->y
    :ivar device: Active device
    :ivar train_dir: Training directory of the embedding model
    :ivar model: Embedding model
    :ivar optimizer: Optimizer
    :ivar scheduler: Learning rate scheduler
    :ivar loss_function: Loss function
    :ivar train_data: Train dataset
    :ivar test_data: Test dataset
    :ivar val_data: Validation dataset
    :ivar train_loader: Train dataloader
    :ivar test_loader: Test dataloader
    :ivar val_loader: Validation dataloader
    """
    def __init__(self, args, wkdir, writer, expert_pred=False):
        self.global_step = 0
        self.args = args
        self.writer = writer
        self.expert_pred = expert_pred
        self.device = prep.get_device()
        self.train_dir = get_train_dir(wkdir, args, 'emb_net')
        self.model = self.get_model()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args['lr'], weight_decay=5e-4, momentum=0.9,
                                         nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [60, 120, 160], gamma=0.2)
        self.loss_function = nn.CrossEntropyLoss()
        self.train_data, self.test_data, self.val_data = prep.get_train_val_test_data(expert=None,
                                                                                      model=args['model'],
                                                                                      dataset=args['dataset'])
        self.train_loader, self.test_loader, self.val_loader, device = prep.get_data_loader(self.train_data,
                                                                                            self.test_data,
                                                                                            self.val_data,
                                                                                            batch_size=args['batch'])
        self.save_model_args()

    def get_model(self):
        """Initialize model

        :return: model
        """
        # load model
        # if self.args['model'] == 'wideresnet':
        #     model = WideResNet(28, 10, 0.2, self.args['num_classes'])
        # elif self.args['model'] == 'resnet18':
        #     model = Resnet(self.args['num_classes'])
        # else:
        #     model = timm.create_model(self.args['model'], pretrained=True, num_classes=self.args['num_classes'])
        model = Resnet(self.args['num_classes'])
        print('Loaded Model', self.args['model'])
        # load model to device
        model = prep.to_device(model, self.device)

        return model

    def train_one_epoch(self, epoch):
        """Train one epoch

        :param epoch: Epoch
        :return: loss
        """
        self.model.train()
        for ii, (data, target, gt, index) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            gt = gt.to(self.device)
            target = target.long()
            if self.expert_pred:
                _, pred = self.model(data, target)
                loss = self.loss_function(pred, gt) # 和 y 比较
            else:
                _, pred = self.model(data)
                loss = self.loss_function(pred, target) # 和 e 比较

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += len(index)
            printProgressBar(ii + 1, math.ceil((len(self.train_data.targets) / self.args['batch'])),
                             prefix='Train Epoch ' + str(epoch + 1) + ':',
                             suffix='Complete', length=40)
            self.writer.add_scalar('Loss/total', loss, self.global_step)
            self.writer.add_scalar('LR/lr', self.optimizer.param_groups[0]["lr"], self.global_step)
        return loss

    def get_embeddings(self, x):
        self.model.eval()
        with torch.no_grad():
            embs, _ = self.model(x)
        return embs

    def get_validation_accuracy(self, epoch, return_acc=False, print_acc=True):
        """Get validation accuracy

        :param epoch: Epoch
        :param return_acc: Boolean flag for returning the accuracy
        :param print_acc: Boolean flag for printing the accuracy
        :return: (Accuracy) optional
        """
        predict = []
        targets = []
        self.model.eval()
        for i, (data, target, gt, indices) in enumerate(self.val_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            gt = gt.to(self.device)
            ex = None
            if self.expert_pred != None:
                ex = self.get_embeddings(data)
            # get model artificial_expert_labels
            with torch.no_grad():
                _, output = self.model(data, ex)

            # get predicted classes from model output
            predicted_class = torch.argmax(output, dim=1).cpu().numpy()

            for p in predicted_class:
                predict.append(p)
            if self.expert_pred == None:
                for t in target:
                    targets.append(t.item())
            else:
                for t in gt:
                    targets.append(t.item())

        # calculate accuracy score
        acc = accuracy_score(targets, predict)
        if print_acc:
            if epoch is None:
                print('Val-Accuracy:', acc)
            else:
                print('Epoch:', epoch + 1, '- Val-Accuracy:', acc)
        self.writer.add_scalar('Acc/valid', acc, self.global_step)
        if return_acc: return acc

    def get_test_accuracy(self, return_acc=False, print_acc=True):
        """Get test accuracy

        :param return_acc: Boolean flag for returning the accuracy
        :param print_acc: Boolean flag for printing the accuracy
        :return: (Accuracy) optional
        """
        predict = []
        targets = []
        self.model.eval()
        for i, (data, target, gt, _) in enumerate(self.test_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            gt = gt.to(self.device)
            with torch.no_grad():
                if self.expert_pred:
                    _, output = self.model(data, target)
                else:
                    _, output = self.model(data)

            # get predicted classes from model output
            m = nn.Softmax(dim=1)
            predicted_class = torch.argmax(output, dim=1).cpu().numpy()
            for p in predicted_class:
                predict.append(p)
            if self.expert_pred:
                for t in gt:
                    targets.append(t.item())
            else:
                for t in target:
                    targets.append(t.item())  
   
        # calculate accuracy score
        acc = accuracy_score(targets, predict)

        cm_true = get_confusion_matrix(targets, predict)
        cat_acc = cm_true.diagonal() / cm_true.sum(axis=1)
        if print_acc: print('Test-Accuracy:', acc, '\nTest-Acc-Class', cat_acc)
        if return_acc: return acc

    def predict_test_data(self):
        """Predict test data

        :return: artificial_expert_labels
        """
        predict = []
        self.model.eval()
        for i, (data, target, indices) in enumerate(self.test_loader):
            data.to(self.device)
            target.to(self.device)
            with torch.no_grad():
                output = self.model(data)

            m = nn.Softmax(dim=1)
            predicted_class = torch.argmax(m(output), dim=1).cpu().numpy()
            for p in predicted_class:
                predict.append(int(p))

        return predict

    def load_from_checkpoint(self, mode='best'):
        """Load from checkpoint

        :param mode: Checkpoint to load (best or latest)
        :return: epoch
        """
        cp_dir = self.train_dir + '/checkpoints/checkpoint.' + mode
        try:
            checkpoint = torch.load(cp_dir)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.global_step = checkpoint['global_step']
            epoch = checkpoint['epoch']
            print('Found latest checkpoint at', cp_dir)
            print('Continuing in epoch', epoch + 1)
        except:
            epoch = 0
            print('No Checkpoint found')
            print('Starting new from epoch', epoch + 1)

        return epoch

    def save_to_checkpoint(self, epoch, loss, acc):
        with open(self.train_dir + '/logs/exp_log.json', 'r') as f:
            log = json.load(f)

        if acc >= np.max(log['valid_acc']):
            torch.save({'epoch': epoch,
                        'global_step': self.global_step,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        'valid_acc': acc}, self.train_dir + '/checkpoints/checkpoint.best')

        torch.save({'epoch': epoch,
                    'global_step': self.global_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    'valid_acc': acc}, self.train_dir + '/checkpoints/checkpoint.latest')

    def save_model_args(self):
        with open(self.train_dir + 'args/model_args.json', 'w') as f:
            json.dump(self.args, f)


class Resnet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.resnet = resnet18(pretrained=True)
        # self.embedding_layer = nn.Embedding(1, 512) # 输入应是离散整数
        # del self.resnet.fc

        try:
            print('load Resnet-18 checkpoint')
            self.load_my_state_dict(
                torch.load(
                    os.getcwd()[:-len('Embedding-Supervised-Sparse')] + "/nih_images/checkpoint.pretrain"),
                strict=False)
        except FileNotFoundError:
            print('load Resnet-18 pretrained on ImageNet')

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def load_my_state_dict(self, state_dict, strict=True):
        pretrained_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        self.resnet.load_state_dict(pretrained_dict, strict=strict)

    def forward(self, x, expert=None):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        if expert == None:
            features = torch.flatten(x, 1)
        else:
            expert_expanded = expert.unsqueeze(1).repeat(1, 512)
            # fuse X_emb and E
            features = torch.flatten(x, 1) + torch.flatten(expert_expanded, 1)
            
        out = self.resnet.fc(features)
        out = nn.Softmax(dim=1)(out)
        return features, out

def main():
    data = (
        torch.tensor((3, 224, 224), dtype=torch.float32, device='cpu'),
        1,
        1,
        '00018921_051.png'
    )
    
    
if __name__ == '__main__':
    main()
    