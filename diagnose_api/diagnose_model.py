import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, num_ftrs, num_medterm, mode='classifier'):
        super(DenseNet121, self).__init__()
        if mode == 'densenet':
            self.densenet121 = torchvision.models.densenet121(pretrained=True)
        # num_ftrs = self.densenet121.classifier.in_features
        # num_ftrs = 512
        elif mode == 'classifier':
            self.densenet121 = torchvision.models.densenet121(pretrained=True)
            self.densenet121.medterm_classifier = nn.Sequential(
                nn.Linear(num_ftrs, num_medterm),
                nn.Sigmoid()
            )

    def forward(self, input, imgs, mode='classifier'):
        if mode == 'densenet':
            features = self.densenet121.features(imgs)  #[1,1024,7,7]
            batch_feats1 = F.relu(features, inplace=True)
            out1 = F.avg_pool2d(batch_feats1, kernel_size=7, stride=1).view(batch_feats1.size(0), -1) #[1,1024]
            return out1
        else:
            medterm_probs = self.densenet121.medterm_classifier(input)
            return medterm_probs


class VGG19(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, out_size):
        super(VGG19, self).__init__()
        self.vgg19 = torchvision.models.vgg19(pretrained=True)

        self.vgg19.features = nn.Sequential(*list(self.vgg19.features.children())[0:35])
        self.vgg19.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, out_size),
            nn.Sigmoid()
        )

    def forward(self, imgs, att_size=7):
        batch_feats = self.vgg19.features(imgs)
        fc_feats = batch_feats.mean(3).mean(2)
        att_feats = batch_feats.permute(0, 2, 3, 1)

        batch_feats = F.relu(batch_feats, inplace=True)
        batch_feats = F.max_pool2d(batch_feats, kernel_size=2, stride=2)
        batch_feats = batch_feats.view(batch_feats.size(0), -1)
        class_probs = self.vgg19.classifier(batch_feats)

        return fc_feats, att_feats, class_probs