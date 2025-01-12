import torch
import torchvision.models as models
import torchvision.transforms as transforms



class MobileNet(torch.nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x


def process(batch_size, data):
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    normalize = transforms.Normalize(mean, std)
    inv_normalize = transforms.Normalize(
    mean= [-m/s for m, s in zip(mean, std)],
    std= [1/s for s in std]
    )   
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    normalize])
    dataset = transform(data)
    print("entrée réseau : ",dataset.shape)
    #dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    print(dataset.unsqueeze(0).shape)
    return dataset

if __name__=='__main__':

    x = torch.zeros(100, 3, 224,224)
    net = MobileNet()
    y = net(x)
    assert y.shape == (100, 576)
    