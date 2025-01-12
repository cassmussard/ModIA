import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import pandas as pd
from script_model.image_inference.mobileNet_model import MobileNet
from annoy import AnnoyIndex

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ImageAndPathsDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        img, _= super(ImageAndPathsDataset, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, path
    

def train(net):
    model = net.eval()

def test(model, dataloader):
    features_list = []
    paths_list = []
    for x, paths in tqdm(dataloader):
        with torch.no_grad():
            embeddings = model(x.cuda())
            features_list.extend(embeddings.cpu().numpy())
            paths_list.extend(paths)

            df = pd.DataFrame({
                'features': features_list,
                'path': paths_list
            })
    return df

def create_annoy_vec(features_list):
    dimension = 576 
    n_trees = 10
    index = AnnoyIndex(dimension, 'angular')
    for i, vecteur in enumerate(features_list):
        index.add_item(i, vecteur)
    index.build(n_trees)
    index.save('annoy_db_img.ann')

if __name__=='__main__':
    # Create annoy index and Path for the images if not already done
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
    dataset = ImageAndPathsDataset('MLP-20M', transform)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=False)
    net = MobileNet().to(device)
    train(net)
    features = test(net, dataloader)
    # Annoy index
    create_annoy_vec(features['features'])
    # path.csv
    features['path'].to_csv('path.csv', index=False)
    

