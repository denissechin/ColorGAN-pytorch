import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset




def read_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def transfer_to_gray(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

class Gray_colored_dataset(Dataset):
    
    def __init__(self, paths, transforms):
        self.paths = paths
        self.transforms = transforms
        
    def __len__(self):
        return(len(self.paths))
        
    def __getitem__(self, idx):
        rgb_img = read_image(self.paths[idx])
        gray_img = transfer_to_gray(rgb_img)
        transformed = self.transforms(image=rgb_img, grayscale_image=gray_img)
        return {
            'rgb_image' : transformed['image'],
            'grayscale_image' : transformed['grayscale_image']
        }
