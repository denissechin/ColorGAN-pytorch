import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset


transform_to_input_image = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
])

transform_to_target_image = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor()
])



class Gray_colored_dataset(Dataset):
    
    def __init__(self, path, transform_colored, transform_gray):
        self.gray_dataset = ImageFolder(path, transform=transform_gray)
        self.colored_dataset = ImageFolder(path, transform=transform_colored)
    
    def __len__(self):
        return(len(self.gray_dataset))
        
    def __getitem__(self, idx):
        gray_img = self.gray_dataset[idx][0]
        colored_img = self.colored_dataset[idx][0]
        return (gray_img, colored_img)


