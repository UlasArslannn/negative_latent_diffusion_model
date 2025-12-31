import os 
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

 
 
class CelebDataset(Dataset):
    
    def __init__(self, image_paths, transform = True):
        
        self.base_path = image_paths
        self.transform = Compose([
            Resize((64,64)),
            ToTensor(),
        ]) if transform else None
        self.images = self.load_images()
        
        
    
    
    def load_images(self):
        
        base = self.base_path
        celebritys = os.listdir('./archive (6)/Celebrity Faces Dataset')


        base_dir = base + '/' # Store the base directory path
        all_celeb_fotos_path = []

        all_celeb_fotos_path = [
            os.path.join(base, celeb, file_name)
            for celeb in celebritys                                    # Iterate through each celebrity folder
            for file_name in os.listdir(os.path.join(base, celeb))     # Iterate through each file in that folder
        ]
        
        return all_celeb_fotos_path
    
    def __len__(self):
        return len(self.images)
    
    
    def __getitem__(self, idx):
        
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        im_tensor = self.transform(image)
        
        im_tensor = (im_tensor - 0.5) * 2
        return im_tensor
        
    
        