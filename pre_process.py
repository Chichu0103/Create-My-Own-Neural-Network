from PIL import Image
from torchvision import transforms
import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    pil_img = Image.open(image)
    
    edit_img = transforms.Compose([transforms.Resize(226),
                                  transforms.RandomCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224,0.225])])
    
    img_tensor = edit_img(pil_img)
    
    processed = np.array(img_tensor)
    
    processed = processed.transpose((0,2,1))
    
    return processed