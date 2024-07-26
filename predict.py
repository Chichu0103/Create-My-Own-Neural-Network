from load_my_model import load_my_checkpoint
from pre_process import process_image
from my_label_mapping import get_label_map

import torch.nn.functional as F
import torch

import argparse

def my_predict(img_path, check_path, cat_names, topk=5, gpu=False):
    
    print(img_path)
    print(check_path)
    print(cat_names)


    if (gpu == True) and (not(torch.cuda.is_available())):
        print("GPU available. CPU in use.")
        gpu = False
    
    model, _, _  = load_my_checkpoint(check_path, gpu)
    
    if gpu == True:
        model.to("cuda")

    img_tensor = torch.tensor(process_image(img_path))
    img_tensor = img_tensor.unsqueeze(0)

    model.eval()
    
    with torch.no_grad():
        preds = model(img_tensor)
        
        probs = F.softmax(preds, dim=1)
        
        top_p, top_class = probs.topk(topk, dim=1)
    
    top_p = top_p.squeeze().tolist()
    top_class = top_class.squeeze().tolist()
    
    top_class = [i+1 for i in top_class ]

    pred_class = ""
    max_val = -1

    for prob, clas in zip(top_p, top_class):
        if prob>max_val:
            max_val = prob
            pred_class = clas
    
    print("Top Classes: ", top_class, "\n",
          "Top Probabilities: ", top_p, "\n",
          "Predicted Class: ", get_label_map(cat_names)[pred_class])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get prediction of image using model.")
    parser.add_argument('image_path', type=str, help="Specify path to the image.")
    parser.add_argument('checkpoint_path', type=str, help="Specify path to the saved model checkpoint.")
    parser.add_argument('--topk', type=int, default=5, help="Spefify the number of top classes to return.")
    parser.add_argument('--category_names', type=str, default="cat_to_name.json", help="Specify path to category names.")
    parser.add_argument('--gpu', action='store_true', help="Specify whether to use gpu or not.")

    args = parser.parse_args()

    print("Arguements Parsed")

    my_predict(img_path=args.image_path, check_path=args.checkpoint_path, cat_names=args.category_names, topk=args.topk, gpu=args.gpu)
    


