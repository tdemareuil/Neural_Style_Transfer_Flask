from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models


def model():
    # get the "features" portion of VGG19 (we don't need the "classifier" portion)
    vgg = models.vgg19(pretrained=True).features

    # freeze all VGG parameters since we're only optimizing the target image
    for param in vgg.parameters():
        param.requires_grad_(False)
 
    return vgg


def load_image(img_path, max_size=400, shape=None):
    '''Load in and transform the user image'''
    
    image = Image.open(img_path).convert('RGB')
    
    # resize large images so that they don't slow down processing
    if max(image.size) > max_size: # max_size has been set to 400 pixels in the x-y dims
        size = max_size
    else:
        size = max(image.size)
    if shape is not None: # I don't really get this one
        size = shape   
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # discard the transparent alpha channel with [:3] and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0) # unsqueeze adds a dimension of size 1 along the specified axis
    
    return image


def im_convert(tensor):
    '''Unnormalize and convert a tensor image to numpy to be able to display it'''
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze() # squeeze removes axis of size 1
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def get_features(image, model, layers=None):
    '''Run an image forward through a model and get the features for a set of layers.
       Default layers are for VGGNet matching Gatys et al. (2016)'''
   
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  # layer chosen by Gaty et al. for content representation
                  '28': 'conv5_1'}
   
    features = {}
    x = image
    
    
    for name, layer in model._modules.items(): # model._modules is a dictionary holding each module in the model
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features


def gram_matrix(tensor):
    
    _, depth, height, width = tensor.size()
    
    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(depth, height * width)
    
    # compute the gram matrix
    gram = torch.mm(tensor, tensor.t())
    
    return gram


def stylize(content, style, model):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu (cuda) if possible
    
    # load in content, style and model
    content = content.to(device)
    style = style.to(device)
    vgg = model.to(device)
    
    # get content and style features before forming the target image
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # calculate the gram matrices for each layer of our style representation (style_features)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # create a third "target" image (copy of the content image) and iteratively change its style
    target = content.clone().requires_grad_(True).to(device)
    
    # iteration hyperparameters
    style_weights = {'conv1_1': 0.6,
                     'conv2_1': 0.5,
                     'conv3_1': 0.5,
                     'conv4_1': 0.8,
                     'conv5_1': 0.8}
    content_weight = 1  # alpha (for total loss computation)
    style_weight = 1e5  # beta (for total loss computation)
    optimizer = optim.Adam([target], lr=0.003)
    steps = 1000  # decide how many iterations to update your image

    # iterate over the nb of steps
    for i in range(1, steps+1):

        # CONTENT LOSS: get the features from your target image and compute content loss
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        # STYLE LOSS: initialize the style loss to 0 and iterate through each style layer to increment it
        style_loss = 0
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape # for normalization below

            # get the "style" style representation for the layer
            style_gram = style_grams[layer]

            # compute style loss for the layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            
            # normalize and add to total style loss
            style_loss += layer_style_loss / (d * h * w)

        # TOTAL LOSS:
        total_loss = content_weight * content_loss + style_weight * style_loss

        # update the target image to end iteration
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return target
