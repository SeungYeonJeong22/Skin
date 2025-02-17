from efficientnet_pytorch import EfficientNet

def enet(model_name='efficientb0', num_classes=23):
    if model_name == 'efficientb0':
        return EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    elif model_name == 'efficientb1':
        return EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
    elif model_name == 'efficientb2':
        return EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
    elif model_name == 'efficientb3':
        return EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
    elif model_name == 'efficientb4':
        return EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
    elif model_name == 'efficientb5':
        return EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)
    elif model_name == 'efficientb6':
        return EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)
    elif model_name == 'efficientb7':
        return EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)
    else:
        raise ValueError('Invalid model name')