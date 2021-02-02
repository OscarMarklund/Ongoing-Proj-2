def trainandsave(model_architecture, hidden_layer_1_units, hidden_layer_2_units, learning_rate, gpu, n_epochs, save_directory):
    # Receives user input with defaults in case to train a suitable model architecture on images of flowers within the folder 'flowers.'

    # Imports are here
    import numpy as np
    import torch.nn.functional as F

    import argparse
    import json
    import time
    import torch
    from torch import nn, optim
    from torchvision import datasets, transforms, models
    import os
    from collections import OrderedDict
    import PIL
    from PIL import Image

    # enumerate the paths for image folders
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    # Transforms data pretraining into data_sets listed below
    data_sets = ['test', 'valid', 'train']
    data_transforms = {
        data_sets[0]: transforms.Compose([transforms.Resize(255), 
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])]),
        data_sets[1]: transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])]),
        data_sets[2]: transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])}

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in data_sets}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=102, shuffle = True) for x in data_sets}


    # Load in a dictionary to map class integers to class names
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)


    ############# USER DECIDES DEVICE IF AVAILALBE
    if gpu is True:
        if torch.cuda.is_available() is True:
            device = torch.device('cuda')
            print('GPU will have your model doing sit-ups')
        else:
            device = torch.device('cpu')
            print('GPU was not found so CPU is being used')
    else:
        device = torch.device('cpu')
        print('Training will commence on CPU')


    # Model is chosen
    ############### USER INPUT ARCHITECTURE HERE
    vgg11 = 'vgg11'
    vgg13 = 'vgg13'
    vgg16 = 'vgg16'
    vgg19 = 'vgg19'
    print(('You have chosen {} as the model architecture.').format(model_architecture))
    if model_architecture == vgg11:
        model = models.vgg11(pretrained=True)
    elif model_architecture == vgg13:
        model = models.vgg13(pretrained=True)
    elif model_architecture == vgg16:
        model = models.vgg16(pretrained=True)
    elif model_architecture == vgg19:
        model = models.vgg16(pretrained=True)


    # Freeze paramaters (turning off gradients)
    for param in model.parameters():
        param.requires_grad = False


    # Redefine Model to suit our needs
    ############### USER INPUT HIDDEN LAYERS HERE
    classifier = nn.Sequential(OrderedDict([
        ('fcl', nn.Linear(25088, hidden_layer_1_units)),
        ('relu', nn.ReLU()),
        ('Drop', nn.Dropout(0.5)),
        ('fc2', nn.Linear (hidden_layer_1_units, hidden_layer_2_units)),
        ('relu2', nn.ReLU()),
        ('Drop2', nn.Dropout(0.2)),
        ('fc3', nn.Linear (hidden_layer_2_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
        ]))

   
    # Replacing classifier and produce criterion definition
    model.classifier = classifier
    criterion = nn.NLLLoss()


    # Only train the classifier parameters, feature parameters are frozen, 
    ############### USER INPUT LEARNING RATE HERE
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)


    ############### USER INPUT GPU OR NOT HERE
    model.to(device);
    


    # This cell trains the classifier in approx 35 minutes default
    # Training loss, validation loss and validation accuracy are printed while training. A compeltion time is printed at the end
    ############### USER INPUT EPOCHS HERE
    epochs = n_epochs
    steps = 0
    running_loss = 0
    print_every = 10
    since = time.time()
    
    print("Training your model..." "\n" "If using the GPU, time is approx 4 minutes per epoch")
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1

            # Moving input and tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():

                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        val_loss += batch_loss.item()

                        #calculate actual probabilities (accuracy)
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Val loss: {val_loss/len(dataloaders['valid']):.3f}.. "
                          f"Val accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                    running_loss = 0
                    model.train() # sets back to training mode

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    # Tests the training model in around 5 minutes against the dataloaders['test'] images
    model.eval()
    model.to(device)
    accuracy = 0

    print("Now testing trained model against unseen testing images, won't be too long...")
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

            
    # confirming classifier numbers with the model set
    model.class_to_idx = image_datasets['train'].class_to_idx
    #checking to make sure three important values actually exist
    print(model.class_to_idx)
    print(model.state_dict())
    print(model.classifier)
    #new_class_to_idx = model.class_to_idx
    #new_state_dict = model.state_dict()
    #new_classifier = model.classifier
    
    ############# SAVES TRAINED AND TESTED MODEL INTO DIRECTORY OF USER'S CHOICE WHERE /OPT/ EXISTS IN TORCH.SAVE
    ############# EPOCH INPUT RE-RECORDED HERE DUE TO ERROR ATTEMPTING TO RECALL IT FROM TRAINING PARAMETERS FIX THIS
    
    def save_checkpoint():
        checkpoint = {
            'input_size': 150528,
            'output_size': 102,
            'epochs': n_epochs,
            'model': 'models.{}(pretrained=True)'.format(model_architecture),
            'classifier': model.classifier,
            'class_to_idx': model.class_to_idx,
            'batch_size': 102,
            'lr': learning_rate,
            'state_dict': model.state_dict()}

        torch.save(checkpoint, save_directory)
        print ('model checkpoint saved in ' + save_directory)
    save_checkpoint()