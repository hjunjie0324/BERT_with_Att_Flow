import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections, time, sys, logging
#from layers.bert_plus_bidaf import BERT_plus_BiDAF
from layers.two_bert_plus_bidaf import Two_BERT_plus_BiDAF
#from utils import data_processing
from utils import data_processing_for_two_bert
from torch.utils.data import DataLoader

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self,encodings):
        self.encodings = encodings
    def __getitem__(self,idx):
        return {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

def train(device, model, optimizer, dataloader, num_epochs = 3):
    """
    Inputs:
    model: a pytorch model
    dataloader: a pytorch dataloader
    loss_func: a pytorch criterion, e.g. torch.nn.CrossEntropyLoss()
    optimizer: an optimizer: e.g. torch.optim.SGD()
    """
    start = time.time()

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}:'.format(epoch, num_epochs - 1))
        logger.info('-'*10)
        # Each epoch we make a training and a validation phase
        model.train()
            
        # Initialize the loss and binary classification error in each epoch
        running_loss = 0.0
        iteration = 1
        curr_loss = 0.0
        for batch in dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()
            # Send data to GPU
            context_input_ids = batch['context_input_ids'].to(device)
            context_attention_mask = batch['context_attention_mask'].to(device)
            question_input_ids = batch['question_input_ids'].to(device)
            question_attention_mask = batch['question_attention_mask']
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            # Forward computation
            # Get the model outputs
            outputs = model(context_input_ids, context_attention_mask, question_input_ids, question_attention_mask, start_positions, end_positions)
            loss = outputs[0]
            curr_loss += loss.item()
            if iteration % 2000 == 0:
                logger.info('Iteration: {}'.format(iteration))
                logger.info('Loss: {:.4f}'.format(curr_loss/2000))
                curr_loss = 0.0
            iteration += 1
            # In training phase, backprop and optimize
            loss.backward()
            optimizer.step()                   
            # Compute running loss/accuracy
            running_loss += loss.item()

        epoch_loss = running_loss/len(dataloader)
        logger.info('Loss: {:.4f}'.format(epoch_loss))
        if epoch == 1:
            torch.save(model.state_dict(),'checkpoint.pt')
    # Output info after training
    time_elapsed = time.time() - start
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model.state_dict()

def main(learing_rate = 5e-5, batch_size = 4, num_epochs = 3):
    train_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    train_encodings, _ =  data_processing_for_two_bert.data_processing(train_url)

    for key in train_encodings:
        train_encodings[key] = train_encodings[key][0:100]

    train_dataset = SquadDataset(train_encodings)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(device)

    model = Two_BERT_plus_BiDAF(if_extra_modeling=True)
    model.to(device)
    logger.info("Model Structure:"+"\n"+"-"*10)
    logger.info(model)

    parameters = model.parameters()
    logger.info("Parameters to learn:"+"\n"+"-"*10)
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info("\t"+str(name))
    
    logger.info("Hyperparameters:"+"\n"+"-"*10)
    logger.info("Learning Rate: " + str(learing_rate))
    logger.info("Batch Size: "+ str(batch_size))
    logger.info("-"*10)
    logger.info("Number of Epochs: "+ str(num_epochs))

    optimizer = optim.Adam(parameters, lr=learing_rate)
    dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    trained_model = train(device, model, optimizer, dataloader, num_epochs=num_epochs)
    torch.save(trained_model,'two_bert_model.pt')

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler("train_log.log"))
    if len(sys.argv) == 4:
        main(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    else:
        main()