# PyTorch with BERT
This project demonstrates how to use the BERT model with PyTorch for a classification task. It includes data preprocessing, model training, and evaluation.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#Installation)
- [Data loading](#Data-loading)
- [Model Initialization](#Model-Initialization)
- [Model Training](#Model-Training)
- [Evaluation](#evaluation)
- [Results](#Results) 

## Introduction
This project shows how to implement the BERT model using PyTorch. It covers steps from data loading and preprocessing to model training and evaluation, providing a comprehensive example for text classification tasks.

## Installation
To run this project, you'll need to install the following dependencies:
- `transformers`
- `Scikit-learn`
- `torch`
- `Other standard libraries (numpy, pandas, etc.)`

## Data loading
The CustomDataset class is designed to preprocess and prepare the data for BERT. Here is the implementation and explanation of the class:
```bash
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

class CustomDataset (torch.utils.data.Dataset):
    def __init__ (self , tokenizer, df,max_len ):
        self.tokenizer = tokenizer
        self.df = df
        self.max_len = max_len
        self.title = self.df['content']
        self.targets = self.df[listed].values
    def __len__(self):
        return len(self.title)
    def __getitem__(self,index):
        title = str(self.title[index])
        title = ''.join(title.split())

        inputs = self.tokenizer.encode_plus(title,
                                            None , add_special_tokens = True,
                                            max_length = self.max_len,
                                            truncation = True,
                                            padding = 'max_length',
                                            return_tensors = 'pt',
                                            return_attention_mask = True)
        return {'input_ids':inputs['input_ids'].flatten(),
                'attention_mask':inputs['attention_mask'].flatten(),
                'token_type_ids':inputs['token_type_ids'].flatten(),
                'targets': torch.FloatTensor(self.targets[index])
}
```
### Explanation
- Initialization: The class is initialized with a tokenizer, a DataFrame (df), and the maximum length of tokens (max_len).
- Length: The __len__ method returns the number of samples in the dataset.
- Get Item: The __getitem__ method retrieves an item at the specified index, processes the text (removing spaces), and uses the tokenizer to encode the text. It returns a dictionary containing the input IDs, attention mask, token type IDs, and targets.

## Model Initialization
The BertClass class defines the BERT model used for text classification. Here is the implementation and explanation of the class:
```bash
class BertClass(nn.Module):
    def __init__(self):
        super (BertClass,self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased',return_dict =True)
        self.dropout = nn.Dropout(.3)
        self.linear = nn.Linear(768,6)
    def forward (self,input_ids,attention_mask  ,token_type_ids ):
        output = self.bert_model(input_ids , attention_mask , token_type_ids)
        output_dropout = self.dropout(output.pooler_output)
        output  =self.linear(output_dropout)
        return output
```
### Explanation
- ### Initialization
- The class initializes the BERT model (bert-base-uncased), a dropout layer, and a linear layer for classification.
- self.linear = nn.Linear(768, 6): This line initializes a fully connected linear layer with an input dimension of 768 and an output dimension of 6. The input dimension 768 corresponds to the hidden size of the BERT model's pooled output. The output dimension 6 indicates that the model is designed to classify the input text into one of six different classes. Previously, if the output dimension was 2, the model was configured for binary classification (two classes). Now, it has been modified to perform multi-class classification with six classes.


- ### Forward Method:
- The forward method defines the forward pass of the model. It takes input IDs, attention mask, and token type IDs as inputs.
- The inputs are passed through the BERT model to obtain the pooled output.
- This output is then passed through a dropout layer to prevent overfitting.
- Finally, the dropout output is passed through the linear layer to obtain the final classification scores for the six classes.

## Model Training
The train_model function handles the training and validation process. Here is the implementation and explanation of the function:

```bash
def train_model(n_epochs, training_loader, validation_loader, model,
                optimizer, checkpoint_path, best_model_path):

  # initialize tracker for minimum validation loss
  valid_loss_min = np.Inf

  for epoch in range(1, n_epochs+1):
    train_loss = 0
    valid_loss = 0

    model.train()
    print('############# Epoch {}: Training Start   #############'.format(epoch))
    for batch_idx, data in enumerate(training_loader):
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

    print('############# Epoch {}: Training End     #############'.format(epoch))
    print('############# Epoch {}: Validation Start   #############'.format(epoch))

    model.eval()

    with torch.no_grad():
      for batch_idx, data in enumerate(validation_loader, 0):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

      print('############# Epoch {}: Validation End     #############'.format(epoch))
      train_loss = train_loss/len(training_loader)
      valid_loss = valid_loss/len(validation_loader)

      print('Epoch: {} \tAverage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))

      checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
      }

      save_path(checkpoint, False, checkpoint_path, best_model_path)

      if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
        save_path(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = valid_loss

    print('############# Epoch {}  Done   #############\n'.format(epoch))

  return model
```
### Explanation
 #### Function Arguments:
-n_epochs: The number of training epochs.
-training_loader: DataLoader for the training dataset.
-validation_loader: DataLoader for the validation dataset.
-model: The BERT model to be trained.
-optimizer: The optimizer used for training.
-checkpoint_path: Path to save checkpoints during training.
-best_model_path: Path to save the best model based on validation loss.

#### Training Loop:
- The function initializes the minimum validation loss (valid_loss_min) to infinity.
- For each epoch, it resets the training and validation losses and sets the model to training mode.
- It iterates through the training DataLoader, performs a forward pass, computes the loss, backpropagates, and updates the


## Evaluation
The notebook includes a function to evaluate the trained model. It calculates the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

## Results
After training and evaluation, the model's performance is printed. Example output metrics:
```bash
Accuracy: 0.6307
Precision: 0.8046
Recall: 0.7463
F1 Score: 0.7740
ROC AUC: 0.8363
```

