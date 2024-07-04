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
## Explanation
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
## Explanation
- ## Initialization
- The class initializes the BERT model (bert-base-uncased), a dropout layer, and a linear layer for classification.
- self.linear = nn.Linear(768, 6): This line initializes a fully connected linear layer with an input dimension of 768 and an output dimension of 6. The input dimension 768 corresponds to the hidden size of the BERT model's pooled output. The output dimension 6 indicates that the model is designed to classify the input text into one of six different classes. Previously, if the output dimension was 2, the model was configured for binary classification (two classes). Now, it has been modified to perform multi-class classification with six classes.


- ## Forward Method:
- The forward method defines the forward pass of the model. It takes input IDs, attention mask, and token type IDs as inputs.
- The inputs are passed through the BERT model to obtain the pooled output.
- This output is then passed through a dropout layer to prevent overfitting.
- Finally, the dropout output is passed through the linear layer to obtain the final classification scores for the six classes.

