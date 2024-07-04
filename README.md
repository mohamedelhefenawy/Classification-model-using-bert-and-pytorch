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
