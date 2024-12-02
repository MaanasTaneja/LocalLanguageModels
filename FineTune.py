from LanguageModel import LanguageModel
import datasets

class FineTuner:
    def __init__(self, dataset : dataset, model : LanguageModel, tokenizer):
        self.dataset = dataset
        self.model = model.model
        self.tokenizer = tokenizer

        self.dataloader = None

    def load(self):
        # I am expecting data to be
       
        pass
    
    def fit():
        # Implement the fine-tuning logic here
        pass

    def evaluate():
        # Implement the evaluation logic here
        pass

    def save_model(self, path):
        # Implement the logic to save the model
        # save it locally and then reload into language model.
        pass

    def return_model(self):
        return self.model
    