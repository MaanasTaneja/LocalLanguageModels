from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

class LanguageModel:
    def __init__(self, model : str, params_model : dict, params_tokenizer : dict, system_prompt : str):
        self.model_path = model
        self.conversation_history = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.__init_model(model, params_model, params_tokenizer, system_prompt)

        print("inti done")

    def __init_model(self, model_name : str, params_model : dict, params_tokenizer : dict, system_prompt : str):
        if(not (params_tokenizer and params_model)):
            try:
                self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, **params_model)
                self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, **params_tokenizer)
            except Exception as e:
                print(f"Error occured while initializing Model: {e}")
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, device_map = "auto", load_in_4bit = True)
                self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, padding_side = "left")
            except Exception as e:
                print(f"Error occured while initializing Model: {e}")

        self.conversation_history.append({"role" : "system", "content" : system_prompt})
    
    def _generate(self, input_embeddings : dict, generation_params : dict = {}) -> str:
        '''
        After looking at the current input (let’s say a partial sentence), the model generates a probability distribution over the 
        vocabulary for the next token. This means it assigns probabilities to each token in its vocabulary being the next one.
        For example, if the sentence so far is: “The cat sat on the”, the model might give:
	    •	“mat” a probability of 0.6
	    •	“table” a probability of 0.3
	    •	“ground” a probability of 0.1
	    3.	Sampling/Selection of the Next Token:
        Based on this probability distribution, the next token is chosen. There are different ways to do this:
	    •	Greedy decoding: Always pick the token with the highest probability at each step (i.e., the “most likely” next word).
	    •	Beam search: Keep track of multiple high-probability sequences (paths), evaluating several possibilities before committing 
            to a final sequence.
	    •	Sampling: Instead of always choosing the highest-probability token, sample according to the distribution to introduce 
            randomness and variety.

        By default, the model uses Greedy Search Decode for generation, this leads to repetition of tokens, and sometimes, the model might
        choose a token which is highly probable right now, but the tokens after that might not make sense.

        That is why Beam Search Decode is used, it keeps track of multiple high-probability sequences (paths), evaluating several possibilities.
        And choosing the sequence with the highest joint probablity. 

        N-grams are sequences of N words. no_repeat_ngram_size is the size of n-grams that the model should avoid repeating.

        When we sample, we use the probabilty distribuition of p(token, given these previous tokens). 
        And we will have all sorts of words that are probable, but we need to decide which one to choose...

        Greedy? or Beam,  It becomes obvious that language generation using sampling is not deterministic anymore. 
        The word ("car") ("car") is sampled from the conditioned 
        probability distribution P(w∣"The")P(w∣"The"), followed by sampling ("drives"("drives") from P(w∣"The","car") P(w∣"The","car") .
        '''
        if(not generation_params):
            #Default generation parameters; using Beam Search Decode.
            generation_params = {"max_new_tokens" : 512, "num_beams" : 5, "early_stopping" : True, "no_repeat_ngram_size" : 2} 
        
        generated_embeddings = self.model.generate(**input_embeddings, **generation_params)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input_embeddings.input_ids, generated_embeddings)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def prompt(self, prompt : str, generation_config : dict = {}):
        pass

    def get_latest_response(self) -> str:
        return self.conversation_history[-1]["content"]

    def print_conversation_history(self):
        for dialogue in self.conversation_history:
            for k, v in dialogue.items():
                print(f"{k} : {v}")

class InstructionModel(LanguageModel):
    def __init__(self, model : str, params_model : dict, params_tokenizer : dict, system_prompt : str):
        super().__init__(model, params_model, params_tokenizer, system_prompt)
    
    def prompt(self, prompt : str, generation_config : dict = {}):
        text = None
        model_embedding = None
        self.conversation_history.append({"role" : "user", "content" : prompt})
        text = self.tokenizer.apply_chat_template(conversation=self.conversation_history, tokenize=False, add_generation_prompt=False)
        model_embedding = self.tokenizer([text], return_tensors="pt").to(self.device)
            
        response = self._generate(model_embedding, generation_config)
        self.conversation_history.append({"role" : "lm", "content" : response})

class NonInstructionModel(LanguageModel):
    def __init__(self, model : str, params_model : dict, params_tokenizer : dict, system_prompt : str):
        super().__init__(model, params_model, params_tokenizer, system_prompt)
    
    def prompt(self, prompt : str, generation_config : dict = {}):
        model_embedding = None
        self.conversation_history.append({"role" : "user", "content" : prompt})
        model_embedding = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        response = self._generate(model_embedding, generation_config)
        self.conversation_history.append({"role" : "lm", "content" : response})



if __name__ == "__main__":
    model = "qwen 0.5b Instruct"
    gen_model = "gpt2"
    gen_init_prompt = "once upon a time there was a red fox"
    params_model = {
        "device_map" : "cpu",
    }
    params_tokenizer = {}
    system_prompt = "You are a helpful assistant."

    lm_non_instruct = NonInstructionModel(gen_model, params_model, params_tokenizer, gen_init_prompt)
    lm_non_instruct.prompt("once upon a time there was a red fox and he")
    print(lm_non_instruct.get_latest_response())

    lm_instruct = InstructionModel(model, params_model, params_tokenizer, system_prompt)
    lm_instruct.prompt("what are airpods?")
    print(lm_instruct.get_latest_response())

    print("Conversation History for Non Instruction Model:")
    lm_non_instruct.print_conversation_history()

    print("Conversation History for Instruction Model:")
    lm_instruct.print_conversation_history()

    #i had no idea that _ means protected and __ means private in python. thats crazy.


        

