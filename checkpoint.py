import torch 
from modeling_mpt import MPTForCausalLM
from adapt_tokenizer import AutoTokenizerForMOD
from peft import PeftModel
import json
import gc
import os


def save(model, checkpoint_chain, model_id):
    # Model is a PEFT model so to do a normal save of the LoRA weights you just save_pretrained() it
    # Parse the model_id to make sure it doesn't have illegal characters and replace it if so 
    # Checkpoint chain should be the checkpoint chain that you was return from the load method when you initially loaded the model before training
    model_id = _preprocess_model_id(model_id)
    print("Saving at : " + model_id)
    model.save_pretrained(model_id)
    try:
        #add checkpoint chain to adapter_config.json
        with open(model_id + '/adapter_config.json') as json_data:
            data = json.load(json_data)
        data["checkpoint_chain"] = checkpoint_chain + "$" + model_id
        with open(model_id + '/adapter_config.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except(Exception):
        raise Exception("An Error occured while saving checkpoint chain. You must go in and manually add this to the adapter.json file in the format checkpoint_chain : $path1$path2$etc")

def load( base_model, most_recent_checkpoint, load_in_8bit = True, hfModelClass = MPTForCausalLM):
    
    data = ""
    with open(most_recent_checkpoint + '/adapter_config.json') as json_data:
        data = json.load(json_data)
        
    #checkpoint chain is made into an array of the different checkpoint names
    try:
        checkpoint_chain_str = data["checkpoint_chain"]
        checkpoint_chain = checkpoint_chain_str.split("$")[1:]
    except(KeyError):
        print("This checkpoint does not have a checkpoint_chain entry in its adapter_config.json file. Please add it with the value '$path1$path2'")
        checkpoint_chain_str = most_recent_checkpoint
        checkpoint_chain = [checkpoint_chain_str]

    i = len(checkpoint_chain) - 1

    model = MPTForCausalLM.from_pretrained(
        base_model,
        #load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map={'': 0},
    )

    while i >= 0:
        model = PeftModel.from_pretrained(
            model,
            checkpoint_chain[i],
            torch_dtype=torch.float16,
            device_map={'': 0}
        )
        model = model.merge_and_unload()
        i -= 1
    
    config = model.config
    state_dict = model.state_dict()
    try:
        model = hfModelClass.from_pretrained(None, config=config, state_dict=state_dict, torch_dtype=torch.float16, load_in_8bit = load_in_8bit, device_map={'': 0} )
    except(AttributeError):
        raise AttributeError(
            "hfModelClass must be a subclass of huggingface's pretrained model and/or implement the from_pretrained method in the same way"
        )

    config = None
    state_dict = None
    gc.collect()
    torch.cuda.empty_cache()

    return model, checkpoint_chain_str
    
def _preprocess_model_id(model_id):
    if type(model_id) != str:
        raise TypeError("model_id must be of type string")
    #Check for disallowed characters
    if "$" in model_id or "/" in model_id:
        print("Removing all occurences of '$' and '/' in model_id. These characters are not allowed")
        model_id = model_id.replace("$","")
        model_id = model_id.replace("/","")
    #Check if directory already exists
    while os.path.isdir(model_id):
        last_char = model_id[-1]
         #if last character is already an int then add one to it and try again. Else, just concat a 1 to it and try again. 
        if last_char.isdigit():
            new_last_char = str(int(last_char) + 1)
            model_id = model_id[:-1] + new_last_char
        else:
            model_id = model_id + "1"
    
    return model_id