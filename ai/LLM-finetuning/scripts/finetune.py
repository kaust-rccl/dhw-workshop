import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling
from transformers import TrainerCallback
import torch
import transformers

def main():

    model_name="openai-community/gpt2"  

    # Load dataset from the Hugging Face datasets library
    dataset = load_dataset("pierre-pessarossi/climate-question-answers")

    # Split dataset in train, val, test
    dataset=dataset.rename_column("instruction", "question")
    test_dataset=dataset['test']
    train_dataset=dataset['train'].train_test_split(test_size=0.2,keep_in_memory=False)
    val_dataset=train_dataset['test']
    train_dataset.pop('test')  # drop the test column 
    train_dataset=train_dataset['train'] # Converting the type from DatasetDict to Datasets

    climate_dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
    })


    # Tokenize the texts
    def preprocess_function(examples):
    
        # To catch any anomaly in the curation of dataset i.e. bad datatype either in question or an answer, 
        # we set the string to a NULL token.
        batch_size=len(examples['question'])
        for i in range(batch_size):
            if examples['question'][i] == None:
                examples['question'][i] = "[NULL]"
            if examples['answer'][i] == None:
                examples['answer'][i] = "[NULL]"

        inputs = [q + " [SEP] " + a for q, a in zip(examples["question"], examples["answer"])]
        
        model_inputs = tokenizer(inputs, 
                                 max_length=200, 
                                 truncation=True, 
                                 padding=True, 
                                 return_tensors="pt")
    
    # The "labels" are the tokenized outputs:
        return model_inputs


    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                trust_remote_code=True)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token


    # Apply "preprocess_function" to all the examples in iterator object "climate_dataset_dict". 
    # "batched=True" implies passing examples in a batch instead of one at a time
    tokenized_dataset = climate_dataset_dict.map(
    preprocess_function, 
    batched=True,batch_size=4,drop_last_batch=True)

    # Load the data collator (the batch maker given a dataset)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal Lang Model uses a causal (not masked) language model, similar to GPT-2
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(os.environ['BASEDIR'],'results'),
        overwrite_output_dir=True,
        eval_strategy="epoch",
        learning_rate=3e-4,
        logging_steps=10,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=1000,
        save_total_limit=2,
        deepspeed=os.path.join('ds_configs',os.environ['DS_CONFIG']),  # Path to DeepSpeed config file        
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':True},
        report_to='tensorboard',
        fp16=True,
        logging_dir=os.path.join(os.environ['BASEDIR'],"logs"),
        logging_strategy='steps',
        log_level='info',
    )
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )


    # Start the training
    trainer.train()


    # Save the final model and tokenizer
    FINAL_MODEL=os.path.join(os.environ['BASEDIR'],'final_model')
    model.save_pretrained(FINAL_MODEL)
    tokenizer.save_pretrained(FINAL_MODEL)


if __name__ == "__main__":
    main()
