from model.qa_model import MRCQA
from transformers import AutoTokenizer, pipeline
from configs.config import *


def read_example(path = 'example.json'):
    result = []
    import json
    f = open(path)
    data = json.load(f)
    for i in data:
        QA_input = {}
        QA_input['question'] = i['question']
        QA_input['context'] = i['context']
        result.append(QA_input)
    f.close()
    return result

if __name__ == "__main__":
    model_checkpoint = MODEL_CHECKPOINT
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = MRCQA.from_pretrained(model_checkpoint)

    #print(model)

    nlp = pipeline('question-answering', model=model_checkpoint,
                    tokenizer=model_checkpoint)
  
    ls_examples = read_example()
    for i in range(len(ls_examples)):
        QA_input = ls_examples[i]
        res = nlp(QA_input)
        print('question: {} \n'.format(QA_input['question']))
        print('context: {} \n'.format(QA_input['context']))
        print('pipeline: {} \n'.format(res))
       

    while True:
        if len(QA_input['question'].strip()) > 0 and len(QA_input['context'].strip()) > 0:
            res = nlp(QA_input)
            print('context: {}\n'.format(QA_input['context']))
            print('question: {}\n'.format(QA_input['question']))
            print('pipeline: {}\n'.format(res))
            QA_input['question'] = ""
            QA_input['context'] = ""
        
        QA_input['context'] = input('Context: ')
        QA_input['question'] = input('Question: ')
    

