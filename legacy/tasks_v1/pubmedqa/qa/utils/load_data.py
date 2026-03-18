import json

def load_train_texts():
    
    train_data = json.load(open("../data/train_set.json"))

    train_texts = []
    for id, value in train_data.items():
        question = value["QUESTION"]
        contexts = " ".join(value["CONTEXTS"])
        answer = value["final_decision"]
        train_texts.append(f"Question: {question}\nContexts: {contexts}\nAnswer: {answer}")
        
    return train_texts


def load_test_texts():
    
    train_data = json.load(open("../data/train_set.json"))

    train_texts = []
    for id, value in train_data.items():
        question = value["QUESTION"]
        contexts = " ".join(value["CONTEXTS"])
        train_texts.append(f"Question: {question}\nContexts: {contexts}\nAnswer: ")
        
    return train_texts


def load_test_answer_texts():
    
    test_data = json.load(open("../data/test_set.json"))

    test_answers = []
    for id, value in test_data.items():
        test_answers.append(value["final_decision"])
        
    return test_answers

