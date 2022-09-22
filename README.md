CHATBOT


An easy to use AI-based package that feeds custom data into a neural network which is trained and outputs a fully functional chatbot.

INSTALLATION


You can install it from Pypi: pip install ChatbotVerse https://pypi.org/project/ChatbotVerse

STEPS TO USE THE CHATBOT:-


#Importing the module
from ChatbotVerse import chatbotVerse as cbv

#Initialize trainer

trainer = cbv.modelTrain()

intents = trainer.loadIntents('intents.json')

words, classes = trainer.preprocess_save_Data(intents) 

train_x, train_y = trainer.prepareTrainingData(words, classes) 

#Create the model

model = trainer.createModel(train_x, train_y, save_path='cbv_model.model')

#Initialize predictor:-
predictor = cbv.modelPredict('intents.json', 'cbv_model.model')

#Get output from the bot

running = True

while running:

    msg = input('You: ')
    
    if msg == 'quit':
    
        running = False
        
    else:
    
        response = predictor.chatbot_response(msg)
        
        print('Bot: ', response)
