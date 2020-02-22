# Keeping Track of a Disaster! Assisting Disaster Relief Agencies by Providing Relevant Messages  

In this project, we are in a situation when a natural disaster has just happened. People are communicating on their social meadia and to disaster relief agencies expressing their needs and asking for help. We want to keep tracks of those relevant conversation, identifying people who are in need by classifying the tweets and text messages sent around that location at that time. For example, we would like to detect a tweet like this 

> We are 36 people at the ABC church without clean water. Please help!

If we simply filter text by keywords such as `water`, we may end up with a tweet such as

> Craving water after the basketball match today. Great jobs team.

Therefore, we would like to build a web app to classify the text, giving out categories that the text may belong to so that disaster relief agencies can act promptly on the right people who need help.

In this project folder, the two Jupyter notebooks are for experimenting the steps we will take to build a ETL pipeline to process the data and an ML pipeline to build the classifier. After that, we moves the codes to the python files to run the web app. To run the web app, follow the steps listed below.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/


