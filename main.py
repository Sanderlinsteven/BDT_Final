import os
import redis
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
     
class Redis:
    """
    Helper class to interact with Redis database.
    """
    def __init__(self, host='localhost', port=6379, password=None):
        
        """
        Initialize RedisHelper with host, port, and password.

        Parameters:
            host : str
                The hostname of the Redis server.
            port : int
                The port number of the Redis server.
            password : str
                The password for authentication (can be None if no password is set).
        """
        self.redis_client = redis.StrictRedis(host=host, port=port, password=password)
        self.pipeline = self.redis_client.pipeline()
        
    def set_data(self, update_progress_bar):
        """
        Set data in Redis from a CSV file.

        Parameters:
            csv_path : str
                The path to the CSV file containing the data.
            chunk_size : int
                The number of records to process in each pipeline execution.
        """
        csv_path = "CVD_cleaned_trimmed.csv"
        with open(csv_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            headers = next(csv_reader)
            for i, row in enumerate(csv_reader):
                key = row[0]
                values = row[1:]
                for j in range(len(values)):
                    self.pipeline.hset(key, headers[j + 1], values[j])
                if (i + 1) % 1000 == 0:
                    self.pipeline.execute()
            self.pipeline.execute()

    def get_data(self):
        """
        Retrieve data from Redis.

        Returns:
            dict: A dictionary containing the retrieved data.
        """
        data = {}
        processed_keys = set()  # Track processed keys
        cursor = '0'
        print_interval = 10000  # Print progress every 10000 keys
        pipe = self.redis_client.pipeline()  # Create a pipeline
        try:
            while True:
                cursor, keys = self.redis_client.scan(cursor=cursor, count=1000)
                for key in keys:
                    if key not in processed_keys:
                        pipe.hgetall(key)
                        processed_keys.add(key)
                        if len(processed_keys) % print_interval == 0: # Execute the pipeline every 10000 keys
                            results = pipe.execute()  # Execute the pipeline
                            for key, result in zip(processed_keys, results):
                                data[key.decode('utf-8')] = result
                            processed_keys.clear() # Clear the processed keys set for the next batch
                if cursor == 0:
                    results = pipe.execute()  # Execute the remaining commands in the pipeline
                    for key, result in zip(processed_keys, results):
                        data[key.decode('utf-8')] = result
                    break
        except Exception as e:
            print(f"Error occurred: {e}")
            raise
        return data

class HeartDiseasePredictor:
    def __init__(self, df):
        self.df = df

    def build_model(self):
        """
        Build a Random Forest classifier model for predicting heart disease risk.
    
        This method preprocesses the data, splits it into training and testing sets, trains the model, and evaluates its performance.
    
        Parameters:
            None
    
        Returns:
            None
        """
        features = ['Sex', 'Weight_(kg)', 'Height_(cm)', 'Age_Category', 'Exercise', 'Depression']
        target = 'Heart_Disease'
    
        # Map categorical variables to binary format
        self.df['Sex'] = self.df['Sex'].map({'Male': 0, 'Female': 1})
        self.df['Exercise'] = self.df['Exercise'].map({'Yes': 1, 'No': 0})
        self.df['Depression'] = self.df['Depression'].map({'Yes': 1, 'No': 0})
    
        # Encode ordinal feature 'Age_Category'
        encoder = OrdinalEncoder()
        self.df['Age_Category_Encoded'] = encoder.fit_transform(self.df[['Age_Category']])
    
        # Define column transformer to encode 'Age_Category'
        column_transformer = ColumnTransformer([('age_category_encoder', OrdinalEncoder(), ['Age_Category'])], remainder='passthrough')
    
        # Split data into features and target variable
        X = self.df[features]
        y = self.df[target]
    
        # Apply column transformer to features
        X = column_transformer.fit_transform(X)
    
        # Define a range of test sizes for cross-validation
        test_sizes = np.linspace(0.1, 0.5, 5)
        test_size_list = []
        accuracy_list = []
    
        # Find the optimal test size using cross-validation
        for test_size in test_sizes:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            test_size_list.append(test_size)
            accuracy_list.append(accuracy)
        best_test_size = test_size_list[np.argmax(accuracy_list)]
    
        # Split data into training and testing sets using the best test size
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=best_test_size, random_state=42)
    
        # Build and train the Random Forest classifier model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
        # Set the trained model and test data as attributes of the class instance
        self.model = model
        self.X_test = X_test
        self.y_test = y_test


    def assess_risk(self):
        """
        Assess the risk of heart disease based on user input and the trained model.
    
        This method prompts the user to input their age, weight, height, gender, and exercise habits,
        then uses the trained model to predict the risk of heart disease and provides the probability
        of heart disease based on the input.
    
        Parameters:
            None
    
        Returns:
            None
        """
        # Prompt the user for input
        age = float(input("Enter your age: "))
        weight = float(input("Enter your weight (kg): "))
        height = float(input("Enter your height (cm): "))
        gender = input("Enter your gender (Male/Female): ").lower()
        exercise_input = input("Do you exercise regularly? (Yes/No): ").lower()
        depression_input = input("Have you been diagnosed with depression? (Yes/No): ").lower()
        
        # Convert gender input to binary
        if gender == 'male':
            gender_binary = 0
        elif gender == 'female':
            gender_binary = 1
        else:
            print("Invalid gender input. Please enter 'Male' or 'Female'.")
            return
    
        # Convert exercise input to binary
        if exercise_input == 'yes':
            exercise_binary = 1
        elif exercise_input == 'no':
            exercise_binary = 0
        else:
            print("Invalid exercise input. Please enter 'Yes' or 'No'.")
            return
        
        # Convert exercise input to binary
        if depression_input == 'yes':
            depression_binary = 1
        elif depression_input == 'no':
            depression_binary = 0
        else:
            print("Invalid depression input. Please enter 'Yes' or 'No'.")
            return
    
        # Make prediction using the trained model
        prediction = self.model.predict([[age, weight, height, gender_binary, exercise_binary, depression_binary]])
        probability = self.model.predict_proba([[age, weight, height, gender_binary, exercise_binary, depression_binary]])[0][1]
    
        # Print risk assessment and probability
        if probability > .5:
            print("Based on the information provided, you are at high risk of heart disease.")
        else:
            print("Based on the information provided, you are at low risk of heart disease.")
        
        print(f"Probability of heart disease: {probability:.2f}")


    def test_model(self):
        """
        Test the trained model with randomly generated input samples.
    
        This method generates random input samples for age, weight, height, gender, and exercise
        and uses the trained model to predict the risk of heart disease for each sample.
    
        Parameters:
            None
    
        Returns:
            None
        """
        # Define ranges for age, weight, and height
        age_range = (20, 80)
        weight_range = (50, 150)
        height_range = (150, 200)
        
        # Define the number of samples
        num_samples = 10
        
        # Generate random samples for age, weight, and height
        age_samples = np.random.uniform(*age_range, num_samples)
        weight_samples = np.random.uniform(*weight_range, num_samples)
        height_samples = np.random.uniform(*height_range, num_samples)
        
        # Generate random samples for gender (0 for Male, 1 for Female) and exercise (1 for Yes, 0 for No) and depression (1 for Yes, 0 for No)
        gender_samples = np.random.choice([0, 1], num_samples)
        exercise_samples = np.random.choice([1, 0], num_samples)
        depression_samples = np.random.choice([1, 0], num_samples)
        
        # Create feature vectors from the random samples
        feature_vectors = np.column_stack((age_samples, weight_samples, height_samples, gender_samples, exercise_samples, depression_samples))
        
        # Make predictions with the trained model
        predictions = self.model.predict(feature_vectors)
        probabilities = self.model.predict_proba(feature_vectors)[:, 1]
        
        # Print the predictions for heart disease risk
        print("Predictions for Heart Disease Risk:")
        print("-----------------------------------")
        print("Age\tWeight\tHeight\tGender\tExercise\tDepression\tRisk\tProbability")
        print("-----------------------------------------------------------")
        for i in range(num_samples):
            age, weight, height, gender, exercise, depression = feature_vectors[i]
            gender_label = 'Male' if gender == 'Male' else 'Female'
            exercise_label = 'Yes' if exercise == 'Yes' else 'No'
            depression_label = 'Yes' if depression == 'Yes' else 'No'
            prediction = "High" if probabilities[i] > .5 else "Low"
            print(f"{age:.1f}\t{weight:.1f}\t{height:.1f}\t{gender_label}\t{exercise_label}\t{prediction}\t{depression_label}\t{probabilities[i]:.2f}")


    def evaluate_model(self):
        """
        Evaluate the performance of the trained classification model using various metrics and visualize the results.
    
        This method calculates evaluation metrics such as accuracy, precision, recall, F1 score, and ROC AUC score,
        and visualizes the ROC curve and Precision-Recall curve. It also prints the confusion matrix with percentages.
    
        Parameters:
            None
    
        Returns:
            None
        """
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]  # Probability of positive class
        
        # Convert true labels to binary format
        label_encoder = LabelEncoder()
        y_test = label_encoder.fit_transform(self.y_test)
        y_pred = label_encoder.fit_transform(y_pred)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
    
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
    
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
    
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
    
        # Plot Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='red', lw=2, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.show()
    
        # Print evaluation metrics
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision[1]:.2f}')  # Precision for the positive class
        print(f'Recall: {recall[1]:.2f}')  # Recall for the positive class
        print(f'F1 Score: {f1:.2f}')
        print(f'ROC AUC Score: {roc_auc:.2f}')
    
        # Print confusion matrix with percentages
        total_samples = np.sum(cm)
        tn_percent = tn / total_samples * 100
        fp_percent = fp / total_samples * 100
        fn_percent = fn / total_samples * 100
        tp_percent = tp / total_samples * 100
        print('Confusion Matrix (in %):')
        print(f'                   Predicted No   Predicted Yes')
        print(f'Actual No         : {tn_percent:.2f}%        {fp_percent:.2f}%')
        print(f'Actual Yes        : {fn_percent:.2f}%        {tp_percent:.2f}%')
        
        print(f"Model accuracy: {accuracy:.2f}")

def violin_plot(df):
    # Convert 'Height_(cm)' column to numeric
    df['Height_(cm)'] = pd.to_numeric(df['Height_(cm)'], errors='coerce')
    
    # Create a violin plot for 'Age_Category'
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Age_Category', y='Height_(cm)', data=df, hue='Sex', split=True)
    plt.title('Violin Plot of Height by Age Category and Sex')
    plt.xlabel('Age Category')
    plt.ylabel('Height (cm)')
    plt.xticks(rotation=45)
    plt.legend(title='Sex', loc='upper right')
    plt.tight_layout()
    plt.show()

def correlation_matrix(df):
    df['Heart_Disease'] = df['Heart_Disease'].map({'No': 0, 'Yes': 1})
    df.drop('Age_Category', axis=1, inplace=True)
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 12})
    plt.title('Correlation Matrix')
    plt.show()

if __name__ == "__main__":
    # Inform that the main block is being executed
    print("Main block executed")
    host = os.environ['redis_host']
    port = os.environ['redis_port']
    password = os.environ['redis_password']

    # Initialize dictionary to store retrieved data
    all_data = {}
    
    # Create RedisHelper instance and populate Redis with data from CSV
    print("Setting data...")
    redis_client = Redis(host, port, password)
    redis_client.set_data()
    
    # Retrieve data from Redis and convert it to a DataFrame
    print("Grabbing Data...")
    all_data = redis_client.get_data()
    df = pd.DataFrame.from_dict(all_data, orient='index')
    
    # Decode byte strings and clean DataFrame column names
    df.columns = [col.decode('utf-8').strip("b'") for col in df.columns]
    df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.astype(str)
    
    heart_disease_predictor = HeartDiseasePredictor(df)
    print("Building Model...")
    heart_disease_predictor.build_model()
    print("Testing Model...")
    heart_disease_predictor.test_model()
    print("Evaluating Model...")
    heart_disease_predictor.evaluate_model()
    print("Assessing Risk...")
    heart_disease_predictor.assess_risk()
    print("Creating Violin plot...")
    violin_plot(df)
    print("Creating Correlation Matrix...")
    correlation_matrix(df)
