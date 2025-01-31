from sklearn.datasets import load_iris
from sklearn.utils._bunch import Bunch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import itertools
from sklearn.neighbors import KNeighborsClassifier
import pickle as pkl

#load the iris dataset
class IrisKNNClassifier:
    
    def __init__(self):
        self.iris_dataset:tuple[Bunch,tuple] = load_iris()
        self.markers: list[str] = ['o', 's', '^']
        self.knn: KNeighborsClassifier = None
        
    
    @property
    def model_trained(self):
        return False if self.knn is None else True
    
        
    def _split_data(self,train_size: float,random_state: int):
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
            self.iris_dataset['data'], 
            self.iris_dataset['target'], 
            random_state=random_state,
            train_size=train_size
        )

    def _train_model(self,n_neighbors: int):
        """
        The fit method is used to train the K-Nearest Neighbors (KNN) model on the given dataset.
        It takes two main parameters: the feature matrix (X) and the target vector (y).

        X: This is the feature matrix, where each row represents a data point and each column represents a feature.
        For example, if you have a dataset of flowers, each row might represent a flower, and the columns might represent
        features such as petal length, petal width, sepal length, and sepal width.

        y: This is the target vector, where each element corresponds to the target value (or class label) of the
        corresponding row in the feature matrix X. For example, in a classification problem, y might contain the species
        of each flower.

        The fit method stores the feature matrix (X) and the target vector (y) in the model's internal state.
        This allows the model to reference the training data when making predictions on new, unseen data.
        """
        
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)   
        self.knn.fit(self.data_train, self.target_train)
        
    def train_and_evaluate(self,train_size: float,random_state: int,n_neighbors: int):
        self._split_data(train_size,random_state)
        self._train_model(n_neighbors)
        return self.knn.score(self.data_test, self.target_test)
    
    def predict(self,features: list[float]) -> tuple[int,str]:
        """ Returns the prediction and the name of the species

        Args:
            features (list[list[float]]): the features to predict, must have 4 values 

        Raises:
            ValueError: Case when features is empty
            ValueError: Case when features does not have 4 values
            ValueError: Case when model is not trained

        Returns:
            tuple[int,str]: the prediction and the name of the species
        """
        if len(features) == 0:
            raise ValueError("Features cannot be empty")
        if len(features[0]) != 4:
            raise ValueError("Features must have 4 values")
        if not self.model_trained:
            raise ValueError("Model not trained")
        
        prediction = self.knn.predict(features)
        
        return prediction, self.iris_dataset['target_names'][prediction]
    
    def save_model(self, path:str):
        """Saves the model to a file

        Args:
            path (str): the path to save the model

        Raises:
            ValueError: Case when model is not trained
        """
        if not self.model_trained:
            raise ValueError("Model not trained")
        with open(path, 'wb') as file:
            pkl.dump(self.knn, file)
    
    def load_model(self, path:str):
        """Loads the model from a file

        Args:
            path (str): the path to load the model
        """
        with open (path, 'rb') as file:
            self.knn = pkl.load(file)
    
    #* Prints
    
    def print_description(self):
        print(self.iris_dataset['DESCR'])
        
    def print_shape_of_data(self):
        print(f"Shape of training data: {self.data_train.shape}")
        print(f"Shape of testing data: {self.data_test.shape}")
        print(f"Shape of training target: {self.target_train.shape}")
        print(f"Shape of testing target: {self.target_test.shape}")
    
    def print_target_names(self):
        print(f"Target names: {self.iris_dataset['target_names']}")
        
    def print_feature_names(self):
        print(f"Feature names:{self.iris_dataset['feature_names']}")
        
    def print_type_of_data(self):
        print(f"Type of data: {type(self.iris_dataset['data'])}")
        
    def print_shape_of_data(self):
        print(f"Shape of data: {self.iris_dataset['data'].shape}")
        

if __name__ == "__main__":
    path = "model.pkl"
    iris = IrisKNNClassifier()
    """"iris.print_description()
    iris.print_shape_of_data()
    iris.print_target_names()
    iris.print_feature_names()
    
    train_size=0.6
    n_neighbors=3
    
    
    score = iris.train_and_evaluate(train_size=train_size,random_state=42,n_neighbors=n_neighbors)
    print(f"Test set score (train size = {(train_size)*100:.1f}%): {score*100:.1f}%")
    
    #Predicts new data example
    new_data = [[5.1, 3.5, 1.4, 0.2]]
    _,species_predicted =iris.predict(new_data)
    
    print(f"Predicted species: {species_predicted}")
    
    #Save model
    iris.save_model(path)"""
    
    #Load model
    iris.load_model(path)
    
    #Predicts new data example
    new_data = [[5.1, 3.5, 1.4, 0.2]]
    _,species_predicted =iris.predict(new_data)
    print(f"Predicted species with loaded model: {species_predicted}")
    
    
    