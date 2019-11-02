from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess(X_train, y_train):
    return train_test_split(
            X_train, 
            y_train, 
            test_size=0.25, 
            shuffle=False, 
            random_state=0) # reproducable results
