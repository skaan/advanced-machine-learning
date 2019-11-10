from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def split(X_train, y_train, random_state = 0):
    return train_test_split(
            X_train, 
            y_train, 
            test_size=0.25, 
            shuffle=False,
            random_state=random_state) # reproducable results
