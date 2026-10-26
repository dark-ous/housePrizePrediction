import numpy as np
import pandas as pd

df = pd.read_csv("dataset/Housing.csv")


#mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea
binary_colums= ["mainroad",
                "guestroom",
                "basement",
                "hotwaterheating",
                "airconditioning",
                "prefarea"]


for col in binary_colums:
    df[col]=df[col].map({"yes":1,"no":0})


#we cannot use 0,1,2 as tags to represent furnishedStatus as logistic regression will endup learning that furnished is 2x of semi-furnised therefore we ehave to make sure we represent them in one-hot encoding 

#what are approch will be is we will implement 2 new columns that will help us represent the status in 0 and 1 only 

df["is_furnished"]=(df["furnishingstatus"]=="furnished").astype(int)
df["is_semi_furnished"]=(df["furnishingstatus"]=="semi-furnished").astype(int)

feature_names = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "parking",
    "prefarea",
    "is_furnished",
    "is_semi_furnished"]


#splitting the data into features(input) and prices(output)
X = df[feature_names].to_numpy(dtype=float)

house_prices=df["price"].to_numpy(dtype=float)

#getting a random seed so we can use it later for generting random data 
np.random.seed(42)

#we shuffle the whole dataset so there is not linear/sorted data system as that can affect out training 
shuffled_indices = np.random.permutation(len(X))

X = X[shuffled_indices]

house_prices = house_prices[shuffled_indices]


#we will split thiss so we can later on use a small part of the dataset to test/evaluate our model
split_index = int(0.8 * len(X))

X_train = X[:split_index]
X_test= X[split_index:]

prices_train=house_prices[:split_index]
prices_test=house_prices[split_index:]

print("Training houses:", len(X_train))
print("Testing houses:", len(X_test))


feature_means=X_train.mean(axis=0)
feature_stds=X_train.std(axis=0)

feature_stds[feature_stds==0]=1

X_train=(X_train - feature_means) / feature_stds
X_test=(X_test - feature_means) / feature_stds

weights = np.zeros(X_train.shape[1])

bias = 0.0

learning_rate = 0.05
epochs = 5000

#actual training process here we run out dataset through out linear regression algo 5000 times which helps the model to find patters ( there is always a sweetspot of epoches having more does not make it better the model might get overfitted ( memoizes the prices instead of understanding that makes out model predict wrongly for new data or test data which it has never seen ))
for epoch in range(epochs):
    
    predicted_prices = X_train @ weights + bias
    
    #thiss is the line of code that tells lets us know how wrong was our model
    errors = predicted_prices - prices_train
    
    #calculates the mean of the square of errors (the squaring is to make sure that -ve err do not cancel the postive err)
    #we take the mean to find out what is the avg error per house so it does not just try to fix each house prices predi also penalizes the large erros like 1000 or 1201 this are some heavy errors can happen due to some edge case usally 
    #instead we try to find the avg loss simontanesoly or whatever idk how to spell it 
    loss = np.mean(errors ** 2)
    
    # andd this one tells the model how to fix the loss or error
    weights_gradients = ( 2 * (X_train.T @ errors)/len(prices_train))
    
    #same here 
    bias_gradients = 2 * np.mean(errors)
    
    #update the weights and bias ( part where the model fixes its mistake)
    weights -= learning_rate*weights_gradients
    bias -=learning_rate*bias_gradients
    
    if epoch % 500 == 0:
        print(f"Epoch: {epoch:4d} | MSE: {loss:.2f}")
        
        
#feeding the new test input to eval
def predict_price(house_features):

    predicted_price = house_features @ weights + bias

    return predicted_price

test_predictions = predict_price(X_test)

new_house = np.array([[
    6000,  
    3,     
    2,  
    2,    
    1,     
    0,  
    1,     
    0,  
    1,     
    1,    
    1,     
    0,     
    1      
]], dtype=float)

# apply the same scaling used for training.
standardized_new_house = (
    new_house - feature_means
) / feature_stds

predicted_price = predict_price(standardized_new_house)[0]

print(f"Predicted price: {predicted_price:,.2f}")
