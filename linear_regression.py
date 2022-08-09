class mylinearregression():
    def __init__(self):
        self.beta=np.random.randn(2,1)
        
    def predict(self, X):
        return np.dot(X,self.beta) #X @ self.beta
        
    def error(self, X, Y):
        n=Y.shape[0]
        Yhat=self.predict(X)
        return (np.dot((Y-Yhat).T,(Y-Yhat))/n).item()
        
    def gradients(self, X, Y):
        derivative=np.dot(self.beta.T, np.dot(X.T,X))-np.dot(Y.T,X)
        return derivative.T
    
    def train(self, X, Y, num_epochs=50, lr=0.001):
        history=[]
        for _ in range(num_epochs):
            mse=self.error(X,Y)
            self.beta=self.beta-lr*self.gradients(X,Y)
            print('Iteration: {:03}, Error:{:.4f}'.format(_,mse))
            history.append(mse)
            
        plt.figure()
        plt.plot(history)
        plt.grid()
            
    def plot(self, X, Y):
        Yhat=self.predict(X)
        plt.figure()
        x=X[:,1]
        plt.scatter(x,Y)
        plt.plot(x,Yhat)
  
