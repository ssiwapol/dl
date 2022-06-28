import numpy as np

class MSELoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        se     = (A - Y) * (A - Y)
        sse    = np.ones((N, 1)).T @ se @ np.ones((C, 1)) 
        mse    = np.squeeze(sse)/(N*C)
        
        return mse
    
    def backward(self):
    
        dLdA = self.A - self.Y
        
        return dLdA

class CrossEntropyLoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        Ones   = np.ones((C, 1), dtype="f")

        self.softmax = np.exp(A) * (1 / (np.dot(np.exp(A), Ones))) 
        crossentropy = -Y * np.log(self.softmax)
        L = np.sum(crossentropy) / N
        
        return L
    
    def backward(self):
    
        dLdA = self.softmax - self.Y
        
        return dLdA
