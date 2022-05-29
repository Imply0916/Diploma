import test as bt

order = 7
X = bt.butcher(order, 15)
A, B, C = X.radau() 
# Ainv = X.inv(A)        
# T, TI = X.Tmat(Ainv)  
# P = X.P(C)

print(A)
print(B)
print(C)