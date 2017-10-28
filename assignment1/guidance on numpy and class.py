import numpy as np


class Sam:
    count = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def mini(self):
        return min(self.x, self.y)

    mini2 = mini

    def maxi(self):
        return max(self.x, self.y)

ins = Sam(2, 3)
# print ins.mini()
# print ins.maxi()
# print ins.mini2()


class Bag:
    def __init__(self):
        self.data = []

    def add(self, x):
        self.data.append(x)

    def add_twice(self, x):
        self.add(x)
        self.add(x)

ins2 = Bag()
# print ins2.data
# print ins2.add(2)
# print ins2.data
# print ins2.add_twice(3)
# print ins2.data

a = np.array([1,2,3])
print(a.shape)

b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.shape)

c = np.zeros((2, 3)) # ones, full, eye, random.random
print c

print b[:,2]
print b[:,2:3]

print b[1, :]
print b[1:2, :]

print b[[0,1,0],[1,0,1]]
print np.array([b[0,1], b[1,0], b[0,1]])

print np.arange(3)
# print b[np.arange(2), np.array([0, 1, 2])] # this is the basic idea

a=np.array([[1, 2],[3, 4]])
b=np.array([[5, 6],[7, 8]])
v=np.array([9, 10])
c=np.empty_like(a)

print np.add(a, b) #substract, multiply, divide, dot
print np.sqrt(a)
print a.dot(b) #np.dot(a, b)

print np.sum(a)
print np.sum(a, axis = 0)

print a.T

print np.add(a, v) #add by row
print np.add(a.T, v).T #if you insist add by column, do two transaction
print a+np.reshape(v, (2,1)) #or reshape the vector to be broadcast









