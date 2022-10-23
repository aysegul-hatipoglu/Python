# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:23:26 2018
@author: user
"""

# importing
import numpy as np

# numpy basics
array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])  # 1*15 vector

print(array.shape) #kaç boyutlu olduğu ile ilgili bilgi verir. (15,) - 15e 1 lik demek

a = array.reshape(3,5)
print("shape: ",a.shape)
print("dimension: ", a.ndim)

print("data type: ",a.dtype.name)
print("size: ",a.size)

print("type: ",type(a))

array1 = np.array([[1,2,3,4],[5,6,7,8],[9,8,7,5]])

zeros = np.zeros((3,4)) #yer ayırmak için kullanılır. 
                        #Append ile eklemek performans olarak daha kötüdür.
                        #Önce 0lar ile ayırıp sonra güncellemek daha doğrudur

zeros[0,0] = 5 #0lardan oluşan bir dizi
print(zeros)

np.ones((3,4))   # 1 lerden oluşan bir dizi

np.empty((2,3))  # boş bir dizi

a = np.arange(10,50,5) #belli aralık ve belli değerlerden oluşan bir dizi
print(a)

a = np.linspace(10,50,20)
print(a)



# %% numpy basic operations

a = np.array([1,2,3])
b = np.array([4,5,6])

print(a+b)
print(a-b)
print(a**2)

print(np.sin(a))

print(a<2)


a = np.array([[1,2,3],[4,5,6]])
b = np.array([[1,2,3],[4,5,6]])

# element wise prodcut
print(a*b)

# matrix prodcut: matris çarpımı
a.dot(b.T) #B'nin transpozu

print(np.exp(a))

a = np.random.random((5,5))# 0 ile 1 arasında 5'e 5lik random sayılar oluşturur.

print(a.sum())
print(a.max())
print(a.min())


print(a.sum(axis=0))#sütunları toplar
print(a.sum(axis=1))#satırları toplar

print(np.sqrt(a))
print(np.square(a)) # a**2


print(np.add(a,a))


# %% indexing and slicing
import numpy as np
array = np.array([1,2,3,4,5,6,7])   #  vector dimension = 1

print(array[0])

print(array[0:4])

reverse_array = array[::-1] #array'i ters çevirir
print(reverse_array)


array1 = np.array([[1,2,3,4,5],[6,7,8,9,10]])

print(array1[1,1])

print(array1[:,1])#satırlardan hepsini, sütunlardan ise sadece 1inciyi al


print(array1[1,1:4])


print(array1[-1,:])
print(array1[:,-1])

# %%
# shape manipulation
array = np.array([[1,2,3],[4,5,6],[7,8,9]])

# flatten
a = array.ravel() #arrayi vector haline getirir. array([1, 2, 3, 4, 5, 6, 7, 8, 9])

array2 = a.reshape(3,3) #resize arrayi değiştiriyor. reshape arrayi değiştirmiyor.

arrayT = array2.T

print(arrayT.shape)


array5 = np.array([[1,2],[3,4],[4,5]])


#array5 = np.column_stack((array1,array1))


# %% stacking arrays

array1 = np.array([[1,2],[3,4]])
array2 = np.array([[-1,-2],[-3,-4]])

# veritical
#array([[1, 2],
#       [3, 4]])
#array([[-1, -2],
#       [-3, -4]])
array3 = np.vstack((array1,array2)) #vertical dikey birleştirme

# horizontal
#array([[1, 2],[-1, -2],
#       [3, 4]],[-3, -4]]

array4 = np.hstack((array1,array2))  #horizontal yatay birleştirme

#%% convert and copy

liste = [1,2,3,4]   # list

array = np.array(liste) #np.array

liste2 = list(array)

a = np.array([1,2,3])

b = a
b[0] = 5
c = a
#burada hafıza bölümleri ile çalışılıyor. kopyalama olmadığından biri değişince hepsi değişir.
#aşağıdaki gibi kopyalar isek hepsi farklı olur birbirini etkilemez

d =  np.array([1,2,3])

e = d.copy()

f = d.copy()