import tensorflow as ts
import numpy as np
import matplotlib.pyplot as plt


x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
y = np.array([1,2,2,4,5,4,6,4,6,7,9,10,11,12,10])


capa1 = ts.keras.layers.Dense(units = 4, input_shape = [1])
salida = ts.keras.layers.Dense(units = 1, input_shape = [1])


modelo = ts.keras.Sequential([capa1,salida])


modelo.compile(
    optimizer = ts.keras.optimizers.Adam(0.01),
    loss = 'mean_squared_error'
)


historial = modelo.fit(x, y, epochs = 500)


datos_prueba = [4,7,9,16,20]
resultados = []
print('Variables de prueba:')
for i in datos_prueba:
    var = modelo.predict([i])[0]
    print(var)
    resultados.append(var) 

datos_prueba2 = np.array(datos_prueba)
resultados2 = np.array(resultados)

plt.plot(x,y,"o", label="Entrada")
plt.plot(datos_prueba2,resultados2,label="Salida")

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regresion Lineal Utilizando TensorFlow')
plt.grid()
plt.legend()
plt.show()