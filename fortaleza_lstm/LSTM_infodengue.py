import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, accuracy_score

# Carregar os dados
df = pd.read_csv("Fortaleza-Dengue.csv")

# Converter a coluna de datas para o formato datetime
df['data_iniSE'] = pd.to_datetime(df['data_iniSE'])

# Criar uma nova coluna 'ano_mes' com o formato AAAA-MM
df['ano_mes'] = df['data_iniSE'].dt.to_period('M')

# Agrupar os casos por mês
df_mensal = df.groupby('ano_mes')['casos_est'].sum().reset_index()

# Converter 'ano_mes' para datetime novamente se quiser plotar ou usar depois
df_mensal['ano_mes'] = df_mensal['ano_mes'].dt.to_timestamp()

# Visualizar o resultado
print(df_mensal)

# Usar apenas os valores de casos
data = df_mensal['casos_est'].values
data = data.reshape(-1, 1)

# Normalizar os dados
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)


#Criar função para transformar os dados em janelas de tempo

def create_dataset(dataset, window_size=5):
    dataX, dataY = [], []
    for i in range(len(dataset) - window_size):
        a = dataset[i:(i + window_size), 0]
        dataX.append(a)
        dataY.append(dataset[i + window_size, 0])
    return np.array(dataX), np.array(dataY)

#Dividir em treino e teste

train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:]

window_size = 5 # Tamanho da janela de tempo
trainX, trainY = create_dataset(train, window_size)
testX, testY = create_dataset(test, window_size)

# Reshape para [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#Criar e treinar o modelo LSTM

model = Sequential()
model.add(LSTM(64, input_shape=(1, window_size)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

#Fazer previsões

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Inverter a normalização
trainPredict = scaler.inverse_transform(trainPredict)
trainY_inv = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY_inv = scaler.inverse_transform([testY])

mape = np.mean(np.abs((testY_inv - testPredict) / testY_inv)) * 100
print(f"MAPE (Desnormalizado): {mape:.2f}%")

#Plotar os resultados
#print(accuracy_score(testY, trainY))
# Plotar dados reais e previsões
plt.plot(scaler.inverse_transform(data), label='Dados reais')
plt.plot(np.arange(window_size, window_size+len(trainPredict)), trainPredict, label='Previsão Treino')
plt.plot(np.arange(len(data)-len(testPredict), len(data)), testPredict, label='Previsão Teste')
plt.legend()
plt.show()
