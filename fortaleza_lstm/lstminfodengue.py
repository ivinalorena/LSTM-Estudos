import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import RootMeanSquaredError,mean_absolute_error
from sklearn.metrics import mean_squared_error
#carrega os dados
df = pd.read_csv("Fortaleza-Dengue.csv")
#remover valroes nulos
#df = df.dropna()
df_modificado = df.loc[:, ['casos_est']].copy()

#normalizar os dados
normalizador = MinMaxScaler(feature_range=(0,1))
df_normalizado = normalizador.fit_transform(df_modificado)
#cria janelas de tempo
previsao = []
valor_real = []
window_size = 4 # Tamanho da janela de tempo 4 semanas =  (~1 mês) // para para prever com os dados de 3 meses = 12

for i in range(window_size,len(df_normalizado)):

    janela = df_normalizado[i-window_size:i, 0]
    previsao.append(janela)
    valor_real.append(df_normalizado[i, 0])

# Converte as listas para arrays numpy
previsao = np.array(previsao)
valor_real = np.array(valor_real)


previsao = np.reshape(previsao, (previsao.shape[0], previsao.shape[1], 1))



tam_treinamento = int(len(df_normalizado) * 0.8) # 80% dos dados para treinamento e 20% para teste
X_treinamento = previsao[:tam_treinamento]
x_teste = previsao[tam_treinamento:]
y_treinamento = valor_real[:tam_treinamento]
y_teste = valor_real[tam_treinamento:]

# criar o modelo LSTM
modelo = Sequential()
#return_sequences: Booleano. Se deve retornar a última saída da sequência de saída ou a sequência completa.
modelo.add(LSTM(units=100, return_sequences=True, input_shape=(previsao.shape[1], 1)))
modelo.add(Dropout(0.3))

modelo.add(LSTM(units=50, return_sequences=True))
modelo.add(Dropout(0.3))
#return_sequences é só para as camadas intermediárias, não para a última camada LSTM.
modelo.add(LSTM(units=50))
modelo.add(Dropout(0.3))

modelo.add(Dense(units=1)) # Camada de saída com 1 unidade
modelo.compile(optimizer='adam', loss='mean_squared_error',metrics=['mean_absolute_error', RootMeanSquaredError()])

modelo.fit(X_treinamento, y_treinamento, batch_size=32, epochs=100, verbose=1)


previsao_treinamento_lstm = modelo.predict(X_treinamento)
previsao_treinamento_desnormalizada = normalizador.inverse_transform(previsao_treinamento_lstm)
y_treinamento_desnormalizado = normalizador.inverse_transform(y_treinamento.reshape(-1, 1))


# Previsão no conjunto de teste
previsao_lstm = modelo.predict(x_teste)
previsao_teste_desnormalizada = normalizador.inverse_transform(previsao_lstm)
y_teste_desnormalizado = normalizador.inverse_transform(y_teste.reshape(-1, 1))


# Cálculo de métricas para TREINAMENTO 
mae_treinamento = mean_absolute_error(y_treinamento_desnormalizado, previsao_treinamento_desnormalizada)
rmse_treinamento = np.sqrt(mean_squared_error(y_treinamento_desnormalizado, previsao_treinamento_desnormalizada))
mape_treinamento = np.mean(np.abs((y_treinamento_desnormalizado - previsao_treinamento_desnormalizada) / y_treinamento_desnormalizado)) * 100

#calculo de metricas para o teste
mae_teste = mean_absolute_error(y_teste_desnormalizado, previsao_teste_desnormalizada)
rmse_teste = np.sqrt(mean_squared_error(y_teste_desnormalizado, previsao_teste_desnormalizada))
mape_teste = np.mean(np.abs((y_teste_desnormalizado - previsao_teste_desnormalizada) / y_teste_desnormalizado)) * 100

print('VERIFICAR SE AS AMOSTRAS ESTÃO SENDO SEPARADAS\n')
print(f"Total de amostras: {len(df_normalizado)}")
print(f"Treinamento: {len(X_treinamento)} amostras ({len(X_treinamento)/len(df_normalizado):.1%})")
print(f"Teste: {len(x_teste)} amostras ({len(x_teste)/len(df_normalizado):.1%})")

print("\nMétricas de TREINAMENTO:")
print(f"MAPE (Treino): {mape_treinamento:.2f}%")
print(f"RMSE (Treino): {rmse_treinamento:.2f}")

print("\nMétricas de TESTE:")
print(f"MAPE (Teste): {mape_teste:.2f}%")
print(f"RMSE (Teste): {rmse_teste:.2f}")



# Plotando os resultados
plt.figure(figsize=(14, 6))

# Dados completos
plt.plot(normalizador.inverse_transform(df_normalizado), color='blue', alpha=0.3, label='Dados Reais')

# Previsões no TREINAMENTO (novo)
train_range = range(window_size, window_size+len(previsao_treinamento_desnormalizada))
plt.plot(train_range, previsao_treinamento_desnormalizada, color='orange', label='Previsão (Treino)')

# Previsões no TESTE
test_range = range(tam_treinamento, tam_treinamento+len(previsao_teste_desnormalizada))
plt.plot(test_range, previsao_teste_desnormalizada, color='red',  label='Previsão (Teste)')


plt.axvline(x=tam_treinamento, color='black', linestyle='--', label='Divisão Treino/Teste')
plt.title('Comparação: Previsões no Treino e Teste')
plt.xlabel('Período')
plt.ylabel('Casos de Dengue')
plt.legend()
plt.grid(True)
plt.show()
