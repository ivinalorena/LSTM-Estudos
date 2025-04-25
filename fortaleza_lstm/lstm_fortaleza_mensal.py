import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.metrics import RootMeanSquaredError, mean_absolute_error


#carrega os dados
df= pd.read_csv('Fortaleza-Dengue.csv')

# Converter a coluna de data para datetime (se ainda não estiver)
df['data_iniSE'] = pd.to_datetime(df['data_iniSE'])
# Criar uma nova coluna no formato 'ano-mês' (AAA-MM)
df['ano_mes'] = df['data_iniSE'].dt.to_period('M')
# Agrupar os casos por mês somando os casos semanais
df_mensal = df.groupby('ano_mes')['casos_est'].sum().reset_index()

#print(df_mensal)

#normalizar os dados
normalizador = MinMaxScaler(feature_range=(0,1))
# Normalizar apenas a coluna de casos
df_normalizado = normalizador.fit_transform(df_mensal[['casos_est']])
#cria janelas de tempo
previsao = []
valor_real = []
window_size = 3 #como mudamos para mensal, o tamanho da janela de tempo é 3 meses (com três meses ocorreu muito erros, com 6 teve ainda mais)

for i in range(window_size,len(df_normalizado)):
    # Cria a janela de tempo
    #não entendi essa linha
    janela = df_normalizado[i-window_size:i, 0]
    # Adiciona a previsão
    previsao.append(janela)
    # Adiciona o valor real
    valor_real.append(df_normalizado[i, 0])

# Converte as listas para arrays numpy
previsao = np.array(previsao)
valor_real = np.array(valor_real)


# Reshape para [samples, time steps, features]
#ajusta o formato dos dados para o modelo LSTM
# O modelo LSTM espera os dados de entrada na forma [amostras, passos de tempo, características]
previsao = np.reshape(previsao, (previsao.shape[0], previsao.shape[1], 1))


#separar o conjunto de treinamento
tam_treinamento = int(len(df_normalizado)*0.8)  # 80% dos dados para treinamento e 20% para teste
#separação dos dados em treino e teste 
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

# O parâmetro batch_size define quantas amostras serão processadas antes de atualizar os pesos do modelo.
# O parâmetro epochs define quantas vezes o modelo verá todo o conjunto de dados durante o treinamento.
modelo.fit(X_treinamento, y_treinamento, batch_size=32, epochs=100, verbose=1)



# Previsão no conjunto de TREINAMENTO (novo)
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

# Valores reais de TESTE
#plt.plot(test_range, y_teste_desnormalizado, color='green', linewidth=2, label='Valores Reais (Teste)')

plt.axvline(x=tam_treinamento, color='black', linestyle='--', label='Divisão Treino/Teste')
plt.title('Comparação: Previsões no Treino e Teste')
plt.xlabel('Período')
plt.ylabel('Casos de Dengue')
plt.legend()
plt.grid(True)
plt.show()