import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU

# Definimos días para la prediccion
prediction_days = 60

# Funciones para el modelo y predicciones
def preparar_datos(company):
    ticker = yf.Ticker(company)
    hist = ticker.history(start='2012-1-1', end='2020-1-1')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(hist['Close'].values.reshape(-1, 1))

    x_train, y_train = [], []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler, hist

def construir_modelo(input_shape):
    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(units=50))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def predecir(company, scaler, model, hist):
    ticker = yf.Ticker(company)
    hist_test = ticker.history(start='2018-1-1', end='2022-1-1')
    actual_prices = hist_test["Close"].values

    total_dataset = pd.concat((hist['Close'], hist_test['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset)-len(hist_test)-prediction_days:].values
    model_inputs = scaler.transform(model_inputs.reshape(-1, 1))

    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
    return actual_prices, predicted_prices

def calcular_rentabilidad(actual_prices, predicted_prices):
    rentability = 1
    for i in range(1, len(actual_prices)):
        if predicted_prices[i] > actual_prices[i-1]:
            rentability *= actual_prices[i] / actual_prices[i-1]
    return (rentability - 1) * 100

# Funciones de la interfaz gráfica
def iniciar_analisis():
    try:
        company1 = combo1.get()
        company2 = combo2.get()

        if company1 == company2:
            messagebox.showerror("Error", "Selecciona dos acciones diferentes.")
            return

        # Preparar y entrenar modelos para ambas empresas
        x_train1, y_train1, scaler1, hist1 = preparar_datos(company1)
        x_train2, y_train2, scaler2, hist2 = preparar_datos(company2)

        model1 = construir_modelo((x_train1.shape[1], 1))
        model2 = construir_modelo((x_train2.shape[1], 1))

        model1.fit(x_train1, y_train1, epochs=25, batch_size=32)
        model2.fit(x_train2, y_train2, epochs=25, batch_size=32)

        # Predecir precios
        actual_prices1, predicted_prices1 = predecir(company1, scaler1, model1, hist1)
        actual_prices2, predicted_prices2 = predecir(company2, scaler2, model2, hist2)

        rentability1 = calcular_rentabilidad(actual_prices1, predicted_prices1)
        rentability2 = calcular_rentabilidad(actual_prices2, predicted_prices2)

        # Actualizar gráficos
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(actual_prices1, color="black", label=f"{company1} real prices")
        ax.plot(predicted_prices1, color="blue", label=f"{company1} predicted prices")
        ax.plot(actual_prices2, color="red", label=f"{company2} real prices")
        ax.plot(predicted_prices2, color="green", label=f"{company2} predicted prices")
        ax.legend()

        canvas.draw()

        # Mostrar rentabilidad
        result_label.config(text=f"Rentabilidad de {company1}: {rentability1:.2f}%\nRentabilidad de {company2}: {rentability2:.2f}%")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Interfaz gráfica con Tkinter
root = tk.Tk()
root.title("Comparador de Predicción de Acciones")
root.geometry("800x600")

# Dropdowns para seleccionar acciones
options = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'FB', 'TSLA', 'NVDA']

label1 = ttk.Label(root, text="Selecciona la primera acción:")
label1.pack(pady=10)
combo1 = ttk.Combobox(root, values=options)
combo1.pack()

label2 = ttk.Label(root, text="Selecciona la segunda acción:")
label2.pack(pady=10)
combo2 = ttk.Combobox(root, values=options)
combo2.pack()

# Botón para iniciar análisis
start_button = ttk.Button(root, text="Iniciar Análisis", command=iniciar_analisis)
start_button.pack(pady=20)

# Gráfico de resultados
fig = plt.Figure(figsize=(6, 4), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Label para mostrar rentabilidad
result_label = ttk.Label(root, text="")
result_label.pack(pady=20)

root.mainloop()
