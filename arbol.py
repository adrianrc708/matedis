from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU

# Funciones para el modelo y predicciones (idénticas)
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


# Aplicar estilo de matplotlib
plt.style.use("bmh")

# Definimos días para la prediccion
prediction_days = 60


def preparar_datos(company, prediction_days):
    # Obtener los datos históricos
    ticker = yf.Ticker(company)
    hist = ticker.history(start='2012-1-1', end='2020-1-1')
    
    # Verificar que haya suficientes datos antes de escalar
    if len(hist['Close']) < prediction_days:
        raise ValueError(f"No hay suficientes datos para el ticker {company}. Se requieren al menos {prediction_days} días.")

    # Escalar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(hist['Close'].values.reshape(-1, 1))

    # Preparar los datos para entrenamiento
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
def mostrar_ventana_seleccion():
    ventana_inicio.withdraw()
    ventana_seleccion.deiconify()
    
# Funciones de la interfaz gráfica
def iniciar_analisis():
    try:
        company1 = combo1.get()
        company2 = combo2.get()

        if company1 == company2:
            messagebox.showerror("Error", "Selecciona dos acciones diferentes.")
            return

        # Preparar y entrenar modelos para ambas empresas
        x_train1, y_train1, scaler1, hist1 = preparar_datos(company1, prediction_days)
        x_train2, y_train2, scaler2, hist2 = preparar_datos(company2, prediction_days)

        model1 = construir_modelo((x_train1.shape[1], 1))
        model2 = construir_modelo((x_train2.shape[1], 1))

        model1.fit(x_train1, y_train1, epochs=12, batch_size=64)
        model2.fit(x_train2, y_train2, epochs=12, batch_size=64)

        # Predecir precios
        global actual_prices1, predicted_prices1, actual_prices2, predicted_prices2
        actual_prices1, predicted_prices1 = predecir(company1, scaler1, model1, hist1)
        actual_prices2, predicted_prices2 = predecir(company2, scaler2, model2, hist2)

        # Mostrar gráficos de la segunda ventana
        fig1.clear()
        fig2.clear()
        ax1 = fig1.add_subplot(111)
        ax1.plot(actual_prices1, color="black", label=f"{company1} real prices")
        ax1.plot(predicted_prices1, color="blue", label=f"{company1} predicted prices")
        ax1.legend()
        
        ax2 = fig2.add_subplot(111)
        ax2.plot(actual_prices2, color="red", label=f"{company2} real prices")
        ax2.plot(predicted_prices2, color="green", label=f"{company2} predicted prices")
        ax2.legend()

        canvas1.draw()
        canvas2.draw()
    except Exception as e:
        messagebox.showerror("Error", str(e))

def regresar_seleccion():
    ventana_graficos.withdraw()  
    ventana_seleccion.deiconify()

def mostrar_ventana_resultados():
    ventana_seleccion.withdraw()
    ventana_graficos.deiconify()
    mostrar_resultados_finales()

def mostrar_resultados_finales():  
    company1 = combo1.get()
    company2 = combo2.get()
    
    fig.clear()
    ax = fig.add_subplot(111)
    ax.plot(actual_prices1, color="black", label=f"{company1} real prices")
    ax.plot(predicted_prices1, color="blue", label=f"{company1} predicted prices")
    ax.plot(actual_prices2, color="red", label=f"{company2} real prices")
    ax.plot(predicted_prices2, color="green", label=f"{company2} predicted prices")
    ax.legend()
        
    # Cuadrícula y leyenda
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="upper left", fontsize=10)

    canvas.draw()

    rentability1 = calcular_rentabilidad(actual_prices1, predicted_prices1)
    rentability2 = calcular_rentabilidad(actual_prices2, predicted_prices2)
    result_label.configure(text=f"Rentabilidad de {company1}: {rentability1:.2f}%\nRentabilidad de {company2}: {rentability2:.2f}%")


def centrar_ventana(ventana, ancho_ventana, alto_ventana):
    ventana.update_idletasks()
    ancho_pantalla = ventana.winfo_screenwidth()
    alto_pantalla = ventana.winfo_screenheight()
    x = (ancho_pantalla // 2) - (ancho_ventana // 2)
    y = (alto_pantalla // 2) - (alto_ventana // 2)
    ventana.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")
    ventana.resizable(True, True)  # Deshabilitar redimensionamiento

def cerrar_programa():
    if ventana_seleccion.winfo_exists():
        ventana_seleccion.destroy()
    if ventana_graficos.winfo_exists():
        ventana_graficos.destroy()
    ventana_inicio.quit()
    ventana_inicio.destroy()

def generar_arbol_decisiones(company1, company2, actual_prices1, actual_prices2):
    # Generar datos de rentabilidad como ejemplo
    labels = [0 if actual_prices1[i] > actual_prices2[i] else 1 for i in range(len(actual_prices1))]
    features = np.array(list(zip(actual_prices1, actual_prices2)))

    # Crear y entrenar el modelo
    tree_model = DecisionTreeClassifier(max_depth=None, random_state=42)
    tree_model.fit(features, labels)
    
    # Crear gráfico del árbol de decisiones
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(tree_model, 
              feature_names=["Precio 1", "Precio 2"], 
              class_names=[company1, company2], 
              filled=True, ax=ax, rounded=True, 
              proportion=False, fontsize=12, 
              impurity=False,  # No mostrar el índice de impureza
              node_ids=True,   # Mostrar ID de nodos
              label='all')  # Mostrar etiquetas con la decisión final

    # Mostrar el árbol
    plt.show()

    # Ejemplo con los últimos precios para la predicción
    ejemplo_precio = np.array([[actual_prices1[-1], actual_prices2[-1]]])  # Usa el último par de precios
    prediccion = tree_model.predict(ejemplo_precio)

    # Mostrar recomendación en base a la predicción
    if prediccion == 0:
        decision = f"La mejor opción es {company1}."
    else:
        decision = f"La mejor opción es {company2}."
    
    decision_label.configure(text=decision)



# Configuración de la interfaz gráfica con customtkinter
ctk.set_appearance_mode("dark")  # Modo oscuro
ctk.set_default_color_theme("blue")  # Tema de color

# Ventana de inicio
ventana_inicio = ctk.CTk()
ventana_inicio.title("Comparador de Predicción de Acciones")

ventana_inicio.geometry("1300x680")
centrar_ventana(ventana_inicio, 1300, 680)

inicio_label = ctk.CTkLabel(ventana_inicio, text="Comparador de Predicción de Acciones", font=("Segoe UI", 60))
inicio_label.pack(pady=20)
boton_iniciar = ctk.CTkButton(ventana_inicio, text="Iniciar", command=mostrar_ventana_seleccion)
boton_iniciar.pack(pady=20)

# Ventana de selección
ventana_seleccion = ctk.CTkToplevel(ventana_inicio)
ventana_seleccion.title("Selección de Acciones")

ventana_seleccion.geometry("1300x680")
ventana_seleccion.withdraw()
centrar_ventana(ventana_seleccion, 1300, 680)

# Dropdowns para seleccionar acciones
options = [
    'AAPL', 'MSFT', 'AMZN', 'GOOG', 'FB', 'TSLA', 'NVDA', 'NFLX', 'CSCO', 'PEP', 'KO', 'MRK', 
    'PFE', 'BA', 'XOM', 'WMT', 'IBM', 'ORCL', 'AMD', 'UBER', 'TWTR','SHOP', 'ADBE', 'CRM', 
    'SPOT', 'SBUX', 'T', 'VZ', 'MCD', 'NKE', 'COST', 'BKNG', 'CVX', 'F', 'GM', 'ZM', 'ROKU', 'PLTR'
]

label1 = ctk.CTkLabel(ventana_seleccion, text="Selecciona la primera acción:", font=("Segoe UI", 18))
label1.pack(pady=10)
combo1 = ctk.CTkComboBox(ventana_seleccion, values=options)
combo1.pack()

label2 = ctk.CTkLabel(ventana_seleccion, text="Selecciona la segunda acción:", font=("Segoe UI", 18))
label2.pack(pady=10)
combo2 = ctk.CTkComboBox(ventana_seleccion, values=options)
combo2.pack()

# Botón para iniciar análisis
start_button = ctk.CTkButton(ventana_seleccion, text="Iniciar Análisis", command=iniciar_analisis)
start_button.pack(pady=20)

# Gráfico de resultados
fig1, fig2 = plt.Figure(figsize=(6, 4), dpi=100), plt.Figure(figsize=(6, 4), dpi=100)
canvas1 = FigureCanvasTkAgg(fig1, master=ventana_seleccion)
canvas1.get_tk_widget().pack(side="left", fill="both", expand=False)
canvas2 = FigureCanvasTkAgg(fig2, master=ventana_seleccion)
canvas2.get_tk_widget().pack(side="right", fill="both", expand=False)

boton_comparar = ctk.CTkButton(ventana_seleccion, text="Comparar", command=mostrar_ventana_resultados)
boton_comparar.pack(pady=20)

# Ventana de gráficos combinados
ventana_graficos = ctk.CTkToplevel(ventana_inicio)
ventana_graficos.title("Resultados de comparación")
ventana_graficos.geometry("1300x680")
ventana_graficos.withdraw()
centrar_ventana(ventana_graficos, 1300, 680)

# Botón para regresar a la ventana principal (desde la ventana de gráficos)
boton_regresar = ctk.CTkButton(ventana_graficos, text="Regresar", command=regresar_seleccion)
boton_regresar.pack(pady=20)

#RECOMENDACIÓN
decision_label = ctk.CTkLabel(ventana_graficos, text="", font=("Segoe UI", 18))
decision_label.pack(pady=20)

# Continúa con el resto de widgets de la ventana de gráficos...
boton_arbol = ctk.CTkButton(ventana_graficos, text="Generar Árbol de Decisiones", 
                            command=lambda: generar_arbol_decisiones(combo1.get(), combo2.get(), 
                                                                     actual_prices1, actual_prices2))
boton_arbol.pack(pady=10)


# Gráfico para los resultados finales
fig = plt.Figure(figsize=(8, 6), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=ventana_graficos)
canvas.get_tk_widget().pack()

# Label para mostrar rentabilidad
result_label = ctk.CTkLabel(ventana_graficos, text="", font=("Segoe UI", 20))
result_label.pack(pady=20)

ventana_inicio.protocol("WM_DELETE_WINDOW", cerrar_programa)

ventana_inicio.mainloop()