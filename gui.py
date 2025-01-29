from single_neuron import SingleNeuron
from file_handler import load_data

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from ttkthemes import ThemedTk

class GUI:
    def __init__(self):
        self.window = ThemedTk(theme="arc")
        self.window.title("Regresión Lineal - Neurona Simple")
        self.window.geometry("1350x1250")
        
        self.colors = {
            'bg': '#F5F5F5',           
            'frame_bg': '#FFFFFF',      
            'text': '#2C3E50',          
            'accent': '#34495E',        
            'button': '#2980B9',        
            'button_hover': '#3498DB'   
        }
        
        self.learning_rate = tk.DoubleVar(value=0.01)
        self.epochs = tk.IntVar(value=100)
        self.tolerance = tk.DoubleVar(value=0.0001)
        
        self.data_file_path = tk.StringVar()
        self.X_data = None
        self.Y_data = None
        
        self.setup_interface()
        
    def setup_interface(self):
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Control Frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(control_frame, text="Archivo de datos:").pack(side='left', padx=5)
        ttk.Entry(control_frame, textvariable=self.data_file_path, width=50).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Buscar", command=self.load_data_file).pack(side='left', padx=5)
        
        # Parameters Frame
        params_frame = ttk.LabelFrame(main_frame, text="Parámetros de Entrenamiento")
        params_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(params_frame, text="Tasa de aprendizaje:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.learning_rate, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Épocas:").grid(row=0, column=2, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.epochs, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Tolerancia:").grid(row=0, column=4, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.tolerance, width=10).grid(row=0, column=5, padx=5, pady=5)
        
        ttk.Button(params_frame, text="Iniciar Entrenamiento", command=self.run_training).grid(row=0, column=6, padx=20, pady=5)
        
        # Plots Frame
        plots_frame = ttk.Frame(main_frame)
        plots_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Frame para las gráficas superiores
        top_plots_frame = ttk.Frame(plots_frame)
        top_plots_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        # Gráfica de datos inicial
        self.data_fig = Figure(figsize=(6, 4))
        self.data_ax = self.data_fig.add_subplot(111)
        self.data_canvas = FigureCanvasTkAgg(self.data_fig, master=top_plots_frame)
        self.data_canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)

        # Gráfica de regresión
        self.regression_fig = Figure(figsize=(6, 4))
        self.regression_ax = self.regression_fig.add_subplot(111)
        self.regression_canvas = FigureCanvasTkAgg(self.regression_fig, master=top_plots_frame)
        self.regression_canvas.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)

        # Frame para las gráficas inferiores
        bottom_plots_frame = ttk.Frame(plots_frame)
        bottom_plots_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        # Gráfica de evolución de pesos
        self.weights_fig = Figure(figsize=(6, 4))
        self.weights_ax = self.weights_fig.add_subplot(111)
        self.weights_canvas = FigureCanvasTkAgg(self.weights_fig, master=bottom_plots_frame)
        self.weights_canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)

        # Gráfica de error
        self.error_fig = Figure(figsize=(6, 4))
        self.error_ax = self.error_fig.add_subplot(111)
        self.error_canvas = FigureCanvasTkAgg(self.error_fig, master=bottom_plots_frame)
        self.error_canvas.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)
        
        # Tabla de resultados
        results_frame = ttk.LabelFrame(main_frame, text="Resultados")
        results_frame.pack(fill='x', padx=10, pady=5)
        
        self.results_text = tk.Text(results_frame, height=10, width=50)
        self.results_text.pack(fill='x', padx=5, pady=5)
        
    def load_data_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx;*.xls")
            ]
        )
        if filename:
            try:
                self.data_file_path.set(filename)
                self.X_data, self.Y_data = load_data(filename)
                messagebox.showinfo("Éxito", "Datos cargados correctamente")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def run_training(self):
        if self.X_data is None or self.Y_data is None:
            messagebox.showerror("Error", "Por favor, cargue los datos primero")
            return
                
        try:
            # Normalizar los datos
            X_mean = np.mean(self.X_data, axis=0)
            X_std = np.std(self.X_data, axis=0)
            X_normalized = (self.X_data - X_mean) / (X_std + 1e-8)
            
            # Normalizar Y
            Y_mean = np.mean(self.Y_data)
            Y_std = np.std(self.Y_data)
            Y_normalized = (self.Y_data - Y_mean) / (Y_std + 1e-8)
            
            neuron = SingleNeuron(
                learning_rate=self.learning_rate.get(),
                epochs=self.epochs.get(),
                tolerance=self.tolerance.get()
            )
            
            error_history, weight_history = neuron.train(X_normalized, Y_normalized)
            
            # Desnormalizar para las predicciones
            Y_pred_norm = neuron.predict(X_normalized)
            Y_pred = Y_pred_norm * Y_std + Y_mean
            
            self.plot_training_results(neuron, error_history, weight_history, Y_pred)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_training_results(self, neuron, error_history, weight_history, Y_pred):
        # Plot datos originales y predicciones
        self.data_ax.clear()
        n_features = self.X_data.shape[1]
        
        for i in range(n_features):
            self.data_ax.scatter(self.X_data[:, i], 
                                self.Y_data, 
                                color=f'C{i}', 
                                alpha=0.5, 
                                label=f'X{i+1} vs Y')
                                
        self.data_ax.set_xlabel('Variables X')
        self.data_ax.set_ylabel('Y')
        self.data_ax.set_title('Datos Originales')
        self.data_ax.grid(True)
        self.data_ax.legend()
        self.data_canvas.draw()
        
        # Plot regresión - predicciones vs valores reales
        self.regression_ax.clear()
        self.regression_ax.scatter(self.Y_data, Y_pred, color='blue', alpha=0.5, label='Predicciones vs Real')
        
        min_val = min(min(self.Y_data), min(Y_pred))
        max_val = max(max(self.Y_data), max(Y_pred))
        self.regression_ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Línea ideal')
        
        self.regression_ax.set_xlabel('Valores Reales')
        self.regression_ax.set_ylabel('Predicciones')
        self.regression_ax.set_title('Predicciones vs Valores Reales')
        self.regression_ax.grid(True)
        self.regression_ax.legend()
        self.regression_canvas.draw()
        
        # Plot evolución de pesos
        self.weights_ax.clear()
        weight_history = np.array(weight_history)
        for i in range(weight_history.shape[1]):
            self.weights_ax.plot(range(len(weight_history)), 
                                weight_history[:, i], 
                                label=f'w{i+1}')
        
        self.weights_ax.plot(range(len(weight_history)), 
                            [neuron.bias] * len(weight_history), 
                            '--', 
                            label='bias')
        
        self.weights_ax.set_xlabel('Época')
        self.weights_ax.set_ylabel('Valor')
        self.weights_ax.set_title('Evolución de Pesos')
        self.weights_ax.grid(True)
        self.weights_ax.legend()
        self.weights_canvas.draw()
        
        # Plot evolución del error
        self.error_ax.clear()
        self.error_ax.plot(error_history, 'g-', label='Error MSE')
        self.error_ax.set_xlabel('Época')
        self.error_ax.set_ylabel('Error')
        self.error_ax.set_title('Evolución del Error')
        self.error_ax.grid(True)
        self.error_ax.legend()
        self.error_canvas.draw()
        
        # Mostrar resultados numéricos
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Resultados del entrenamiento:\n")
        for i, w in enumerate(neuron.weights):
            self.results_text.insert(tk.END, f"Peso X{i+1}: {w:.4f}\n")
        self.results_text.insert(tk.END, f"Bias: {neuron.bias:.4f}\n")
        self.results_text.insert(tk.END, f"Error final: {error_history[-1]:.6f}\n")
        self.results_text.insert(tk.END, f"Épocas completadas: {len(error_history)}\n")
        
        # Construir ecuación
        equation = "y = sigmoid("
        for i, w in enumerate(neuron.weights):
            equation += f"{w:.4f}*X{i+1} + "
        equation += f"{neuron.bias:.4f})"
        self.results_text.insert(tk.END, f"Ecuación: {equation}")
        
    def start(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = GUI()
    app.start()