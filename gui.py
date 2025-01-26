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
        self.window.title("Neurona Simple - Entrenamiento")
        self.window.geometry("1350x1400")
        
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
        
        # Gráfica de error
        self.error_fig = Figure(figsize=(6, 4))
        self.error_ax = self.error_fig.add_subplot(111)
        self.error_canvas = FigureCanvasTkAgg(self.error_fig, master=plots_frame)
        self.error_canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
        
        # Gráfica de evolución de pesos
        self.weights_fig = Figure(figsize=(6, 4))
        self.weights_ax = self.weights_fig.add_subplot(111)
        self.weights_canvas = FigureCanvasTkAgg(self.weights_fig, master=plots_frame)
        self.weights_canvas.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)
        
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
            neuron = SingleNeuron(
                learning_rate=self.learning_rate.get(),
                epochs=self.epochs.get(),
                tolerance=self.tolerance.get()
            )
            
            error_history, weight_history = neuron.train(self.X_data, self.Y_data)
            self.plot_training_results(neuron, error_history, weight_history)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def plot_training_results(self, neuron, error_history, weight_history):
        # Gráfica de error
        self.error_ax.clear()
        self.error_ax.plot(error_history, 'b-', label='Error')
        self.error_ax.set_xlabel('Época')
        self.error_ax.set_ylabel('Error Cuadrático Medio')
        self.error_ax.set_title('Evolución del Error')
        self.error_ax.grid(True)
        self.error_ax.legend()
        self.error_canvas.draw()
        
        # Gráfica de evolución de pesos
        self.weights_ax.clear()
        weight_history = np.array(weight_history)
        for i in range(weight_history.shape[1]):
            self.weights_ax.plot(weight_history[:, i], label=f'w{i+1}')
        
        self.weights_ax.set_xlabel('Época')
        self.weights_ax.set_ylabel('Valor del Peso')
        self.weights_ax.set_title('Evolución de los Pesos')
        self.weights_ax.grid(True)
        self.weights_ax.legend()
        self.weights_canvas.draw()
        
        # Mostrar tabla de resultados
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Resultados del entrenamiento:\n\n")
        self.results_text.insert(tk.END, "Pesos finales:\n")
        for i, w in enumerate(neuron.weights):
            self.results_text.insert(tk.END, f"w{i+1}: {w:.6f}\n")
        self.results_text.insert(tk.END, f"bias: {neuron.bias:.6f}\n")
        self.results_text.insert(tk.END, f"\nError final: {error_history[-1]:.6f}")
        self.results_text.insert(tk.END, f"\nÉpocas completadas: {len(error_history)}")
        
    def start(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = GUI()
    app.start()