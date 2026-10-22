import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading

# Novas bibliotecas permitidas
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================
# Funções Auxiliares da Rede Neural
# ==========================================
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

# ==========================================
# Lógica de Treinamento e Avaliação
# ==========================================
def run_model(output_text, frame_graficos):
    try:
        output_text.insert(tk.END, "Carregando dados, normalizando e iniciando treinamento...\n")
        
        # 1. Preparação dos Dados (usando pandas)
        df = pd.read_csv("dataset.csv")
        df = df.dropna(subset=['Performance'])
        df['Performance'] = df['Performance'].map({'Bad': 0, 'Good': 1})

        features_to_drop = ['Player', 'Pos', 'Tm', 'FG%.1', 'Performance']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_names = [c for c in numeric_cols if c not in features_to_drop]

        X = df[feature_names].copy()
        y = df['Performance'].values.reshape(-1, 1)
        X = X.fillna(X.median())

        # Normalização usando biblioteca (MinMaxScaler)
        scaler = MinMaxScaler()
        X_np = scaler.fit_transform(X)

        np.random.seed(42)
        indices = np.random.permutation(len(X_np))
        test_size = int(0.2 * len(X_np))

        test_idx, train_idx = indices[:test_size], indices[test_size:]
        X_train, X_test = X_np[train_idx], X_np[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 2. Implementação do Modelo (MLP do Zero)
        input_size = X_train.shape[1]
        hidden_size = 16
        output_size = 1

        weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        bias_hidden = np.zeros((1, hidden_size))
        bias_output = np.zeros((1, output_size))

        epochs = 2000
        learning_rate = 0.01
        
        # Armazenar o erro para o gráfico
        historico_erro = []

        output_text.insert(tk.END, f"Treinando por {epochs} épocas...\n")
        
        # 3. Treinamento (Matemática Manual)
        for epoch in range(epochs):
            # Forward pass
            hidden_input = np.dot(X_train, weights_input_hidden) + bias_hidden
            hidden_output = sigmoid(hidden_input)
            final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
            final_output = sigmoid(final_input)
            
            # Backpropagation
            error = y_train - final_output
            d_output = error * sigmoid_derivative(final_output)
            error_hidden = d_output.dot(weights_hidden_output.T)
            d_hidden = error_hidden * sigmoid_derivative(hidden_output)
            
            # Atualização
            weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
            bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
            weights_input_hidden += X_train.T.dot(d_hidden) * learning_rate
            bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
            
            # Salvar erro médio absoluto da época
            if epoch % 10 == 0:
                historico_erro.append(np.mean(np.abs(error)))

        # 4. Avaliação
        hidden_test = sigmoid(np.dot(X_test, weights_input_hidden) + bias_hidden)
        final_test = sigmoid(np.dot(hidden_test, weights_hidden_output) + bias_output)
        y_pred = (final_test >= 0.5).astype(int)

        # Geração da Matriz de Confusão usando biblioteca
        cm = confusion_matrix(y_test, y_pred)
        # Extração manual para cálculo de métricas
        tn, fp, fn, tp = cm.ravel()

        # Cálculo manual de métricas estritamente sem biblioteca
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Atualização da Interface de Texto
        output_text.insert(tk.END, "-"*40 + "\n")
        output_text.insert(tk.END, "Treinamento Concluído!\n")
        output_text.insert(tk.END, f"Acurácia: {accuracy:.4f}\n")
        output_text.insert(tk.END, f"Precisão: {precision:.4f}\n")
        output_text.insert(tk.END, f"Recall:   {recall:.4f}\n\n")
        output_text.see(tk.END)

        # Atualização dos Gráficos (usando event loop principal para evitar erros de thread no Matplotlib)
        output_text.after(0, plot_graphics, frame_graficos, historico_erro, cm)

    except FileNotFoundError:
        messagebox.showerror("Erro", "O arquivo 'dataset.csv' não foi encontrado no diretório!")
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro inesperado: {str(e)}")

# ==========================================
# Geração de Gráficos (Matplotlib)
# ==========================================
def plot_graphics(frame, historico_erro, cm):
    # Limpar gráficos antigos
    for widget in frame.winfo_children():
        widget.destroy()
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Gráfico 1: Curva de Erro
    ax1.plot(range(0, 2000, 10), historico_erro, color='red')
    ax1.set_title("Curva de Aprendizado (Erro Absoluto)")
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Erro")
    
    # Gráfico 2: Matriz de Confusão
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bad', 'Good'])
    disp.plot(ax=ax2, cmap='Blues', colorbar=False)
    ax2.set_title("Matriz de Confusão")
    
    plt.tight_layout()
    
    # Embutir no Tkinter
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Usa threading para manter a interface responsiva durante o loop da rede
def start_training_thread(output_text, frame_graficos):
    output_text.delete(1.0, tk.END)
    thread = threading.Thread(target=run_model, args=(output_text, frame_graficos))
    thread.start()

# ==========================================
# Construção da Interface Gráfica (UI)
# ==========================================
def create_ui():
    janela = tk.Tk()
    janela.title("Previsor de Desempenho NBA - MLP do Zero")
    janela.geometry("900x650")
    janela.configure(padx=10, pady=10)

    # Painel Superior (Texto e Botão)
    frame_top = tk.Frame(janela)
    frame_top.pack(fill=tk.X, pady=5)
    
    lbl_title = tk.Label(frame_top, text="Rede Neural Multicamada (MLP) com Gráficos", font=("Arial", 14, "bold"))
    lbl_title.pack()

    output_text = scrolledtext.ScrolledText(frame_top, width=100, height=8, font=("Consolas", 10))
    output_text.pack(pady=5)
    output_text.insert(tk.END, "Clique no botão abaixo para treinar a rede e gerar os gráficos...\n")

    # Painel Inferior (Gráficos)
    frame_graficos = tk.Frame(janela, bg="white", relief=tk.SUNKEN, bd=1)
    frame_graficos.pack(fill=tk.BOTH, expand=True, pady=5)

    btn_train = tk.Button(frame_top, text="Treinar Modelo e Avaliar", 
                          font=("Arial", 12), bg="lightblue", 
                          command=lambda: start_training_thread(output_text, frame_graficos))
    btn_train.pack(fill=tk.X)

    janela.mainloop()

if __name__ == "__main__":
    create_ui()