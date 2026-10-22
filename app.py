import pandas as pd
import numpy as np
import threading
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================
# Configurações de Tema (CustomTkinter)
# ==========================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Cores Temáticas da NBA
NBA_ORANGE = "#F58426"
NBA_ORANGE_HOVER = "#D9691F"
BG_DARK = "#242424"
FRAME_DARK = "#2b2b2b"

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
def run_model(output_text, frame_graficos, epochs, learning_rate, hidden_size, btn_train):
    try:
        output_text.insert("end", "\n Carregando estatísticas da NBA, normalizando e iniciando treinamento...\n")
        output_text.insert("end", f" Parâmetros: {epochs} Épocas | LR: {learning_rate} | Neurônios: {hidden_size}\n")
        output_text.see("end")
        
        # 1. Preparação dos Dados
        df = pd.read_csv("dataset.csv")
        df = df.dropna(subset=['Performance'])
        df['Performance'] = df['Performance'].map({'Bad': 0, 'Good': 1})

        features_to_drop = ['Player', 'Pos', 'Tm', 'FG%.1', 'Performance']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_names = [c for c in numeric_cols if c not in features_to_drop]

        X = df[feature_names].copy()
        y = df['Performance'].values.reshape(-1, 1)
        X = X.fillna(X.median())

        scaler = MinMaxScaler()
        X_np = scaler.fit_transform(X)

        np.random.seed(42)
        indices = np.random.permutation(len(X_np))
        test_size = int(0.2 * len(X_np))

        test_idx, train_idx = indices[:test_size], indices[test_size:]
        X_train, X_test = X_np[train_idx], X_np[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 2. Implementação do Modelo
        input_size = X_train.shape[1]
        output_size = 1

        weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        bias_hidden = np.zeros((1, hidden_size))
        bias_output = np.zeros((1, output_size))

        historico_erro = []
        
        # 3. Treinamento
        for epoch in range(epochs):
            hidden_input = np.dot(X_train, weights_input_hidden) + bias_hidden
            hidden_output = sigmoid(hidden_input)
            final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
            final_output = sigmoid(final_input)
            
            error = y_train - final_output
            d_output = error * sigmoid_derivative(final_output)
            error_hidden = d_output.dot(weights_hidden_output.T)
            d_hidden = error_hidden * sigmoid_derivative(hidden_output)
            
            weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
            bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
            weights_input_hidden += X_train.T.dot(d_hidden) * learning_rate
            bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
            
            if epoch % (max(1, epochs // 100)) == 0:
                historico_erro.append(np.mean(np.abs(error)))

        # 4. Avaliação
        hidden_test = sigmoid(np.dot(X_test, weights_input_hidden) + bias_hidden)
        final_test = sigmoid(np.dot(hidden_test, weights_hidden_output) + bias_output)
        y_pred = (final_test >= 0.5).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        output_text.insert("end", "-"*50 + "\n")
        output_text.insert("end", " TREINAMENTO CONCLUÍDO!\n")
        output_text.insert("end", f" Acurácia: {accuracy:.4f}\n")
        output_text.insert("end", f" Precisão: {precision:.4f}\n")
        output_text.insert("end", f" Recall:   {recall:.4f}\n\n")
        output_text.see("end")

        x_axis_labels = range(0, epochs, max(1, epochs // 100))
        output_text.after(0, plot_graphics, frame_graficos, historico_erro, cm, list(x_axis_labels))
        
        # Reabilitar botão
        btn_train.configure(state="normal", text="INICIAR NOVO TREINAMENTO")

    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro: {str(e)}")
        btn_train.configure(state="normal", text="TREINAR MODELO")

# ==========================================
# Geração de Gráficos (Matplotlib Dark Mode)
# ==========================================
def plot_graphics(frame, historico_erro, cm, x_axis_labels):
    for widget in frame.winfo_children():
        widget.destroy()
        
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    fig.patch.set_facecolor(FRAME_DARK)
    ax1.set_facecolor(FRAME_DARK)
    ax2.set_facecolor(FRAME_DARK)
    
    # Gráfico 1: Curva de Aprendizado
    ax1.plot(x_axis_labels, historico_erro, color=NBA_ORANGE, linewidth=2)
    ax1.set_title("Curva de Aprendizado (Erro Absoluto)", color='white', pad=10)
    ax1.set_xlabel("Épocas", color='white')
    ax1.set_ylabel("Erro", color='white')
    ax1.tick_params(colors='white')
    
    # Gráfico 2: Matriz de Confusão
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bad', 'Good'])
    disp.plot(ax=ax2, cmap='Oranges', colorbar=False)
    ax2.set_title("Matriz de Confusão", color='white', pad=10)
    ax2.tick_params(colors='white')
    
    # CORREÇÃO: Ajuste de cores dos textos da matriz de confusão
    for text in disp.text_.ravel():
        # Valores maiores que a metade do máximo ficam escuros, exigindo texto branco.
        # Valores menores ficam claros, exigindo texto preto.
        text.set_color('white' if float(text.get_text()) > (cm.max() / 2) else 'black')
    
    plt.tight_layout()
    
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# ==========================================
# Threading e Validação
# ==========================================
def start_training_thread(output_text, frame_graficos, ent_epochs, ent_lr, ent_hidden, btn_train):
    try:
        epochs = int(ent_epochs.get())
        learning_rate = float(ent_lr.get())
        hidden_size = int(ent_hidden.get())
    except ValueError:
        messagebox.showerror("Erro", "Valores inválidos! Use números inteiros para Épocas/Neurônios e ponto para Taxa de Aprendizado.")
        return

    btn_train.configure(state="disabled", text="TREINANDO... AGUARDE")
    output_text.delete("1.0", "end")
    
    thread = threading.Thread(target=run_model, args=(output_text, frame_graficos, epochs, learning_rate, hidden_size, btn_train))
    thread.start()

# ==========================================
# Construção da Interface Gráfica Moderna
# ==========================================
def create_ui():
    app = ctk.CTk()
    app.title("NBA Predictor - Rede Neural MLP")
    app.geometry("950x750")
    
    # Título Principal
    lbl_title = ctk.CTkLabel(app, text="PREVISOR DE DESEMPENHO DA NBA", 
                             font=ctk.CTkFont(family="Arial", size=24, weight="bold"), 
                             text_color=NBA_ORANGE)
    lbl_title.pack(pady=(20, 15))

    # --- PAINEL DE CONFIGURAÇÕES ---
    frame_config = ctk.CTkFrame(app, corner_radius=15, fg_color=FRAME_DARK)
    frame_config.pack(fill="x", padx=30, pady=5)
    
    lbl_config_title = ctk.CTkLabel(frame_config, text="⚙️ Configurações da Rede Neural", font=ctk.CTkFont(size=14, weight="bold"))
    lbl_config_title.grid(row=0, column=0, columnspan=6, sticky="w", padx=15, pady=(10, 5))

    ctk.CTkLabel(frame_config, text="Épocas:").grid(row=1, column=0, padx=(15, 5), pady=10, sticky="e")
    ent_epochs = ctk.CTkEntry(frame_config, width=100, border_color=NBA_ORANGE)
    ent_epochs.insert(0, "2000")
    ent_epochs.grid(row=1, column=1, padx=5, pady=10)

    ctk.CTkLabel(frame_config, text="Taxa de Aprendizado (LR):").grid(row=1, column=2, padx=(20, 5), pady=10, sticky="e")
    ent_lr = ctk.CTkEntry(frame_config, width=100, border_color=NBA_ORANGE)
    ent_lr.insert(0, "0.01")
    ent_lr.grid(row=1, column=3, padx=5, pady=10)

    ctk.CTkLabel(frame_config, text="Neurônios Ocultos:").grid(row=1, column=4, padx=(20, 5), pady=10, sticky="e")
    ent_hidden = ctk.CTkEntry(frame_config, width=100, border_color=NBA_ORANGE)
    ent_hidden.insert(0, "16")
    ent_hidden.grid(row=1, column=5, padx=(5, 15), pady=10)
    
    for i in range(6):
        frame_config.grid_columnconfigure(i, weight=1)

    # --- CAIXA DE LOGS (TEXTO) ---
    output_text = ctk.CTkTextbox(app, height=120, corner_radius=15, 
                                 font=("Consolas", 13), fg_color="#1a1a1a", text_color="#e0e0e0")
    output_text.pack(fill="x", padx=30, pady=15)
    output_text.insert("end", "Pronto para iniciar. Ajuste as configurações acima e inicie a partida!\n")

    # ==========================================
    # CORREÇÃO: ANCORAR O BOTÃO E OS GRÁFICOS
    # ==========================================
    
    # 1. Empacotar o botão na base (side="bottom")
    btn_train = ctk.CTkButton(app, text="INICIAR TREINAMENTO DO MODELO", 
                              font=ctk.CTkFont(family="Arial", size=15, weight="bold"),
                              fg_color=NBA_ORANGE, hover_color=NBA_ORANGE_HOVER, text_color="black",
                              corner_radius=10, height=45)
    btn_train.pack(side="bottom", fill="x", padx=30, pady=(0, 20))

    # 2. Empacotar o painel de gráficos para ocupar o espaço livre no meio
    frame_graficos = ctk.CTkFrame(app, corner_radius=15, fg_color=FRAME_DARK)
    frame_graficos.pack(side="top", fill="both", expand=True, padx=30, pady=(0, 15))

    # 3. Adicionar o comando de execução no botão
    btn_train.configure(command=lambda: start_training_thread(output_text, frame_graficos, ent_epochs, ent_lr, ent_hidden, btn_train))

    app.mainloop()

if __name__ == "__main__":
    create_ui()