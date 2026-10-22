import pandas as pd
import numpy as np
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import customtkinter as ctk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================
# Configurações de Tema
# ==========================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

NBA_ORANGE = "#F58426"
NBA_ORANGE_HOVER = "#D9691F"
BG_DARK = "#242424"
FRAME_DARK = "#2b2b2b"

# ==========================================
# Visualizador de Dados (Item 1 do Regimento)
# ==========================================
def abrir_visualizador_dados():
    try:
        # Carregando com Pandas apenas para leitura e visualização básica
        df = pd.read_csv("dataset.csv")
        
        janela_dados = ctk.CTkToplevel()
        janela_dados.title("Tabela de Dados (Dataset NBA)")
        janela_dados.geometry("1000x500")
        janela_dados.transient() # Mantém a janela sobre a principal
        
        lbl_titulo = ctk.CTkLabel(janela_dados, text="📊 Visualização Bruta dos Dados (Primeiras 150 linhas)", font=ctk.CTkFont(size=16, weight="bold"))
        lbl_titulo.pack(pady=(15, 5))
        
        # Estilizando a tabela para combinar com o Dark Mode
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background=FRAME_DARK, foreground="white", fieldbackground=FRAME_DARK, borderwidth=0)
        style.map("Treeview", background=[("selected", NBA_ORANGE)])
        style.configure("Treeview.Heading", background="#1a1a1a", foreground="white", relief="flat", font=('Arial', 10, 'bold'))

        frame_tabela = ctk.CTkFrame(janela_dados)
        frame_tabela.pack(fill="both", expand=True, padx=20, pady=(5, 20))
        
        scroll_y = ttk.Scrollbar(frame_tabela, orient="vertical")
        scroll_x = ttk.Scrollbar(frame_tabela, orient="horizontal")
        
        tree = ttk.Treeview(frame_tabela, yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        scroll_y.config(command=tree.yview)
        scroll_x.config(command=tree.xview)
        
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        tree.pack(side="left", fill="both", expand=True)
        
        tree["columns"] = list(df.columns)
        tree["show"] = "headings"
        
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="center")
            
        # Inserir dados na tabela
        for index, row in df.head(150).iterrows():
            tree.insert("", "end", values=list(row))
            
        lbl_resumo = ctk.CTkLabel(janela_dados, text=f"Total de Registros no Arquivo: {len(df)} linhas | Total de Colunas: {len(df.columns)}", font=ctk.CTkFont(size=13))
        lbl_resumo.pack(pady=(0, 15))
        
    except FileNotFoundError:
        messagebox.showerror("Erro", "O arquivo 'dataset.csv' não foi encontrado!")
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao carregar os dados: {str(e)}")

# ==========================================
# Funções Matemáticas da Rede Neural
# ==========================================
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

def manual_evaluation(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    cm = np.array([[TN, FP], 
                   [FN, TP]])
                   
    return cm, accuracy, precision, recall

# ==========================================
# Lógica Principal e Treinamento
# ==========================================
def carregar_dados_para_treino():
    df = pd.read_csv("dataset.csv")
    df = df.dropna(subset=['Performance'])
    df['Performance'] = df['Performance'].map({'Bad': 0, 'Good': 1})

    features_to_drop = ['Player', 'Pos', 'Tm', 'FG%.1', 'Performance']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_names = [c for c in numeric_cols if c not in features_to_drop]

    X = df[feature_names].copy()
    y = df['Performance'].values.reshape(-1, 1)
    
    # Tratamento de dados faltantes (mediana)
    X = X.fillna(X.median())

    # NORMALIZAÇÃO MANUAL (Sem sklearn)
    X_min = X.min()
    X_max = X.max()
    X_np = ((X - X_min) / (X_max - X_min)).values

    np.random.seed(42)
    indices = np.random.permutation(len(X_np))
    test_size = int(0.2 * len(X_np))

    test_idx, train_idx = indices[:test_size], indices[test_size:]
    X_train, X_test = X_np[train_idx], X_np[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test, feature_names

def treinar_mlp(X_train, y_train, epochs, learning_rate, hidden_size):
    input_size = X_train.shape[1]
    output_size = 1

    weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
    weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
    bias_hidden = np.zeros((1, hidden_size))
    bias_output = np.zeros((1, output_size))

    historico_erro = []
    
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
            
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, historico_erro

def run_model(output_text, frame_graficos, epochs, learning_rate, hidden_size, btn_train):
    try:
        output_text.insert("end", "\n Extraindo variáveis, aplicando normalização manual e treinando a rede...\n")
        output_text.see("end")
        
        X_train, X_test, y_train, y_test, feature_names = carregar_dados_para_treino()
        
        w_in_hid, w_hid_out, b_hid, b_out, historico_erro = treinar_mlp(
            X_train, y_train, epochs, learning_rate, hidden_size
        )

        hidden_test = sigmoid(np.dot(X_test, w_in_hid) + b_hid)
        final_test = sigmoid(np.dot(hidden_test, w_hid_out) + b_out)
        y_pred = (final_test >= 0.5).astype(int)

        cm, accuracy, precision, recall = manual_evaluation(y_test, y_pred)

        importances = np.sum(np.abs(w_in_hid), axis=1)
        top_indices = np.argsort(importances)[::-1][:3]
        
        output_text.insert("end", "-"*50 + "\n")
        output_text.insert("end", " TREINAMENTO CONCLUÍDO!\n")
        output_text.insert("end", f" Acurácia: {accuracy:.4f} | Precisão: {precision:.4f} | Recall: {recall:.4f}\n\n")
        output_text.insert("end", " VARIÁVEIS DE MAIOR IMPACTO NO DESEMPENHO:\n")
        for idx in top_indices:
            output_text.insert("end", f"   - {feature_names[idx]} (Peso absoluto: {importances[idx]:.2f})\n")
        output_text.insert("end", "-"*50 + "\n")
        output_text.see("end")

        x_axis_labels = range(0, epochs, max(1, epochs // 100))
        output_text.after(0, plot_graphics, frame_graficos, historico_erro, cm, list(x_axis_labels))
        
        btn_train.configure(state="normal", text=" INICIAR TREINAMENTO DO MODELO")

    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro: {str(e)}")
        btn_train.configure(state="normal", text=" INICIAR TREINAMENTO DO MODELO")

# ==========================================
# Busca por Melhores Hiperparâmetros
# ==========================================
def run_grid_search(output_text, ent_epochs, ent_lr, ent_hidden, btn_tune, btn_train):
    try:
        output_text.insert("end", "\n🔍 Iniciando busca pelos melhores hiperparâmetros...\n")
        output_text.see("end")
        
        X_train, X_test, y_train, y_test, _ = carregar_dados_para_treino()
        
        grid_epochs = [500, 1000, 2000]
        grid_lrs = [0.01, 0.05, 0.1]
        grid_hidden = [8, 16, 32]
        
        best_acc = 0
        best_params = None
        
        for ep in grid_epochs:
            for lr in grid_lrs:
                for hd in grid_hidden:
                    w_in_hid, w_hid_out, b_hid, b_out, _ = treinar_mlp(X_train, y_train, ep, lr, hd)
                    
                    h_test = sigmoid(np.dot(X_test, w_in_hid) + b_hid)
                    f_test = sigmoid(np.dot(h_test, w_hid_out) + b_out)
                    y_pred = (f_test >= 0.5).astype(int)
                    
                    _, acc, _, _ = manual_evaluation(y_test, y_pred)
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (ep, lr, hd)

        output_text.insert("end", f" MELHOR COMBINAÇÃO ENCONTRADA (Acurácia: {best_acc:.4f}):\n")
        output_text.insert("end", f"   - Épocas: {best_params[0]}\n   - LR: {best_params[1]}\n   - Neurônios: {best_params[2]}\n")
        output_text.insert("end", " Os campos foram atualizados! Clique em iniciar treinamento.\n")
        output_text.see("end")
        
        ent_epochs.delete(0, 'end'); ent_epochs.insert(0, str(best_params[0]))
        ent_lr.delete(0, 'end'); ent_lr.insert(0, str(best_params[1]))
        ent_hidden.delete(0, 'end'); ent_hidden.insert(0, str(best_params[2]))
        
        btn_tune.configure(state="normal", text=" BUSCAR MELHORES HIPERPARÂMETROS")
        btn_train.configure(state="normal")
        
    except Exception as e:
        messagebox.showerror("Erro", f"Erro na otimização: {str(e)}")
        btn_tune.configure(state="normal", text=" BUSCAR MELHORES HIPERPARÂMETROS")
        btn_train.configure(state="normal")

# ==========================================
# Gráficos (Matplotlib Manual)
# ==========================================
def plot_graphics(frame, historico_erro, cm, x_axis_labels):
    for widget in frame.winfo_children():
        widget.destroy()
        
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    fig.patch.set_facecolor(FRAME_DARK)
    ax1.set_facecolor(FRAME_DARK)
    ax2.set_facecolor(FRAME_DARK)
    
    ax1.plot(x_axis_labels, historico_erro, color=NBA_ORANGE, linewidth=2)
    ax1.set_title("Curva de Aprendizado", color='white', pad=10)
    ax1.set_xlabel("Épocas", color='white')
    ax1.set_ylabel("Erro", color='white')
    
    cax = ax2.matshow(cm, cmap='Oranges')
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = 'white' if val > (cm.max() / 2) else 'black'
            ax2.text(j, i, str(val), va='center', ha='center', color=color, fontweight='bold')
            
    ax2.set_title("Matriz de Confusão", color='white', pad=10)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Bad (0)', 'Good (1)'])
    ax2.set_yticklabels(['Bad (0)', 'Good (1)'])
    ax2.set_xlabel("Previsto", color='white')
    ax2.set_ylabel("Real", color='white')
    ax2.xaxis.set_ticks_position('bottom')
    
    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# ==========================================
# Threads
# ==========================================
def start_training_thread(output_text, frame_graficos, ent_epochs, ent_lr, ent_hidden, btn_train):
    try:
        epochs = int(ent_epochs.get())
        learning_rate = float(ent_lr.get())
        hidden_size = int(ent_hidden.get())
    except ValueError:
        return messagebox.showerror("Erro", "Valores inválidos nos hiperparâmetros.")

    btn_train.configure(state="disabled", text="TREINANDO... AGUARDE")
    threading.Thread(target=run_model, args=(output_text, frame_graficos, epochs, learning_rate, hidden_size, btn_train)).start()

def start_grid_search_thread(output_text, ent_epochs, ent_lr, ent_hidden, btn_tune, btn_train):
    btn_tune.configure(state="disabled", text="BUSCANDO... AGUARDE")
    btn_train.configure(state="disabled")
    threading.Thread(target=run_grid_search, args=(output_text, ent_epochs, ent_lr, ent_hidden, btn_tune, btn_train)).start()

# ==========================================
# Interface
# ==========================================
def create_ui():
    app = ctk.CTk()
    app.title("NBA Predictor - Rede Neural MLP")
    app.geometry("950x820")
    
    lbl_title = ctk.CTkLabel(app, text=" PREVISOR DE DESEMPENHO DA NBA", font=ctk.CTkFont(family="Arial", size=24, weight="bold"), text_color=NBA_ORANGE)
    lbl_title.pack(pady=(20, 15))

    frame_config = ctk.CTkFrame(app, corner_radius=15, fg_color=FRAME_DARK)
    frame_config.pack(fill="x", padx=30, pady=5)
    
    lbl_config_title = ctk.CTkLabel(frame_config, text=" Configurações da Rede Neural", font=ctk.CTkFont(size=14, weight="bold"))
    lbl_config_title.grid(row=0, column=0, columnspan=6, sticky="w", padx=15, pady=(10, 5))

    ctk.CTkLabel(frame_config, text="Épocas:").grid(row=1, column=0, padx=(15, 5), pady=10, sticky="e")
    ent_epochs = ctk.CTkEntry(frame_config, width=100, border_color=NBA_ORANGE)
    ent_epochs.insert(0, "2000")
    ent_epochs.grid(row=1, column=1, padx=5, pady=10)

    ctk.CTkLabel(frame_config, text="Taxa de Aprendizado:").grid(row=1, column=2, padx=(20, 5), pady=10, sticky="e")
    ent_lr = ctk.CTkEntry(frame_config, width=100, border_color=NBA_ORANGE)
    ent_lr.insert(0, "0.01")
    ent_lr.grid(row=1, column=3, padx=5, pady=10)

    ctk.CTkLabel(frame_config, text="Neurônios:").grid(row=1, column=4, padx=(20, 5), pady=10, sticky="e")
    ent_hidden = ctk.CTkEntry(frame_config, width=100, border_color=NBA_ORANGE)
    ent_hidden.insert(0, "16")
    ent_hidden.grid(row=1, column=5, padx=(5, 15), pady=10)
    
    for i in range(6): frame_config.grid_columnconfigure(i, weight=1)

    # Painel de Botões Secundários
    frame_botoes = ctk.CTkFrame(app, fg_color="transparent")
    frame_botoes.pack(fill="x", padx=30, pady=(5, 10))
    frame_botoes.grid_columnconfigure(0, weight=1)
    frame_botoes.grid_columnconfigure(1, weight=1)

    btn_dados = ctk.CTkButton(frame_botoes, text=" VISUALIZAR DADOS (DATASET)", font=ctk.CTkFont(size=13, weight="bold"),
                              fg_color="#1f538d", hover_color="#14375e", corner_radius=10, height=35, command=abrir_visualizador_dados)
    btn_dados.grid(row=0, column=0, padx=(0, 5), sticky="ew")

    btn_tune = ctk.CTkButton(frame_botoes, text=" BUSCAR MELHORES HIPERPARÂMETROS", font=ctk.CTkFont(size=13, weight="bold"),
                             fg_color="#4CAF50", hover_color="#45a049", corner_radius=10, height=35)
    btn_tune.grid(row=0, column=1, padx=(5, 0), sticky="ew")

    output_text = ctk.CTkTextbox(app, height=180, corner_radius=15, font=("Consolas", 13), fg_color="#1a1a1a", text_color="#e0e0e0")
    output_text.pack(fill="x", padx=30, pady=5)
    output_text.insert("end", "Pronto para iniciar.\n1. Use 'Visualizar Dados' para ver as estatísticas brutas.\n2. Use 'Buscar Melhores Hiperparâmetros' para encontrar os melhores valores matematicamente.\n3. Inicie o treinamento para ver a Matriz de Confusão e o Impacto das Variáveis.\n")

    btn_train = ctk.CTkButton(app, text=" INICIAR TREINAMENTO DO MODELO", font=ctk.CTkFont(family="Arial", size=15, weight="bold"),
                              fg_color=NBA_ORANGE, hover_color=NBA_ORANGE_HOVER, text_color="black", corner_radius=10, height=45)
    btn_train.pack(side="bottom", fill="x", padx=30, pady=(0, 20))

    frame_graficos = ctk.CTkFrame(app, corner_radius=15, fg_color=FRAME_DARK)
    frame_graficos.pack(side="top", fill="both", expand=True, padx=30, pady=(10, 15))

    btn_train.configure(command=lambda: start_training_thread(output_text, frame_graficos, ent_epochs, ent_lr, ent_hidden, btn_train))
    btn_tune.configure(command=lambda: start_grid_search_thread(output_text, ent_epochs, ent_lr, ent_hidden, btn_tune, btn_train))

    app.mainloop()

if __name__ == "__main__":
    create_ui()