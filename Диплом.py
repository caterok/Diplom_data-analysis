from tkinter import *
from tkinter import filedialog
from tkinter import Menu
import pandas as pd
from tkinter import ttk
import matplotlib
import time
import threading
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import d2_absolute_error_score, silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import seaborn as sns
import tkinter as tk
from sklearn.model_selection import train_test_split  # Импортируем train_test_split
from skopt import gp_minimize  # Импортируем gp_minimize
from tkinter.ttk import *


class VisualizationManager:
    def __init__(self, master_master, master):
        self.master = master
        self.master_master = master_master
        self.fig = Figure(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, self.master)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        #self.master_master.create_window((0, 0), window = master, anchor = "nw")

    def destroy(self):

        #for item in self.canvas.get_tk_widget().find_all():
            #self.canvas.place_forget()
            #self.canvas.get_tk_widget().delete(item)
        self.master.destroy()
        self.fig.clear()
        #self.fig.place_forget()
        self.master.update_idletasks()
        self.master_master.update_idletasks()


    def create_missing(self, missing_values):
        ax = self.fig.add_subplot(111)
        missing_values.plot(ax=ax, kind='bar')
        ax.set_title('Гистограмма пропусков')
        ax.set_xlabel('Признаки')
        ax.set_ylabel('Количество пропусков')
        ax.tick_params(axis='x', rotation=90)
        self.fig.tight_layout()
        self.canvas.draw()
        self.master.update_idletasks()

    def create_hotcard(self, corr_matrix):
       ax = self.fig.add_subplot(111)
       im = ax.imshow(corr_matrix, cmap='coolwarm')
       ax.set_title('Тепловая карта')
       self.fig.colorbar(im)
       self.fig.tight_layout()
       self.canvas.draw()
       self.master.update_idletasks()

    def create_boxplot(self, data):
        ax = self.fig.add_subplot(111)
        data.plot(ax=ax, kind='box')
        self.fig.tight_layout()
        self.canvas.draw()
        self.master.update_idletasks()

    def create_boxplot1(self, data, n_clusters):
        ax = self.fig.add_subplot(111)

        ax.boxplot(data, labels=[f'Cluster {cluster+1}' for cluster in range(n_clusters)])

        self.fig.tight_layout()
        self.canvas.draw()
        self.master.update_idletasks()

    def create_plot_area(self, X_reduced, labels):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis')
        ax.set_title('t-SNE Visualization with DBSCAN Clusters')

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster Labels')

        self.canvas.draw()
        self.master.update_idletasks()

    def create_cluster_counts(self, cluster_counts):
        ax = self.fig.add_subplot(111)
        cluster_counts.plot(ax=ax, kind='bar')
        ax.set_title('Гистограмма кластеров')
        ax.set_xlabel('Количество элементов')
        ax.set_ylabel('Кластер')
        self.fig.tight_layout()
        self.canvas.draw()
        self.master.update_idletasks()

    def create_hotcard1(self, corr_matrix):
       ax = self.fig.add_subplot(111)
       im = ax.imshow(corr_matrix, cmap='coolwarm')
       ax.set_title('Тепловая карта')
       self.fig.colorbar(im)
       self.fig.tight_layout()
       self.canvas.draw()
       self.master.update_idletasks()

def browse_file():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        file_entry.delete(0, END)
        file_entry.insert(0, file_path)


def click_button():#диаграмма пропусков

    global passin
    global passin_frame
    global hot_card
    global hot_card_frame
    global visualization_manager
    global visualization_manager1

    data = pd.read_csv(file_path, delimiter=',')
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = data.drop(columns=categorical_cols)

    #пропуски в данных

    if not passin:
        passin = Canvas(tab1_frame, width= 1000, height=500)
        passin_frame = Frame(passin)
        passin.pack(fill = "both", expand = True)
        passin.create_window((0, 0), window = passin_frame, anchor = "nw")
        #passin.create_window((0, 0), window = tab1_frame, anchor = "nw")
        label2 = Label(passin_frame, text="Пропуски в наборе данных")
        label2.pack(padx=10, pady=10)

        visualization_manager = VisualizationManager(passin, passin_frame)
    else:
        visualization_manager.destroy()
        visualization_manager = None
        passin.destroy()

        passin = Canvas(tab1_frame, width= 1000, height=500)
        passin_frame = Frame(passin)
        passin.pack(fill = "both", expand = True)
        passin.create_window((0, 0), window = passin_frame, anchor = "nw")
        #passin.create_window((0, 0), window = tab1_frame, anchor = "nw")
        label2 = Label(passin_frame, text="Пропуски в наборе данных")
        label2.pack(padx=10, pady=10)

        visualization_manager = VisualizationManager(passin, passin_frame)



    #тепловая карта

    if not hot_card:
        hot_card = Canvas(tab1_frame, width= 1000, height=600)
        hot_card_frame = Frame(hot_card)
        hot_card.pack(fill = "both", expand = True)
        hot_card.create_window((0, 0), window = hot_card_frame, anchor = "nw")
        #hot_card.create_window((0, 0), window = tab1_frame, anchor = "nw")
        label2 = Label(hot_card_frame, text="Тепловая карта")
        label2.pack(padx=10, pady=10)

        visualization_manager1 = VisualizationManager(hot_card, hot_card_frame)
    else:
        visualization_manager1.destroy()
        visualization_manager1 = None
        hot_card.destroy()

        hot_card = Canvas(tab1_frame, width= 1000, height=600)
        hot_card_frame = Frame(hot_card)
        hot_card.pack(fill = "both", expand = True)
        hot_card.create_window((0, 0), window = hot_card_frame, anchor = "nw")
        #hot_card.create_window((0, 0), window = tab1_frame, anchor = "nw")
        label2 = Label(hot_card_frame, text="Тепловая карта")
        label2.pack(padx=10, pady=10)

        visualization_manager1 = VisualizationManager(hot_card, hot_card_frame)



    missing_values = data.isnull().sum()

    visualization_manager.create_missing(missing_values)

    corr_matrix = data.corr()

    visualization_manager1.create_hotcard(corr_matrix)

    global label_data
    global manager_data
    global canvas_span_data
    global frame_span_data

    if not manager_data:
        for i, column in enumerate(data.columns):
            #canvas_span = Canvas(tab1_frame, width= 1000, height=700)
            canvas_span_data.append(Canvas(tab1_frame, width= 1000, height=600))
            #frame_span = Frame(canvas_span)
            frame_span_data.append(Frame(canvas_span_data[i]))
            canvas_span_data[i].create_window((0, 0), window = frame_span_data[i], anchor = "nw")

            #label2 = Label(frame_span_data, text=f'Диаграмма размаха {column}')
            label_data.append(Label(frame_span_data[i], text=f'Диаграмма размаха {column}'))
            label_data[i].pack(padx=10, pady=10)

            canvas_span_data[i].pack(fill = "both", expand = True)




            manager = VisualizationManager(canvas_span_data[i], frame_span_data[i])

            manager.create_boxplot(data[column])
            manager_data.append(manager)
    else:
        for i in manager_data:
            i.destroy()
            i = None
        manager_data.clear()

        for i in label_data:
            i.destroy()
            i = None
        label_data.clear()

        for i in canvas_span_data:
            i.destroy()
            i = None
        canvas_span_data.clear()

        for i in frame_span_data:
            i.destroy()
            i = None
        frame_span_data.clear()

        for i, column in enumerate(data.columns):
            #canvas_span = Canvas(tab1_frame, width= 1000, height=700)
            canvas_span_data.append(Canvas(tab1_frame, width= 1000, height=600))
            #frame_span = Frame(canvas_span)
            frame_span_data.append(Frame(canvas_span_data[i]))
            canvas_span_data[i].create_window((0, 0), window = frame_span_data[i], anchor = "nw")

            #label2 = Label(frame_span_data, text=f'Диаграмма размаха {column}')
            label_data.append(Label(frame_span_data[i], text=f'Диаграмма размаха {column}'))
            label_data[i].pack(padx=10, pady=10)

            canvas_span_data[i].pack(fill = "both", expand = True)




            manager = VisualizationManager(canvas_span_data[i], frame_span_data[i])

            manager.create_boxplot(data[column])
            manager_data.append(manager)




def click_button2():




    data = pd.read_csv(file_path, delimiter=',')
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = data.drop(columns=categorical_cols)

    data = data.fillna(data.median())

    scaler = StandardScaler()
    data_standart = scaler.fit_transform(data)
    data_standart = pd.DataFrame(data_standart, columns=data.columns)\

    # Apply t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=0)
    X_reduced = tsne.fit_transform(data_standart)

    # Определяем пространство поиска параметров
    dimensions = [Real(0.1, 10, name='eps'),
                  Integer(10, 100, name='min_samples')]

    # Разбиваем данные на обучающую и тестовую выборки
    X_train, X_test, _, _ = train_test_split(X_reduced, data_standart.index, test_size=0.2, random_state=42)

    # Сэмплируем обучающую выборку
    sample_size = int(len(X_train) * 0.2)  # 10% от обучающей выборки
    X_reduced_sample, _ = X_train[:sample_size], data_standart.iloc[:sample_size].index

    # Функция для оценки качества кластеризации
    @use_named_args(dimensions)
    def evaluate_clustering(**params):
        eps, min_samples = params['eps'], params['min_samples']
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_reduced_sample)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters == 0:
            return -1  # Плохое качество кластеризации

        silhouette_avg = silhouette_score(X_reduced_sample, labels)
        return silhouette_avg

    # Запускаем байесовскую оптимизацию
    res_gp = gp_minimize(evaluate_clustering, dimensions, n_calls=100, random_state=42)

    # Получаем оптимальные параметры
    best_params = res_gp.x
    best_eps, best_min_samples = best_params

    #print(f"Оптимальные параметры: eps={best_eps}, min_samples={int(best_min_samples)}")  # Раскомментируйте, если хотите видеть оптимальные параметры

    # Применяем оптимальные параметры к полному набору данных
    db = DBSCAN(eps=best_eps, min_samples=int(best_min_samples)).fit(X_reduced)
    labels = db.labels_

    global plot_area
    global plot_area_frame
    global visualization_manager3

    if not plot_area:
        plot_area = Canvas(tab2_frame, width= 1000, height=500)
        plot_area_frame = Frame(plot_area)
        plot_area.pack(fill = "both", expand = True)
        plot_area.create_window((0, 0), window = plot_area_frame, anchor = "nw")

        visualization_manager3 = VisualizationManager(plot_area, plot_area_frame)
    else:
        visualization_manager3.destroy()
        visualization_manager3 = None
        plot_area.destroy()

        plot_area = Canvas(tab1_frame, width= 1000, height=500)
        plot_area_frame = Frame(plot_area)
        plot_area.pack(fill = "both", expand = True)
        plot_area.create_window((0, 0), window = plot_area_frame, anchor = "nw")

        visualization_manager3 = VisualizationManager(plot_area, plot_area_frame)

    visualization_manager3.create_plot_area(X_reduced, labels)

    cluster_counts = pd.Series(labels).value_counts().sort_index()

    global cluster_counts_canvas
    global cluster_counts_frame
    global visualization_manager4

    if not cluster_counts_canvas:
        cluster_counts_canvas = Canvas(tab2_frame, width= 1000, height=500)
        cluster_counts_frame = Frame(cluster_counts_canvas)

        label2 = Label(cluster_counts_frame, text="Распределение данных по кластерам")
        label2.pack(padx=10, pady=10)

        cluster_counts_canvas.pack(fill = "both", expand = True)
        cluster_counts_canvas.create_window((0, 0), window = cluster_counts_frame, anchor = "nw")


        visualization_manager4 = VisualizationManager(cluster_counts_canvas, cluster_counts_frame)
    else:
        visualization_manager4.destroy()
        visualization_manager4 = None
        cluster_counts_canvas.destroy()

        cluster_counts_canvas = Canvas(tab2_frame, width= 1000, height=500)
        cluster_counts_frame = Frame(cluster_counts_canvas)

        label2 = Label(cluster_counts_frame, text="Распределение данных по кластерам")
        label2.pack(padx=10, pady=10)

        cluster_counts_canvas.pack(fill = "both", expand = True)
        cluster_counts_canvas.create_window((0, 0), window = cluster_counts_frame, anchor = "nw")

        visualization_manager4 = VisualizationManager(cluster_counts_canvas, cluster_counts_frame)

    visualization_manager4.create_cluster_counts(cluster_counts)

    corr_matrix1 = data_standart.corr()

    global corr_canvas
    global corr_frame
    global visualization_manager5

    if not corr_canvas:
        corr_canvas = Canvas(tab2_frame, width= 1000, height=600)
        corr_frame = Frame(corr_canvas)

        label2 = Label(corr_frame, text="Тепловая карта")
        label2.pack(padx=10, pady=10)

        corr_canvas.pack(fill = "both", expand = True)
        corr_canvas.create_window((0, 0), window = corr_frame, anchor = "nw")


        visualization_manager5 = VisualizationManager(corr_canvas, corr_frame)
    else:
        visualization_manager5.destroy()
        visualization_manager5 = None
        corr_canvas.destroy()

        corr_canvas = Canvas(tab2_frame, width= 1000, height=500)
        corr_frame = Frame(corr_canvas)

        label2 = Label(corr_frame, text="Тепловая карта")
        label2.pack(padx=10, pady=10)

        corr_canvas.pack(fill = "both", expand = True)
        corr_canvas.create_window((0, 0), window = corr_frame, anchor = "nw")


        visualization_manager5 = VisualizationManager(corr_canvas, corr_frame)

    visualization_manager5.create_hotcard(corr_matrix1)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Prepare data for each cluster and feature
    cluster_data = []
    for feature in data.columns:
        feature_data = []
        for cluster in range(n_clusters):
            cluster_indices = np.where(labels == cluster)[0]
            cluster_values = data[feature].iloc[cluster_indices]
            feature_data.append(cluster_values)
        cluster_data.append(feature_data)


    global cluster_label_data
    global cluster_manager_data
    global cluster_canvas_span_data
    global cluster_frame_span_data

    if not cluster_manager_data:
        for i, column in enumerate(cluster_data):
            cluster_canvas_span = Canvas(tab2_frame, width= 1000, height=600)
            cluster_canvas_span_data.append(cluster_canvas_span)
            cluster_frame_span = Frame(cluster_canvas_span)
            cluster_frame_span_data.append(cluster_frame_span)

            label = Label(cluster_frame_span, text=f'Диаграмма размаха {data.columns[i]}')
            cluster_label_data.append(label)
            cluster_label_data[i].pack(padx=10, pady=10)

            cluster_canvas_span_data[i].pack(fill = "both", expand = True)
            cluster_canvas_span_data[i].create_window((0, 0), window = cluster_frame_span_data[i], anchor = "nw")


            manager = VisualizationManager(cluster_canvas_span, cluster_frame_span)

            manager.create_boxplot1(column, n_clusters)
            cluster_manager_data.append(manager)
    else:
        for i in cluster_manager_data:
            i.destroy()
            i = None
        cluster_manager_data.clear()

        for i in cluster_label_data:
            i.destroy()
            i = None
        cluster_label_data.clear()

        for i in cluster_canvas_span_data:
            i.destroy()
            i = None
        cluster_canvas_span_data.clear()

        for i in cluster_frame_span_data:
            i.destroy()
            i = None
        cluster_frame_span_data.clear()

        for i, column in enumerate(cluster_data):
            cluster_canvas_span = Canvas(tab2_frame, width= 1000, height=600)
            cluster_canvas_span_data.append(cluster_canvas_span)
            cluster_frame_span = Frame(cluster_canvas_span)
            cluster_frame_span_data.append(cluster_frame_span)

            label = Label(cluster_frame_span, text=f'Диаграмма размаха {data.columns[i]}')
            cluster_label_data.append(label)
            cluster_label_data[i].pack(padx=10, pady=10)

            cluster_canvas_span_data[i].pack(fill = "both", expand = True)
            cluster_canvas_span_data[i].create_window((0, 0), window = cluster_frame_span_data[i], anchor = "nw")


            manager = VisualizationManager(cluster_canvas_span, cluster_frame_span)

            manager.create_boxplot1(column, n_clusters)
            cluster_manager_data.append(manager)



    #if not cluster_manager_data:

    #    for i, column in enumerate(cluster_data):
    #        cluster_canvas_span = Canvas(tab1_frame, width= 1000, height=700)
    #        cluster_canvas_span_data.append(canvas_span)
    #        cluster_frame_span = Frame(canvas_span)
    #        cluster_frame_span_data.append(frame_span)
    #        cluster_canvas_span_data[i].pack(fill = "both", expand = True)
    #        cluster_canvas_span_data[i].create_window((0, 0), window = frame_span_data[i], anchor = "nw")
    #        label = Label(tab2_frame, text=f'Диаграмма размаха {data.columns[i]}')
    #        label.pack(padx=10, pady=10)
    #        cluster_label_data.append(label)

    #        manager = VisualizationManager(tab2_canvas, tab2_frame)

    #        manager.create_boxplot1(column, n_clusters)
    #        cluster_manager_data.append(manager)

def potoki():
    global pb1
    pb1.start()
    secondary_thread = threading.Thread(target=click_button2)
    secondary_thread.start()
    pb1.stop()




def canvas_conf(canvas):
    w = window.winfo_width()
    h = window.winfo_height()
    canvas.configure(scrollregion=canvas.bbox("all"), width = max((w-100),500), height = max((h-100),500))


window = Tk()
window.title("ClasterTeach")

window.geometry('1024x768')
#window.wm_maxsize(width=1920, height=1080)
menu = Menu(window)
new_item = Menu(menu, tearoff=0)
new_item.add_command(label='Новый', command=browse_file)
new_item.add_separator()
new_item.add_command(label='Изменить', command=browse_file)
menu.add_cascade(label='Файл', menu=new_item)
window.config(menu=menu)

passin_frame = None
passin = None
visualization_manager = None

hot_card_frame = None
hot_card = None
visualization_manager1= None

visualization_manager3 = None
visualization_manager4 = None
visualization_manager5 = None

plot_area = None

label_data = []
manager_data = []
canvas_span_data = []
frame_span_data = []

corr_canvas = None
corr_frame = None

cluster_counts_canvas = None
cluster_counts_frame = None

plot_area = None
plot_area_frame = None

cluster_label_data = []
cluster_manager_data = []
cluster_canvas_span_data = []
cluster_frame_span_data = []

# Текстовое поле для пути файла
file_entry = Entry(window, width=50)
file_entry.grid(sticky='e',row=0, column=1, padx=10, pady=10)
#file_entry.pack(anchor='nw', padx=10, pady=10, side="top")

# Кнопка "Обзор"
browse_button = Button(window, text="Обзор", command=browse_file)
browse_button.grid(sticky='e',row=0, column=0, padx=10, pady=10)
#browse_button.pack(anchor='nw', padx=350, pady=0, side="top")

pb1 = Progressbar(window, orient=HORIZONTAL, length=300, mode='indeterminate')
pb1.grid(sticky='e', row=0, column=2, padx=10, pady=10)
#pb1.pack(anchor='nw', padx=500, pady=10, side="top")

# Создание вкладок
notebook = Notebook(window)
notebook.grid(row=1, column=0, columnspan=5, padx=10, pady=10, sticky="nsew")
#notebook.pack(anchor='e', padx=10, pady=20, side="left")

# Вкладка 1
tab1 = Frame(notebook)
tab1.pack(fill = "both", expand = True)

tab1_canvas = tk.Canvas(tab1)
tab1_frame = tk.Frame(tab1_canvas)
yscrollbar = tk.Scrollbar(tab1, orient = "vertical", command = tab1_canvas.yview)

tab1_canvas.configure(yscrollcommand = yscrollbar.set)

yscrollbar.pack(side = "right", fill = "y")
tab1_canvas.pack(side="left", fill = "both", expand = True)
tab1_canvas.create_window((0, 0), window = tab1_frame, anchor = "nw")

tab1_frame.bind("<Configure>", lambda event, canvas=tab1_canvas:canvas_conf(tab1_canvas))

label1 = Label(tab1_frame)
label1.pack(padx=10, pady=10)
notebook.add(tab1, text="Оригинальные данные")




label_data = []
manager_data = []

build_button = Button(tab1_frame, text="Построй", command=click_button)
build_button.pack(padx=50, pady=10)







# Вкладка 2
tab2 = Frame(notebook)

tab2_canvas = tk.Canvas(tab2, width= 500, height=500)
tab2_frame = tk.Frame(tab2_canvas)
yscrollbar = tk.Scrollbar(tab2, orient = "vertical", command = tab2_canvas.yview)

tab2_canvas.configure(yscrollcommand = yscrollbar.set)

yscrollbar.pack(side = "right", fill = "y")
tab2_canvas.pack(side="left", fill = "both", expand = True)
tab2_canvas.create_window((0, 0), window = tab2_frame, anchor = "nw")

tab2_frame.bind("<Configure>", lambda event, canvas=tab2_canvas:canvas_conf(tab2_canvas))

label2 = Label(tab2)
label2.pack(padx=10, pady=10)
notebook.add(tab2, text="Кластеризация данных")

label2 = Label(tab2_frame, text="Кластеризация набора данных, после заполнения пропусков медианой столбца")
label2.pack(padx=10, pady=10)

build_button = Button(tab2_frame, text="Построй", command=potoki)
build_button.pack(padx=10, pady=10)

window.mainloop()
