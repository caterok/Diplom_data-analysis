import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Настройки страницы
st.set_page_config(
    page_title="ClasterTeach - Анализ и кластеризация данных",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация состояния сессии
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_preprocessed' not in st.session_state:
    st.session_state.data_preprocessed = False
if 'params_optimized' not in st.session_state:
    st.session_state.params_optimized = False
if 'clustering_done' not in st.session_state:
    st.session_state.clustering_done = False
if 'clustering_info' not in st.session_state:
    st.session_state.clustering_info = None

class DataAnalyzer:
    def __init__(self):
        self.data = None
        self.data_processed = None
        self.labels = None
        self.X_reduced = None
        self.best_params = None

    def load_data(self, file_path):
        """Загрузка данных из CSV файла"""
        try:
            self.data = pd.read_csv(file_path)


            categorical_cols = self.data.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                st.info(f"Удалены категориальные столбцы: {list(categorical_cols)}")
                self.data = self.data.drop(columns=categorical_cols)


            st.session_state.data_preprocessed = False
            st.session_state.params_optimized = False
            st.session_state.clustering_done = False
            st.session_state.clustering_info = None

            return True
        except Exception as e:
            st.error(f"Ошибка загрузки файла: {str(e)}")
            return False

    def analyze_missing_values(self):
        """Анализ пропущенных значений"""
        missing_values = self.data.isnull().sum()
        missing_percentage = (missing_values / len(self.data)) * 100

        missing_df = pd.DataFrame({
            'Колонка': missing_values.index,
            'Количество пропусков': missing_values.values,
            'Процент пропусков': missing_percentage.values
        })

        missing_df = missing_df[missing_df['Количество пропусков'] > 0]

        return missing_df

    def preprocess_data(self):
        """Предобработка данных"""
        # Заполняем пропуски медианой
        self.data_processed = self.data.fillna(self.data.median())

        # Стандартизация
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(self.data_processed)
        self.data_processed = pd.DataFrame(data_standardized, columns=self.data.columns)

        # Применяем t-SNE для визуализации
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        self.X_reduced = tsne.fit_transform(self.data_processed)

        return self.data_processed

    def optimize_dbscan(self):
        """Оптимизация параметров DBSCAN с помощью байесовской оптимизации"""

        # Сэмплируем данные для ускорения оптимизации
        sample_size = min(5000, len(self.X_reduced))
        if len(self.X_reduced) > sample_size:
            indices = np.random.choice(len(self.X_reduced), sample_size, replace=False)
            X_sample = self.X_reduced[indices]
        else:
            X_sample = self.X_reduced

        # Определяем пространство поиска параметров
        dimensions = [
            Real(0.1, 10.0, name='eps'),
            Integer(5, 50, name='min_samples')
        ]

        # Функция для оценки качества кластеризации
        @use_named_args(dimensions)
        def evaluate_clustering(**params):
            eps = params['eps']
            min_samples = params['min_samples']

            try:
                db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_sample)
                labels = db.labels_

                # Проверяем, что есть хотя бы 2 кластера (исключая шум)
                unique_labels = set(labels)
                n_clusters = len(unique_labels) - (1 if -1 in labels else 0)

                if n_clusters < 2 or n_clusters > 20:
                    return -1

                # Используем силуэтный коэффициент
                if len(unique_labels) > 1:
                    silhouette_avg = silhouette_score(X_sample, labels)
                    return -silhouette_avg  # Минимизируем отрицательный силуэт
                else:
                    return -1
            except:
                return -1

        # Запускаем байесовскую оптимизацию
        with st.spinner("Оптимизация параметров DBSCAN..."):
            try:
                res_gp = gp_minimize(
                    evaluate_clustering,
                    dimensions,
                    n_calls=30,
                    random_state=42,
                    verbose=False
                )

                self.best_params = {
                    'eps': res_gp.x[0],
                    'min_samples': res_gp.x[1]
                }

                return True
            except Exception as e:
                st.error(f"Ошибка оптимизации: {str(e)}")
                return False

    def apply_dbscan(self):
        """Применение DBSCAN с оптимальными параметрами"""
        if self.best_params is None:
            st.error("Сначала необходимо выполнить оптимизацию параметров!")
            return False

        try:
            db = DBSCAN(
                eps=self.best_params['eps'],
                min_samples=int(self.best_params['min_samples'])
            ).fit(self.X_reduced)

            self.labels = db.labels_

            # Анализ результатов кластеризации
            n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
            n_noise = list(self.labels).count(-1)

            # Вычисляем силуэтный коэффициент только если есть более 1 кластера
            if n_clusters > 1:
                silhouette = silhouette_score(self.X_reduced, self.labels)
            else:
                silhouette = 0

            clustering_info = {
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_percentage': (n_noise / len(self.labels)) * 100,
                'silhouette_score': silhouette
            }

            return clustering_info

        except Exception as e:
            st.error(f"Ошибка кластеризации: {str(e)}")
            return None

    def get_cluster_statistics(self):
        """Статистика по кластерам"""
        if self.labels is None:
            return None

        cluster_stats = []

        for cluster_id in np.unique(self.labels):
            if cluster_id == -1:
                cluster_name = "Шум"
            else:
                cluster_name = f"Кластер {cluster_id + 1}"

            indices = np.where(self.labels == cluster_id)[0]
            size = len(indices)

            if size > 0:
                cluster_data = self.data_processed.iloc[indices]
                stats = {
                    'Кластер': cluster_name,
                    'Размер': size,
                    'Процент': (size / len(self.labels)) * 100
                }

                # Добавляем средние значения для каждой колонки
                for col in self.data_processed.columns[:5]:  # Первые 5 колонок для читаемости
                    stats[f'{col}_среднее'] = cluster_data[col].mean()

                cluster_stats.append(stats)

        return pd.DataFrame(cluster_stats)

# Инициализация анализатора
@st.cache_resource
def init_analyzer():
    return DataAnalyzer()

def main():
    # Заголовок приложения
    st.markdown('<h1 class="main-header"> ClasterTeach - Анализ и кластеризация данных</h1>', unsafe_allow_html=True)

    # Инициализация анализатора
    analyzer = init_analyzer()

    # Сайдбар
    with st.sidebar:
        st.header(" Настройки")

        # Загрузка файла
        uploaded_file = st.file_uploader(
            "Загрузите CSV файл",
            type=['csv'],
            help="Загрузите файл данных в формате CSV"
        )

        if uploaded_file is not None:
            if st.button(" Загрузить и проанализировать данные", key="load_data"):
                with st.spinner("Загрузка данных..."):
                    if analyzer.load_data(uploaded_file):
                        st.session_state.data_loaded = True
                        st.session_state.file_name = uploaded_file.name
                        st.success(" Данные успешно загружены!")
                        st.rerun()

        st.markdown("---")

        # Демо данные
        if not st.session_state.data_loaded:
            if st.button("🚀 Использовать демо-данные", key="demo_data"):
                # Создаем демо данные
                np.random.seed(42)
                n_samples = 1000
                demo_data = pd.DataFrame({
                    'Признак_1': np.random.normal(0, 1, n_samples),
                    'Признак_2': np.random.normal(0, 1, n_samples),
                    'Признак_3': np.random.normal(0, 1, n_samples),
                    'Признак_4': np.random.normal(0, 1, n_samples),
                    'Признак_5': np.random.normal(0, 1, n_samples),
                    'Целевая_переменная': np.random.choice([0, 1], n_samples)
                })

                # Добавляем пропуски
                mask = np.random.random(demo_data.shape) < 0.05
                demo_data = demo_data.mask(mask)

                # Сохраняем во временный файл
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    demo_data.to_csv(f.name, index=False)
                    analyzer.load_data(f.name)

                st.session_state.data_loaded = True
                st.session_state.file_name = "demo_data.csv"
                st.rerun()

        # Статус
        if st.session_state.data_loaded:
            st.markdown("---")
            st.success(" Данные загружены")
            st.info(f"Файл: {st.session_state.file_name}")
            st.info(f"Размер: {analyzer.data.shape[0]} строк, {analyzer.data.shape[1]} столбцов")

            if st.session_state.data_preprocessed:
                st.success(" Данные обработаны")
            if st.session_state.params_optimized:
                st.success(" Параметры оптимизированы")
            if st.session_state.clustering_done:
                st.success(" Кластеризация выполнена")

    # Основное содержимое
    if not st.session_state.data_loaded:
        # Экран приветствия
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### Добро пожаловать в ClasterTeach!")
            st.markdown("""
            Для начала работы загрузите CSV файл через боковую панель.

            **Поддерживаемые возможности:**
            1. Анализ пропущенных значений
            2. Тепловые карты корреляций
            3. Диаграммы размаха
            4. Кластеризация DBSCAN
            5. Автоматическая оптимизация параметров
            """)

        return

    # Если данные загружены, показываем анализ
    st.markdown('<h2 class="section-header"> Обзор данных</h2>', unsafe_allow_html=True)

    # Основные метрики
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Количество строк", analyzer.data.shape[0])
    with col2:
        st.metric("Количество столбцов", analyzer.data.shape[1])
    with col3:
        missing_total = analyzer.data.isnull().sum().sum()
        st.metric("Всего пропусков", missing_total)
    with col4:
        missing_percent = (missing_total / (analyzer.data.shape[0] * analyzer.data.shape[1])) * 100
        st.metric("Процент пропусков", f"{missing_percent:.1f}%")

    # Вкладки для разных видов анализа
    tab1, tab2, tab3, tab4 = st.tabs([
        "Просмотр данных",
        "Анализ пропусков",
        "Визуализация",
        "Кластеризация"
    ])

    with tab1:
        st.markdown('<h3 class="section-header">Просмотр данных</h3>', unsafe_allow_html=True)

        st.subheader("Первые 10 строк данных")
        st.dataframe(analyzer.data.head(10), use_container_width=True)

        st.subheader("Статистическое описание")
        st.dataframe(analyzer.data.describe(), use_container_width=True)

        st.subheader("Информация о колонках")
        col_info = pd.DataFrame({
            'Колонка': analyzer.data.columns,
            'Тип данных': analyzer.data.dtypes.values,
            'Уникальных значений': [analyzer.data[col].nunique() for col in analyzer.data.columns]
        })
        st.dataframe(col_info, use_container_width=True)

    with tab2:
        st.markdown('<h3 class="section-header">Анализ пропущенных значений</h3>', unsafe_allow_html=True)

        missing_df = analyzer.analyze_missing_values()

        if len(missing_df) == 0:
            st.success(" В данных нет пропущенных значений!")
        else:
            # Таблица с пропусками
            st.subheader("Пропущенные значения по колонкам")
            st.dataframe(missing_df, use_container_width=True)

            # График пропусков
            st.subheader("Визуализация пропусков")

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Количество пропусков', 'Процент пропусков'),
                horizontal_spacing=0.2
            )

            fig.add_trace(
                go.Bar(
                    x=missing_df['Колонка'],
                    y=missing_df['Количество пропусков'],
                    name='Количество',
                    marker_color='indianred'
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Bar(
                    x=missing_df['Колонка'],
                    y=missing_df['Процент пропусков'],
                    name='Процент',
                    marker_color='lightcoral'
                ),
                row=1, col=2
            )

            fig.update_xaxes(tickangle=45, row=1, col=1)
            fig.update_xaxes(tickangle=45, row=1, col=2)
            fig.update_yaxes(title_text="Количество", row=1, col=1)
            fig.update_yaxes(title_text="Процент", row=1, col=2)
            fig.update_layout(height=500, showlegend=False)

            st.plotly_chart(fig, use_container_width=True)

            # Кнопка для заполнения пропусков
            if st.button(" Заполнить пропуски медианой и продолжить", key="fill_missing"):
                with st.spinner("Обработка данных..."):
                    analyzer.preprocess_data()
                    st.session_state.data_preprocessed = True
                    st.success(" Пропуски заполнены медианой!")
                    st.rerun()

    with tab3:
        st.markdown('<h3 class="section-header">Визуализация данных</h3>', unsafe_allow_html=True)

        if not st.session_state.data_preprocessed:
            st.warning(" Сначала заполните пропуски на вкладке 'Анализ пропусков'")
        else:
            # Тепловая карта корреляций
            st.subheader("Тепловая карта корреляций")

            corr_matrix = analyzer.data_processed.corr()

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                hoverongaps=False
            ))

            fig.update_layout(
                height=600,
                title="Корреляционная матрица",
                xaxis_tickangle=-45
            )

            st.plotly_chart(fig, use_container_width=True)

            # Диаграммы размаха
            st.subheader("Диаграммы размаха (Box Plots)")

            selected_cols = st.multiselect(
                "Выберите колонки для отображения:",
                options=analyzer.data_processed.columns.tolist(),
                default=analyzer.data_processed.columns[:5].tolist() if len(analyzer.data_processed.columns) > 5 else analyzer.data_processed.columns.tolist(),
                key="boxplot_cols"
            )

            if selected_cols:
                n_cols = min(3, len(selected_cols))
                n_rows = (len(selected_cols) + n_cols - 1) // n_cols

                fig = make_subplots(
                    rows=n_rows, cols=n_cols,
                    subplot_titles=selected_cols
                )

                for i, col in enumerate(selected_cols):
                    row = i // n_cols + 1
                    col_idx = i % n_cols + 1

                    fig.add_trace(
                        go.Box(
                            y=analyzer.data_processed[col],
                            name=col,
                            boxmean='sd'
                        ),
                        row=row, col=col_idx
                    )

                fig.update_layout(
                    height=300 * n_rows,
                    showlegend=False,
                    title_text="Диаграммы размаха по признакам"
                )

                st.plotly_chart(fig, use_container_width=True)

            # Гистограммы распределения
            st.subheader("Распределение признаков")

            selected_col = st.selectbox(
                "Выберите колонку для гистограммы:",
                options=analyzer.data_processed.columns.tolist(),
                key="histogram_col"
            )

            if selected_col:
                fig = px.histogram(
                    analyzer.data_processed,
                    x=selected_col,
                    nbins=50,
                    title=f"Распределение {selected_col}",
                    marginal="box"
                )

                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown('<h3 class="section-header">Кластеризация DBSCAN</h3>', unsafe_allow_html=True)

        if not st.session_state.data_preprocessed:
            st.warning(" Сначала заполните пропуски на вкладке 'Анализ пропусков'")
        else:
            # Область для оптимизации параметров
            if not st.session_state.params_optimized:
                st.info("### Шаг 1: Оптимизация параметров DBSCAN")
                st.markdown("""
                **Что будет сделано:**
                1. Используется байесовская оптимизация для поиска лучших параметров
                2. Оптимизируются параметры `eps` и `min_samples`
                3. Используется силуэтный коэффициент как метрика качества
                """)

                if st.button(" Начать оптимизацию параметров", type="primary", key="optimize_params"):
                    with st.spinner("Оптимизация параметров... Это может занять несколько секунд"):
                        if analyzer.optimize_dbscan():
                            st.session_state.params_optimized = True
                            st.success(" Параметры оптимизированы!")

                            # Показываем оптимальные параметры
                            st.markdown("### Оптимальные параметры:")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("eps (радиус)", f"{analyzer.best_params['eps']:.3f}")
                            with col2:
                                st.metric("min_samples (мин. образцов)", int(analyzer.best_params['min_samples']))

                            st.rerun()

            # Область для кластеризации
            elif st.session_state.params_optimized and not st.session_state.clustering_done:
                st.info("### Шаг 2: Применение кластеризации")
                st.markdown(f"""
                **Будут использованы следующие параметры:**
                - **eps**: {analyzer.best_params['eps']:.3f}
                - **min_samples**: {int(analyzer.best_params['min_samples'])}
                """)

                if st.button(" Выполнить кластеризацию", type="primary", key="apply_clustering"):
                    with st.spinner("Выполнение кластеризации..."):
                        clustering_info = analyzer.apply_dbscan()

                        if clustering_info:
                            st.session_state.clustering_done = True
                            st.session_state.clustering_info = clustering_info
                            st.success(" Кластеризация выполнена успешно!")
                            st.rerun()

            # Показ результатов кластеризации
            if st.session_state.clustering_done and st.session_state.clustering_info:
                info = st.session_state.clustering_info

                st.success("### Результаты кластеризации")

                # Метрики кластеризации
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Количество кластеров", info['n_clusters'])
                with col2:
                    st.metric("Точек шума", info['n_noise'])
                with col3:
                    st.metric("Процент шума", f"{info['noise_percentage']:.1f}%")
                with col4:
                    if info['n_clusters'] > 1:
                        st.metric("Силуэтный коэффициент", f"{info['silhouette_score']:.3f}")
                    else:
                        st.metric("Силуэтный коэффициент", "N/A")

                # Визуализация кластеров
                st.subheader("Визуализация кластеров (t-SNE)")

                # Создаем DataFrame для визуализации
                viz_df = pd.DataFrame({
                    'X': analyzer.X_reduced[:, 0],
                    'Y': analyzer.X_reduced[:, 1],
                    'Кластер': analyzer.labels
                })

                # Преобразуем метки кластеров в строки
                viz_df['Кластер'] = viz_df['Кластер'].apply(
                    lambda x: 'Шум' if x == -1 else f'Кластер {x + 1}'
                )

                fig = px.scatter(
                    viz_df,
                    x='X',
                    y='Y',
                    color='Кластер',
                    title='Визуализация кластеров DBSCAN',
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    hover_data={'Кластер': True}
                )

                fig.update_layout(
                    height=600,
                    legend_title_text='Кластеры'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Статистика по кластерам
                st.subheader("Статистика по кластерам")

                cluster_stats = analyzer.get_cluster_statistics()
                if cluster_stats is not None:
                    st.dataframe(cluster_stats, use_container_width=True)

                    # График распределения по кластерам
                    fig = px.bar(
                        cluster_stats,
                        x='Кластер',
                        y='Размер',
                        color='Кластер',
                        title='Распределение точек по кластерам',
                        text='Размер'
                    )

                    fig.update_layout(
                        height=500,
                        xaxis_tickangle=-45
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Сравнение кластеров по признакам
                st.subheader("Сравнение кластеров по признакам")

                selected_feature = st.selectbox(
                    "Выберите признак для сравнения:",
                    options=analyzer.data_processed.columns.tolist(),
                    key="cluster_feature"
                )

                if selected_feature:
                    # Создаем DataFrame для box plot
                    box_data = []
                    for cluster_id in np.unique(analyzer.labels):
                        indices = np.where(analyzer.labels == cluster_id)[0]
                        if len(indices) > 0:
                            cluster_name = 'Шум' if cluster_id == -1 else f'Кластер {cluster_id + 1}'
                            values = analyzer.data_processed[selected_feature].iloc[indices]

                            for val in values:
                                box_data.append({
                                    'Кластер': cluster_name,
                                    'Значение': val,
                                    'Признак': selected_feature
                                })

                    box_df = pd.DataFrame(box_data)

                    fig = px.box(
                        box_df,
                        x='Кластер',
                        y='Значение',
                        color='Кластер',
                        title=f'Распределение {selected_feature} по кластерам'
                    )

                    fig.update_layout(
                        height=500,
                        xaxis_tickangle=-45
                    )

                    st.plotly_chart(fig, use_container_width=True)


                st.subheader("Экспорт результатов")


                results_df = analyzer.data.copy()
                results_df['Кластер_DBSCAN'] = analyzer.labels
                results_df['Кластер_DBSCAN'] = results_df['Кластер_DBSCAN'].apply(
                    lambda x: 'Шум' if x == -1 else f'Кластер {x + 1}'
                )


                csv = results_df.to_csv(index=False).encode('utf-8')

                col1, col2, col3 = st.columns(3)
                with col2:
                    st.download_button(
                        label="📥 Скачать результаты кластеризации",
                        data=csv,
                        file_name="clustering_results.csv",
                        mime="text/csv",
                        key="download_results"
                    )


                if st.button("🔄 Начать новый анализ", key="reset_analysis"):

                    for key in list(st.session_state.keys()):
                        if key != 'file_name':
                            del st.session_state[key]


                    analyzer.load_data("temp_demo.csv" if st.session_state.file_name == "demo_data.csv" else None)
                    st.session_state.data_loaded = True
                    st.rerun()

if __name__ == "__main__":
    main()
