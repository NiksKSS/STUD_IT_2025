import os
import io
import json
import pickle
import time
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Page config
st.set_page_config(page_title="Предскажем будущее атомов... 🔬", layout="wide")

# Dark blue theme CSS (strong contrast)
st.markdown(
    """
    <style>
    .stApp { background: #061428; color: #e8f2ff; }
    .block-container {
        background: linear-gradient(180deg, #071833 0%, #0b2746 100%);
        border-radius: 12px;
        padding: 28px 32px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.6);
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div { color: #e6f1ff !important; }
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg,#061428 0%, #07192b 100%) !important;
        color: #dbeeff !important;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #0b3b76 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #2b66b0 !important;
        padding: 6px 12px !important;
    }
    .stButton>button:hover, .stDownloadButton>button:hover { background-color: #154f9e !important; }
    input, textarea, select, div[data-baseweb="select"] { background-color: #08203a !important; color: #e6f1ff !important; }
    .stDataFrame table { background-color: #071733 !important; color: #e6f1ff !important; }
    .streamlit-expanderHeader { background-color: rgba(15,54,106,0.6) !important; color: #eaf4ff !important; }
    .streamlit-expanderContent { background-color: #071f35 !important; color: #e6f1ff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Preprocessing utils --------------------
def to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x if isinstance(x, list) else []

def preprocess_data(df):
    df = df.copy()
    for col in ['positions', 'atomicNumbers', 'elements', 'gradient']:
        if col in df.columns:
            df[f'{col}_len'] = df[col].apply(lambda x: len(to_list(x)))
    if 'positions' in df.columns:
        def min_max_dist(pos):
            arr = np.array(to_list(pos))
            if arr.size == 0:
                return 0.0, 0.0
            dists = np.linalg.norm(arr, axis=1)
            return np.min(dists), np.max(dists)
        df[['min_dist_center', 'max_dist_center']] = df['positions'].apply(lambda x: pd.Series(min_max_dist(x)))
        def avg_pairwise_dist(pos):
            arr = np.array(to_list(pos))
            if len(arr) < 2:
                return 0.0
            return pdist(arr).mean()
        df['avg_dist_atoms'] = df['positions'].apply(avg_pairwise_dist)
    elements_of_interest = ['H', 'Li', 'C', 'O', 'F', 'P']
    if 'elements' in df.columns:
        for el in elements_of_interest:
            df[f'count_{el}'] = df['elements'].apply(lambda x: to_list(x).count(el))
    else:
        for el in elements_of_interest:
            df[f'count_{el}'] = 0
    def count_adjacent_bonds(elems, a, b):
        el_list = to_list(elems)
        count = 0
        for i in range(len(el_list) - 1):
            if (el_list[i] == a and el_list[i + 1] == b) or (el_list[i] == b and el_list[i + 1] == a):
                count += 1
        return count
    if 'elements' in df.columns:
        df['CΞO'] = df['elements'].apply(lambda x: count_adjacent_bonds(x, 'C', 'O'))
        df['CΞH'] = df['elements'].apply(lambda x: count_adjacent_bonds(x, 'C', 'H'))
        df['CΞC'] = df['elements'].apply(lambda x: count_adjacent_bonds(x, 'C', 'C'))
    else:
        df['CΞO'] = df['CΞH'] = df['CΞC'] = 0
    if 'gradient' in df.columns:
        def max_grad_norm(grad):
            arr = np.array(to_list(grad))
            if arr.size == 0:
                return 0.0
            norms = np.linalg.norm(arr, axis=1)
            return np.max(norms)
        df['max_gradient'] = df['gradient'].apply(max_grad_norm)
    else:
        df['max_gradient'] = 0
    if 'dipoleMoment' in df.columns:
        def dipole_norm(dvec):
            d = np.array(to_list(dvec))
            return np.linalg.norm(d) if d.size == 3 else 0.0
        df['dipole_magnitude'] = df['dipoleMoment'].apply(dipole_norm)
    else:
        df['dipole_magnitude'] = 0
    cols_to_drop = ['positions', 'atomicNumbers', 'elements', 'gradient', 'dipoleMoment']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    return df

def prepare_data(input_data: pd.DataFrame) -> (pd.DataFrame, dict):
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("input_data должен быть pd.DataFrame")
    df = input_data.copy()
    if 'positions' not in df.columns:
        raise ValueError("Столбец 'positions' отсутствует в наборе")
    reasons = {'no_positions': 0, 'totalEnergy': 0, 'charge': 0}
    start_n = df.shape[0]
    mask_positions = df['positions'].notna()
    reasons['no_positions'] = (~mask_positions).sum()
    df = df[mask_positions].copy()
    if 'totalEnergy' in df.columns:
        mask_e = df['totalEnergy'] < 1000
        reasons['totalEnergy'] = (~mask_e).sum()
        df = df[mask_e]
    if 'charge' in df.columns:
        mask_c = df['charge'] <= 10
        reasons['charge'] = (~mask_c).sum()
        df = df[mask_c]
    df = preprocess_data(df)
    redundant = ['atomicNumbers_len', 'elements_len', 'gradient_len', 'dipoleMoment_len', 'multiplicity']
    df.drop(columns=[c for c in redundant if c in df.columns], inplace=True)
    df = df.select_dtypes(include=[np.number])
    df.fillna(0.0, inplace=True)
    end_n = df.shape[0]
    reasons['total_removed'] = start_n - end_n
    return df, reasons

def safe_preview_dataframe(df: pd.DataFrame, n=5, max_chars=250):
    df_head = df.head(n).copy()
    heavy_examples = {}
    for col in df_head.columns:
        if df_head[col].dtype == 'object':
            def to_str(x):
                try:
                    if x is None:
                        s = "None"
                    else:
                        s = json.dumps(x, default=str, ensure_ascii=False)
                except Exception:
                    try:
                        s = str(x)
                    except Exception:
                        s = "<unserializable>"
                if len(s) > max_chars:
                    s = s[:max_chars] + "..."
                return s
            examples = [to_str(v) for v in df_head[col].tolist()]
            heavy_examples[col] = examples
            df_head[col] = df_head[col].apply(to_str)
    df_head.columns = [str(c) for c in df_head.columns]
    return df_head, heavy_examples

# session state init
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'processed' not in st.session_state:
    st.session_state.processed = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'hide_easter' not in st.session_state:
    st.session_state.hide_easter = False

# Helper: show author info safely (modal if available, else expander)
def show_author_info():
    content = {
        "Автор": "Nika Denisenko",
        "Мероприятие": "СТУД-ИТ 2025",
        "Организация разработки": "РТУ МИРЭА",
        "Версия ПО": "P0-1 — Альфа"
    }
    if hasattr(st, "modal"):
        try:
            with st.modal("Информация об авторе"):
                st.markdown("### Об авторе и проекте")
                for k, v in content.items():
                    st.markdown(f"**{k}:** {v}")
                return
        except Exception:
            pass
    # fallback
    with st.expander("Информация об авторе"):
        st.markdown("### Об авторе и проекте")
        for k, v in content.items():
            st.markdown(f"**{k}:** {v}")

# Sidebar
with st.sidebar:
    if os.path.exists('logo.png'):
        st.image('logo.png', width=140)
    else:
        st.image('https://raw.githubusercontent.com/ageron/handson-ml2/master/images/hands_on_ml.png', width=140)
    st.markdown("## Управление")
    st.markdown("---")
    local_files = [f for f in os.listdir('.') if f.lower().endswith(('.pickle', '.pkl', '.csv'))]
    for fname in local_files:
        if fname not in st.session_state.datasets:
            try:
                if fname.lower().endswith('.csv'):
                    df = pd.read_csv(fname)
                else:
                    df = pd.read_pickle(fname)
                st.session_state.datasets[fname] = df
            except Exception:
                pass
    dataset_names = ['--'] + list(st.session_state.datasets.keys())
    chosen_dataset = st.selectbox("Выбрать набор данных", options=dataset_names, index=0)
    st.markdown("---")
    uploaded = st.file_uploader("Загрузить .pickle / .pkl / .csv", type=['pickle', 'pkl', 'csv'], accept_multiple_files=False)
    if uploaded is not None:
        try:
            uploaded.seek(0)
            if uploaded.name.lower().endswith('.csv'):
                df_up = pd.read_csv(uploaded)
            else:
                uploaded.seek(0)
                df_up = pickle.load(uploaded)
            name = uploaded.name
            st.session_state.datasets[name] = df_up
            st.success(f"Файл '{name}' загружен в сессию")
            chosen_dataset = name
        except Exception as e:
            st.error(f"Не удалось загрузить файл: {e}")
    st.markdown("---")
    # ONLY models from disk (no builtins)
    models_on_disk = []
    if os.path.exists('models'):
        models_on_disk = [f.replace('.pkl', '') for f in os.listdir('models') if f.endswith('.pkl')]
    models_all = ['--'] + models_on_disk
    chosen_model = st.selectbox("Выбрать модель", options=models_all, index=0)
    st.markdown("---")
    st.write("Инструкция: \n1) Выбери/загрузи датасет → 2) Нажми 'Предобработать' → 3) Выбери модель → 4) 'Создать предсказания'.")
    st.markdown("---")
    if st.button('Об авторе'):
        show_author_info()
# Проверка наличия файла инструкции
    instr_pdf = 'Инструкция пользователя.pdf'

    if os.path.exists(instr_pdf):
        with open(instr_pdf, "rb") as f:
            pdf_bytes = f.read()
        
        st.download_button(
            label="Открыть инструкцию пользователя (PDF)",
            data=pdf_bytes,
            file_name=instr_pdf,
            mime="application/pdf"
        )
    else:
        st.warning("Файл инструкции 'Инструкция пользователя.pdf' не найден.")


# Main
st.title("Предскажем будущее атомов...")
col1, col2 = st.columns([2, 3])

with col1:
    st.header("Датасет")
    st.write("Выбранный набор:", chosen_dataset)
    if chosen_dataset != '--' and chosen_dataset in st.session_state.datasets:
        raw_df = st.session_state.datasets[chosen_dataset]
        try:
            preview_df, heavy_examples = safe_preview_dataframe(raw_df, n=5)
            st.dataframe(preview_df)
            if heavy_examples:
                st.markdown("**Вложенные колонки (примеры)**")
                for col, examples in heavy_examples.items():
                    with st.expander(f"{col} — показать примеры"):
                        for i, ex in enumerate(examples):
                            st.write(f"{i}: ", ex)
        except Exception as e:
            st.error(f"Не удалось отобразить preview: {e}")
            try:
                st.write(str(raw_df.head(5)))
            except Exception:
                st.write("Невозможно показать preview — проверь датасет локально.")
    else:
        st.info("Выберите датасет или загрузите файл в боковой панели.")
    can_preprocess = (chosen_dataset != '--') and (chosen_dataset in st.session_state.datasets)
    if st.button("Предобработать набор", disabled=not can_preprocess):
        try:
            raw = st.session_state.datasets[chosen_dataset]
            raw_count = raw.shape[0]
            if os.path.exists('loading.gif'):
                st.image('loading.gif', caption='Идёт предобработка — подождите...', use_column_width=False)
                time.sleep(4)
            else:
                with st.spinner('Идёт предобработка (симуляция 4 сек)...'):
                    time.sleep(4)
            processed, reasons = prepare_data(raw)
            processed_count = processed.shape[0]
            removed_count = reasons.get('total_removed', raw_count - processed_count)
            removed_pct = (removed_count / raw_count * 100) if raw_count > 0 else 0.0
            st.session_state.processed[chosen_dataset] = processed
            st.success("Данные предобработаны")
            st.info(f"Исходных строк: {raw_count} — После предобработки: {processed_count} — Удалено: {removed_count} ({removed_pct:.2f}%)")
            with st.expander('Детали очистки'):
                st.write(reasons)
            if removed_count > 0:
                try:
                    removed_idx = raw.index.difference(processed.index)
                    with st.expander("Показать примеры удалённых строк (до 5)"):
                        st.write(raw.loc[removed_idx].head(5))
                except Exception:
                    st.write("Нельзя показать удалённые строки (сложные типы, вложенные списки).")
        except Exception as e:
            st.error(f"Ошибка предобработки: {e}")
    if chosen_dataset in st.session_state.processed:
        st.markdown("**Признаки после предобработки:**")
        st.write(list(st.session_state.processed[chosen_dataset].columns))

with col2:
    st.header("Модель и предсказания")
    st.write("Выбранная модель:", chosen_model)
    loaded_model = None
    if chosen_model != '--':
        model_path = os.path.join('models', chosen_model + '.pkl')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    loaded_model = pickle.load(f)
                st.success(f"Модель '{chosen_model}' загружена")
            except Exception as e:
                st.error(f"Не удалось загрузить модель: {e}")
    predict_ready = (chosen_dataset in st.session_state.processed) and (chosen_model != '--')
    if st.button("Создать предсказания", disabled=not predict_ready):
        try:
            X = st.session_state.processed[chosen_dataset]
            # Use a fallback simple model if no saved model selected
            if loaded_model is None:
                m = LinearRegression()
                m.fit(X, np.zeros(X.shape[0]))
                preds = m.predict(X)
            else:
                preds = loaded_model.predict(X)
            preds_series = pd.Series(preds, index=X.index, name='prediction')
            st.session_state.predictions[(chosen_dataset, chosen_model)] = preds_series
            st.success("Предсказания созданы")
        except Exception as e:
            st.error(f"Ошибка при создании предсказаний: {e}")

    key = (chosen_dataset, chosen_model)
    if key in st.session_state.predictions:
        preds = st.session_state.predictions[key]
        df_show = pd.DataFrame({'prediction': preds})
        st.subheader("Таблица предсказаний")
        st.dataframe(df_show.head(200))
        csv = df_show.to_csv(index=True).encode('utf-8')
        st.download_button("Скачать предсказания (CSV)", data=csv, file_name=f"preds_{chosen_dataset}_{chosen_model}.csv")
        st.subheader("Просмотр значения по индексу")
        idx_options = ['--'] + [str(i) for i in df_show.index[:500]]
        sel_index = st.selectbox("Выбрать индекс", options=idx_options)
        if sel_index != '--':
            try:
                sel_index_int = int(sel_index)
                val = preds.loc[sel_index_int]
                uncertainty = abs(0.05 * val) + 0.01
                st.write(f"Предсказание: {val:.6f} ± {uncertainty:.6f}")
            except Exception as e:
                st.error(f"Не удалось получить индекс: {e}")

# Graphs (white background for plots, sanitize data)
# Graphs
st.markdown("---")
st.header("Графики и корреляции")

key = (chosen_dataset, chosen_model)
if key in st.session_state.predictions:
    X = st.session_state.processed[chosen_dataset]
    preds = st.session_state.predictions[key]

    st.subheader("График предсказаний")

    feature_list = ['index'] + list(X.columns)
    feature = st.selectbox("Признак для оси X", options=feature_list)
    plot_type = st.selectbox(
        "Тип графика",
        options=['Линейный', 'Точечный', 'Гистограмма', 'Скрипичная диаграмма']
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.tick_params(colors='black', labelcolor='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.grid(True, linestyle='--', alpha=0.4)

    # Данные для графика
    x = np.array(preds.index) if feature == 'index' else np.array(X[feature])
    y = np.array(preds)

    # Фильтрация NaN/Inf
    finite_mask = np.isfinite(y)
    if feature != 'index':
        finite_mask &= np.isfinite(x)
    x_clean = x[finite_mask]
    y_clean = y[finite_mask]

    if len(y_clean) == 0:
        st.warning("Нет валидных данных для построения графика (NaN/Inf).")
    else:
        if plot_type == 'Линейный':
            ax.plot(x_clean, y_clean, color='#0b3b76', linewidth=1.2)  # тонкая линия
            ax.set_xlabel(feature)
            ax.set_ylabel('prediction')
        elif plot_type == 'Точечный':
            ax.scatter(x_clean, y_clean, s=20, color='#154f9e', alpha=0.6)
            ax.set_xlabel(feature)
            ax.set_ylabel('prediction')
        elif plot_type == 'Гистограмма':
            ax.hist(y_clean, bins=40, color='#2b6fd6', alpha=0.7)
            ax.set_xlabel('prediction')
            ax.set_ylabel('count')
        elif plot_type == 'Скрипичная диаграмма':
            sns.violinplot(y=y_clean, ax=ax, inner='quartile', color='#2b6fd6')
            ax.set_xlabel(feature)
            ax.set_ylabel('prediction')

        st.pyplot(fig)

    # Корреляционная матрица
# Корреляционная матрица
    st.subheader("Матрица корреляций признаков")
    if not X.empty:
        corr = X.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))  # увеличили размер
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax_corr,
            cbar=True,
            annot_kws={"size": 8}  # уменьшили шрифт чисел
        )
        ax_corr.set_title("Корреляции между признаками", fontsize=12)
        st.pyplot(fig_corr)
    else:
        st.info("Нет данных для построения корреляционной матрицы.")

    


# Easter
st.markdown("---")
if not st.session_state.get('hide_easter', False):
    if st.button('Показать пасхалку'):
        st.balloons()
        st.snow()

st.caption("Версия - 1. Автор: Nika Denisenko; СТУД-ИТ 2025; Разработано в РТУ МИРЭА")


