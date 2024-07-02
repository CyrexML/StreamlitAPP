import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow import keras
from xgboost import XGBClassifier

# Путь к данным
data_path_raw = "D:/Projektiki/predykcja_rezerwacji/hotel_booking.csv"
data_path_cleaned = "D:/Projektiki/predykcja_rezerwacji/hotel_booking1.csv"

# Загрузка данных
df_raw = pd.read_csv(data_path_raw)
df_cleaned = pd.read_csv(data_path_cleaned)

# Загрузка обучающих и тестовых данных
X_train = pd.read_csv('D:/Projektiki/predykcja_rezerwacji/X_train.csv')
X_test = pd.read_csv('D:/Projektiki/predykcja_rezerwacji/X_test.csv')
X_train_scaled = pd.read_csv('D:/Projektiki/predykcja_rezerwacji/X_train_scaled.csv')
X_test_scaled = pd.read_csv('D:/Projektiki/predykcja_rezerwacji/X_test_scaled.csv')
y_train = pd.read_csv('D:/Projektiki/predykcja_rezerwacji/y_train.csv')['is_canceled']
y_test = pd.read_csv('D:/Projektiki/predykcja_rezerwacji/y_test.csv')['is_canceled']
scaler_path = 'D:/Projektiki/predykcja_rezerwacji/scaler.pkl'
scaler = joblib.load(scaler_path)

# Загрузка моделей
lr_model_path = 'D:/Projektiki/predykcja_rezerwacji/logistic_regression_model.pkl'
xgb_model_path = 'D:/Projektiki/predykcja_rezerwacji/best_model_xgb.pkl'
nn_model_path = 'D:/Projektiki/predykcja_rezerwacji/tf_model.h5'

model_lr = joblib.load(lr_model_path)
model_xgb = joblib.load(xgb_model_path)
model_tf = keras.models.load_model(nn_model_path)

# Заголовок приложения
st.title("Analiza i Predykcja Anulacji Rezerwacji Hotelowych")

# Боковая панель навигации
st.sidebar.header("Ustawienia")
section = st.sidebar.selectbox("Wybierz sekcję", ["Podstawowe informacje", "EDA", "Modelowanie", "Predykcja"])

if section == "Podstawowe informacje":
    st.header("Podstawowe informacje o zestawie danych")

    st.subheader("Nieoczyszczony zestaw danych (hotel_booking)")
    st.write(f"Rozmiar zestawu danych: {df_raw.shape[0]} wierszy, {df_raw.shape[1]} kolumn")
    st.write("Przykładowe 5 wierszy zestawu danych:")
    st.write(df_raw.head())
    st.write("Opis danych:")
    st.write(df_raw.describe())

    st.subheader("Oczyszczony zestaw danych (hotel_booking1)")
    st.write(f"Rozmiar zestawu danych: {df_cleaned.shape[0]} wierszy, {df_cleaned.shape[1]} kolumn")
    st.write("Przykładowе 5 wierszy zestawu danych:")
    st.write(df_cleaned.head())
    st.write("Opis danych:")
    st.write(df_cleaned.describe())

    st.subheader("Macierz korelacji dla zestawu danych liczbowych")
    numeric_cols = df_raw.select_dtypes(include=['number']).columns.tolist()
    corr_matrix_raw = df_raw[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix_raw, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    st.pyplot(plt)

elif section == "EDA":
    st.header("Analiza danych")
    variable_x = st.selectbox("Wybierz zmienną X", df_cleaned.columns)
    variable_y = st.selectbox("Wybierz zmienną Y", df_cleaned.columns)

    plot_type = st.selectbox("Typ wykresu",
                             ["Histogram zależności", "Wykres liniowy zależności", "Boxplot", "Scatterplot"])

    use_is_canceled = st.checkbox("Użyj is_canceled do podziału danych")

    if st.button("Wygeneruj"):
        if plot_type == "Histogram zależności":
            plt.figure(figsize=(10, 6))
            if use_is_canceled:
                sns.histplot(data=df_cleaned, x=variable_x, hue='is_canceled', multiple="stack", bins=30)
            else:
                sns.histplot(data=df_cleaned, x=variable_x, hue=variable_y, multiple="stack", bins=30)
            plt.title(f"Histogram zależności {variable_x} od {variable_y if not use_is_canceled else 'is_canceled'}")
            plt.xlabel(variable_x)
            plt.ylabel("Częstość")
            st.pyplot(plt)

        elif plot_type == "Wykres liniowy zależności":
            plt.figure(figsize=(10, 6))
            if use_is_canceled:
                sns.lineplot(x=df_cleaned[variable_x], y=df_cleaned[variable_y], hue=df_cleaned['is_canceled'])
            else:
                sns.lineplot(x=df_cleaned[variable_x], y=df_cleaned[variable_y])
            plt.title(f"Wykres liniowy zależności {variable_y} od {variable_x}")
            plt.xlabel(variable_x)
            plt.ylabel(variable_y)
            st.pyplot(plt)

        elif plot_type == "Boxplot":
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df_cleaned[variable_x], y=df_cleaned[variable_y])
            plt.title(f"Boxplot: {variable_y} w zależności od {variable_x}")
            plt.xlabel(variable_x)
            plt.ylabel(variable_y)
            st.pyplot(plt)

        elif plot_type == "Scatterplot":
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df_cleaned[variable_x], y=df_cleaned[variable_y])
            plt.title(f"Scatterplot: {variable_y} od {variable_x}")
            plt.xlabel(variable_x)
            plt.ylabel(variable_y)
            st.pyplot(plt)

elif section == "Modelowanie":
    st.header("Porównanie modeli")

    # Обновление метрик моделей
    y_pred_lr = model_lr.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    lr_classification_report = classification_report(y_test, y_pred_lr, output_dict=True)

    y_pred_xgb = model_xgb.predict(X_test_scaled)
    xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
    xgb_classification_report = classification_report(y_test, y_pred_xgb, output_dict=True)

    tf_predictions = model_tf.predict(X_test_scaled)
    y_pred_tf = (tf_predictions > 0.5).astype("int32")
    tf_accuracy = accuracy_score(y_test, y_pred_tf)
    tf_classification_report = classification_report(y_test, y_pred_tf, output_dict=True)

    accuracies = [lr_accuracy, xgb_accuracy, tf_accuracy]
    models = ['Regresja Logistyczna', 'XGBoost', 'Model TensorFlow']

    plot_choice = st.radio("Wybierz wykres", ("Krzywa ROC", "Wykres porównania dokładności modeli"))

    if plot_choice == "Krzywa ROC":
        st.subheader("Porównanie krzywych ROC")

        y_prob_lr = model_lr.predict_proba(X_test_scaled)[:, 1]
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
        roc_auc_lr = auc(fpr_lr, tpr_lr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_lr, tpr_lr, label=f'Regresja Logistyczna (AUC = {roc_auc_lr:.2f})', color='blue')

        y_prob_xgb = model_xgb.predict_proba(X_test_scaled)[:, 1]
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
        roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
        plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})', color='orange')

        # Исправление для модели TensorFlow
        y_prob_tf = tf_predictions.ravel()  # Преобразование предсказаний в одномерный массив
        fpr_tf, tpr_tf, _ = roc_curve(y_test, y_prob_tf)
        roc_auc_tf = auc(fpr_tf, tpr_tf)
        plt.plot(fpr_tf, tpr_tf, label=f'Model TensorFlow (AUC = {roc_auc_tf:.2f})', color='green')

        plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Losowy wybór')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Krzywa ROC')
        plt.legend()
        st.pyplot(plt)

    elif plot_choice == "Wykres porównania dokładności modeli":
        st.subheader("Wykres porównania dokładności modeli")

        plt.figure(figsize=(10, 5))
        plt.bar(models, accuracies, color=['blue', 'orange', 'green'])
        plt.xlabel('Model')
        plt.ylabel('Dokładność')
        plt.title('Porównanie modeli')
        st.pyplot(plt)

    # Добавляем выбор модели для вывода метрик
    model_choice = st.selectbox("Wybierz model do wyświetlenia metryk", models)

    st.subheader(f"Metryki dla {model_choice}")

    def display_classification_report(report):
        report_df = pd.DataFrame(report).transpose()
        st.table(report_df)

    if model_choice == 'Regresja Logistyczna':
        st.write("**Dokładność:**", f"{lr_accuracy:.4f}")
        st.write("**Raport klasyfikacji:**")
        display_classification_report(lr_classification_report)

    elif model_choice == 'XGBoost':
        st.write("**Dokładność:**", f"{xgb_accuracy:.4f}")
        st.write("**Raport klasyfikacji:**")
        display_classification_report(xgb_classification_report)

    elif model_choice == 'Model TensorFlow':
        st.write("**Dokładność:**", f"{tf_accuracy:.4f}")
        st.write("**Raport klasyfikacji:**")
        display_classification_report(tf_classification_report)





if section == "Predykcja":
    st.header("Predykcja anulacji rezerwacji")

    st.subheader("Wprowadź dane do predykcji")

    lead_time = st.number_input("Lead Time", min_value=0)
    arrival_date_year = st.number_input("Arrival Date Year", min_value=2000, max_value=2023)
    arrival_date_month = st.selectbox("Arrival Date Month", df_cleaned['arrival_date_month'].unique())
    arrival_date_week_number = st.number_input("Arrival Date Week Number", min_value=1, max_value=53)
    arrival_date_day_of_month = st.number_input("Arrival Date Day of Month", min_value=1, max_value=31)
    stays_in_weekend_nights = st.number_input("Stays in Weekend Nights", min_value=0)
    stays_in_week_nights = st.number_input("Stays in Week Nights", min_value=0)
    adults = st.number_input("Adults", min_value=0)
    children = st.number_input("Children", min_value=0)
    babies = st.number_input("Babies", min_value=0)
    meal = st.selectbox("Meal", df_cleaned['meal'].unique())
    country = st.selectbox("Country", df_cleaned['country'].unique())
    market_segment = st.selectbox("Market Segment", df_cleaned['market_segment'].unique())
    distribution_channel = st.selectbox("Distribution Channel", df_cleaned['distribution_channel'].unique())
    is_repeated_guest = st.number_input("Is Repeated Guest", min_value=0, max_value=1)
    previous_cancellations = st.number_input("Previous Cancellations", min_value=0)
    previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0)
    reserved_room_type = st.selectbox("Reserved Room Type", df_cleaned['reserved_room_type'].unique())
    assigned_room_type = st.selectbox("Assigned Room Type", df_cleaned['assigned_room_type'].unique())
    booking_changes = st.number_input("Booking Changes", min_value=0)
    deposit_type = st.selectbox("Deposit Type", df_cleaned['deposit_type'].unique())
    agent = st.selectbox("Agent", df_cleaned['agent'].unique())
    company = st.selectbox("Company", df_cleaned['company'].unique())
    days_in_waiting_list = st.number_input("Days in Waiting List", min_value=0)
    customer_type = st.selectbox("Customer Type", df_cleaned['customer_type'].unique())
    adr = st.number_input("ADR", min_value=0.0)
    required_car_parking_spaces = st.number_input("Required Car Parking Spaces", min_value=0)
    total_of_special_requests = st.number_input("Total of Special Requests", min_value=0)

    if st.button("Przewidź"):
        new_data = {
            'lead_time': [lead_time],
            'arrival_date_year': [arrival_date_year],
            'arrival_date_month': [arrival_date_month],
            'arrival_date_week_number': [arrival_date_week_number],
            'arrival_date_day_of_month': [arrival_date_day_of_month],
            'stays_in_weekend_nights': [stays_in_weekend_nights],
            'stays_in_week_nights': [stays_in_week_nights],
            'adults': [adults],
            'children': [children],
            'babies': [babies],
            'meal': [meal],
            'country': [country],
            'market_segment': [market_segment],
            'distribution_channel': [distribution_channel],
            'is_repeated_guest': [is_repeated_guest],
            'previous_cancellations': [previous_cancellations],
            'previous_bookings_not_canceled': [previous_bookings_not_canceled],
            'reserved_room_type': [reserved_room_type],
            'assigned_room_type': [assigned_room_type],
            'booking_changes': [booking_changes],
            'deposit_type': [deposit_type],
            'agent': [agent],
            'company': [company],
            'days_in_waiting_list': [days_in_waiting_list],
            'customer_type': [customer_type],
            'adr': [adr],
            'required_car_parking_spaces': [required_car_parking_spaces],
            'total_of_special_requests': [total_of_special_requests]
        }

        new_df = pd.DataFrame(new_data)
        new_df = pd.get_dummies(new_df, drop_first=True)

        missing_cols = set(X_train.columns) - set(new_df.columns)
        for col in missing_cols:
            new_df[col] = 0
        new_df = new_df[X_train.columns]

        new_df_scaled = scaler.transform(new_df)

        prediction_lr = model_lr.predict(new_df_scaled)
        probability_lr = model_lr.predict_proba(new_df_scaled)[:, 1]

        prediction_xgb = model_xgb.predict(new_df_scaled)
        probability_xgb = model_xgb.predict_proba(new_df_scaled)[:, 1]

        prediction_tf = np.round(model_tf.predict(new_df_scaled)).astype(int)
        probability_tf = model_tf.predict(new_df_scaled).ravel()

        st.subheader("Wyniki predykcji")
        st.write("**Regresja Logistyczna**")
        st.write(f"Predykcja: {'Anulowano' if prediction_lr[0] == 1 else 'Nie anulowano'}")
        st.write(f"Prawdopodobieństwo anulacji: {probability_lr[0]:.4f}")

        st.write("**XGBoost**")
        st.write(f"Predykcja: {'Anulowano' if prediction_xgb[0] == 1 else 'Nie anulowano'}")
        st.write(f"Prawdopodobieństwo anulacji: {probability_xgb[0]:.4f}")

        st.write("**Sieć neuronowa TensorFlow**")
        st.write(f"Predykcja: {'Anulowano' if prediction_tf[0] == 1 else 'Nie anulowano'}")
        st.write(f"Prawdopodobieństwo anulacji: {probability_tf[0]:.4f}")

        ## Uruchomienie aplikacji
        # 1.tf_env\Scripts\activate
        # 2.streamlit run StreamlitAPP.py