import pandas as pd
import numpy as np
from config.settings import DATA_PATH


class FeatureEngineer:
    # Константы для обработки данных
    REQUIRED_COLUMNS = ["Store", "Date", "Sales", "DayOfWeek", "Promo"]
    LAG_DAYS = [1, 7, 14, 28]
    VALID_STORE_TYPES = ["a", "b", "c", "d"]
    VALID_ASSORTMENT_TYPES = ["a", "b", "c"]

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.feature_names = []
        self.nan_info = {}
        self._processed_records = 0

    def handle_nan_values(self, df):
        """Обработка NaN значений с сохранением бизнес-логики"""
        if self.verbose:
            print("Обработка NaN значений...")

        df = df.copy()
        self.nan_info["before"] = df.isna().sum().sum()

        try:
            # Competition features - NaN означает отсутствие конкурента
            competition_columns = ["CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]
            df["HasCompetition"] = (~df["CompetitionOpenSinceYear"].isna()).astype(int)

            # Заполняем Competition пропуски
            df[competition_columns] = df[competition_columns].fillna(0)

            # Promo2 features - NaN означает неучастие в программе
            promo2_columns = ["Promo2SinceWeek", "Promo2SinceYear"]
            df["InPromo2"] = (~df["Promo2SinceYear"].isna()).astype(int)

            # Заполняем Promo2 пропуски
            df[promo2_columns] = df[promo2_columns].fillna(0)
            df["PromoInterval"] = df["PromoInterval"].fillna("NoPromo")

            # Остальные числовые колонки заполняем медианой
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())

            # Категориальные колонки заполняем модой
            categorical_columns = df.select_dtypes(include=["object"]).columns
            for col in categorical_columns:
                if df[col].isna().any():
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknow"
                    df[col] = df[col].fillna(mode_val)

            self.nan_info["after"] = df.isna().sum().sum()

            if self.verbose:
                print(f"Обработано NaN: было {self.nan_info["before"]}, стало {self.nan_info["after"]}")

            return df

        except Exception as e:
            print(f"Ошибка обработки NaN значений: {e}")
            raise

    def fill_lag_missing_values(self, df):
        """Корректное заполнение пропусков в лаговых признаках"""
        if self.verbose:
            print("Заполнение пропусков в лагах...")

        lag_columns = [col for col in df.columns if "Lag" in col or "Rolling" in col]

        initial_na_count = df[lag_columns].isna().sum().sum()
        if self.verbose:
            print(f"Найдено пропусков в лагах {initial_na_count}")

        for col in lag_columns:
            na_count = df[col].isna().sum()
            if na_count > 0:
                if "SalesLag" in col:
                    # Для лагов продаж - заполняем медианой по магазину + дню недели
                    df[col] = df.groupby(["Store", "DayOfWeek"])[col].transform(
                        lambda x: x.fillna(x.median() if not x.isna().all() else 0)
                    )
                    if self.verbose:
                        print(f"{col}: заполнено {na_count} пропусков (медиана по магазину/дню)")

                elif "RollingMean" in col:
                    # Для скользящего среднего - заполняем общим средним по магазину
                    df[col] = df.groupby("Store")[col].transform(
                        lambda x: x.fillna(df["Sales"].mean() if not df["Sales"].isna().all() else 0)
                    )
                    if self.verbose:
                        print(f"{col}: заполнено {na_count} пропусков (общее среднее по магазину)")

                elif "RollingStd" in col:
                    # Для скользящего STD - заполняем общим STD по магазину
                    df[col] = df.groupby("Store")[col].transform(
                        lambda x: x.fillna(df["Sales"].std() if not df["Sales"].isna().all() else 1)
                    )
                    if self.verbose:
                        print(f"{col}: заполнено {na_count} пропусков (общий STD по магазину)")

        final_na_count = df[lag_columns].isna().sum().sum()
        if self.verbose:
            print(f"Осталось пропусков: {final_na_count}")

        return df

    def create_temporal_features(self, df):
        """Создание временных признаков для учета сезонности"""
        if self.verbose:
            print("Создание временных признаков...")

        df = df.copy()

        # Базовые временные признаки
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Week"] = df["Date"].dt.isocalendar().week
        df["DayOfYear"] = df["Date"].dt.dayofyear
        df["IsSaturday"] = (df["DayOfWeek"] == 6).astype(int)
        df["IsSunday"] = (df["DayOfWeek"] == 7).astype(int)

        # Циклические признаки при сезонности
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
        df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
        df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

        # Сезонные признаки
        seasons = [(3, 5, "IsSpring"), (6, 8, "IsSummer"), (9, 11, "IsAutumn"), (12, 2, "IsWinter")]

        for start, end, name in seasons:
            if start <= end:
                df[name] = ((df["Month"] >= start) & (df["Month"] <= end)).astype(int)
            else:
                df[name] = ((df["Month"] >= start) | (df["Month"] <= end)).astype(int)

        return df

    def create_promo_features(self, df):
        """Создание признаков связанных с акциями и промо-кампаниями"""
        if self.verbose:
            print("Создание промо-признаков...")

        df = df.sort_values(["Store", "Date"]).copy()

        # Длина текущей акционной серии
        df["PromoSequence"] = (df["Promo"] != df.groupby("Store")["Promo"].shift(1)).cumsum()

        # Длина последовательности промо-акции
        df["PromoSequenceLength"] = df.groupby(["Store", "PromoSequence"]).cumcount() + 1
        df["PromoSequenceLength"] = np.where(df["Promo"] == 1, df["PromoSequenceLength"], 0)

        # Дни с последней промо-акции
        df["DaysSinceLastPromo"] = df.groupby("Store")["Promo"].transform(
            lambda x: x.where(x == 0, 0).groupby(x.ne(x.shift()).cumsum()).cumcount()
        )

        # Дополнительные бизнес-признаки
        df["IsPromoStart"] = ((df["Promo"] == 1) & (df.groupby("Store")["Promo"].shift(1) == 0)).astype(int)
        df["IsPromoEnd"] = ((df["Promo"] == 0) & (df.groupby("Store")["Promo"].shift(1) == 1)).astype(int)

        return df

    def create_lag_features(self, df, lags=None):
        """Создание лаговых признаков для учета временных зависимостей"""
        if self.verbose:
            print("Создание лаговых признаков...")

        if lags is None:
            lags = self.LAG_DAYS

        df = df.sort_values(["Store", "Date"]).copy()

        for lag in lags:
            # Лаговые признаки
            df[f"SalesLag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)
            # Скользящее среднее только на исторических данных
            df[f"RollingMean_{lag}"] = df.groupby("Store")["Sales"].transform(
                lambda x: x.shift(1).rolling(window=lag, min_periods=1).mean()
            )
            # Скользящее стандартное отклонение
            df[f"RollingStd_{lag}"] = df.groupby("Store")["Sales"].transform(
                lambda x: x.shift(1).rolling(window=lag, min_periods=1).std()
            )

        df = self.fill_lag_missing_values(df)

        return df

    def create_holiday_features(self, df):
        """Создание признаков праздников и выходных дней"""
        if self.verbose:
            print("Создание праздничных признаков...")

        df = df.copy()

        # Бинарные признаки праздников
        df["IsHoliday"] = (df["StateHoliday"] != "0").astype(int)
        df["IsPublicHoliday"] = (df["StateHoliday"] == "a").astype(int)
        df["IsEaster"] = (df["StateHoliday"] == "b").astype(int)
        df["IsChristmas"] = (df["StateHoliday"] == "c").astype(int)

        df["WasHolidayYesterday"] = (df.groupby("Store")["IsHoliday"].shift(1) == 1).astype(int)
        df["WasHolidayLastWeek"] = df.groupby("Store")["IsHoliday"].transform(
            lambda x: x.rolling(7, min_periods=1).max().shift(1)
        ).fillna(0).astype(int)

        return df

    def create_store_features(self, df):
        """Создание признаков характеризующих магазины"""
        if self.verbose:
            print("Создание признаков магазинов...")

        df = df.copy()

        # Тип магазина
        store_type_dummies = pd.get_dummies(df["StoreType"], prefix="StoreType", dtype=int)
        df = pd.concat([df, store_type_dummies], axis=1)

        # Ассортимент
        assortment_dummies = pd.get_dummies(df["Assortment"], prefix="Assortment", dtype=int)
        df = pd.concat([df, assortment_dummies], axis=1)

        return df

    def _validate_input_data(self, df):
        """Валидация входных данных перед обработкой"""
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")

        if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            raise ValueError("Колонка Date должна содержать данные типа datetime")

    def prepare_final_dataset(self, df, verbose=False):
        """Подготовка финального датасета для обучения модели"""
        if verbose:
            self.verbose = True

        if self.verbose:
            print("Начало подготовки финального датасета...")

        try:
            # Валидация входных данных
            self._validate_input_data(df)

            # Обработка NaN в первую очередь
            df = self.handle_nan_values(df)

            # Применяем все преобразования
            df = self.create_temporal_features(df)
            df = self.create_promo_features(df)
            df = self.create_holiday_features(df)
            df = self.create_store_features(df)
            df = self.create_lag_features(df)

            # Определяем финальные признаки
            exclude_columns = ["Date", "Sales", "Customers", "Open", "StateHoliday", "StoreType", "Assortment", "PromoInterval"]
            feature_columns = [col for col in df.columns if col not in exclude_columns]

            self.feature_names = feature_columns
            self._processed_records = len(df)

            if self.verbose:
                print(f"Итоговое количество признаков: {len(feature_columns)}")
                print(f"Финальная форма данных: {df.shape}")
                print(f"Обработано записей: {self._processed_records}")

            return df[feature_columns + ["Sales"]], feature_columns

        except Exception as e:
            print(f"Ошибка подготовки датасета: {e}")
            raise

    def get_feature_statistics(self):
        """Получение статистики по сгенерированным признакам"""
        return {
            "total_features": len(self.feature_names),
            "processed_records": self._processed_records,
            "nan_processed": self.nan_info,
            "feature_categories": {
                "temporal": len([f for f in self.feature_names if any(x in f for x in ["Year", "Month", "Week", "Day"])]),
                "promo": len([f for f in self.feature_names if "Promo" in f]),
                "holiday": len([f for f in self.feature_names if "Holiday" in f]),
                "lag": len([f for f in self.feature_names if "Lag" in f or "Rolling" in f]),
                "store": len([f for f in self.feature_names if "StoreType" in f or "Assortment" in f])
            }
        }


if __name__ == "__main__":
    cleaned_data = pd.read_csv(DATA_PATH / "processed/cleaned_data.csv", parse_dates=["Date"])

    engineer = FeatureEngineer(verbose=True)
    final_data, feature_names = engineer.prepare_final_dataset(cleaned_data)

    final_data.to_csv(DATA_PATH / "processed/final_dataset.csv", index=False)

    stats = engineer.get_feature_statistics()
    print(f"Статистика обработки: {stats}")

    print("Feature engineering успешно завершен")