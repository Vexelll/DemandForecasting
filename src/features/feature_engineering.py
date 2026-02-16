import pandas as pd
import numpy as np
import logging
from config.settings import DATA_PATH


class FeatureEngineer:
    # Константы для валидации данных
    REQUIRED_COLUMNS = ["Store", "Date", "Sales", "DayOfWeek", "Promo"]
    LAG_DAYS = [1, 7, 14, 28]
    VALID_STORE_TYPES = ["a", "b", "c", "d"]
    VALID_ASSORTMENT_TYPES = ["a", "b", "c"]

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_names = []
        self.nan_info = {}
        self._processed_records = 0

    def handle_nan_values(self, df):
        """Обработка NaN значений с сохранением бизнес-логики"""
        self.logger.info("Обработка NaN значений...")

        df = df.copy()
        self.nan_info["before"] = df.isna().sum().sum()

        try:
            # 1. Семантические признаки
            # NaN в CompetitionDistance означает отсутствие конкурента поблизости
            if "CompetitionDistance" in df.columns:
                df["HasCompetition"] = (~df["CompetitionOpenSinceYear"].isna()).astype(int)
                # Большое расстояние для отсутствующего конкурента
                df["CompetitionDistance"] = df["CompetitionDistance"].fillna(100000)

            competition_columns = ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]
            for col in competition_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(0) # 0 = конкурентов нет

            # NaN означает неучастие магазина в программе Promo2
            promo2_columns = ["Promo2SinceWeek", "Promo2SinceYear"]

            if "Promo2SinceYear" in df.columns:
                df["InPromo2"] = (~df["Promo2SinceYear"].isna()).astype(int)

            for col in promo2_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(0)

            if "PromoInterval" in df.columns:
                df["PromoInterval"] = df["PromoInterval"].fillna("NoPromo")

            # 2. Бинарные признаки - заполняем модой
            binary_columns = ["Open", "Promo", "SchoolHoliday"]
            for col in binary_columns:
                if col in df.columns and df[col].isna().any():
                    non_nan_values = df[col].dropna()
                    if len(non_nan_values) > 0:
                        mode_val = non_nan_values.mode()
                        fill_value = mode_val.iloc[0] if not mode_val.empty else 0
                    else:
                        fill_value = 0

                    df[col] = df[col].fillna(fill_value).astype(int)

            # 3. Числовые признаки - медиана с группировкой по магазину
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            exclude_from_numeric = (
                binary_columns + competition_columns + promo2_columns +
                ["HasCompetition", "InPromo2"]
            )
            numeric_columns = [col for col in numeric_columns if col not in exclude_from_numeric]

            for col in numeric_columns:
                if df[col].isna().any():
                    if "Store" in df.columns and df["Store"].nunique() > 1:
                        df[col] = df.groupby("Store")[col].transform(
                            lambda x: x.fillna(x.median() if not x.isna().all() else 0)
                        )
                    else:
                        df[col] = df[col].fillna(df[col].median())

            # 4. Категориальные признаки - мода
            categorical_columns = df.select_dtypes(include=["object"]).columns
            categorical_columns = [col for col in categorical_columns if col != "PromoInterval"]

            for col in categorical_columns:
                if df[col].isna().any():
                    non_nan_values = df[col].dropna()
                    if len(non_nan_values) > 0:
                        mode_val = non_nan_values.mode()
                        fill_value = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                    else:
                        fill_value = "Unknown"

                    df[col] = df[col].fillna(fill_value)

            self.nan_info["after"] = df.isna().sum().sum()
            self.logger.info(f"Обработано NaN: было {self.nan_info["before"]}, стало {self.nan_info["after"]}")

            return df

        except Exception as e:
            self.logger.error(f"Ошибка обработки NaN значений: {e}")
            raise

    def create_temporal_features(self, df):
        """Создание временных признаков для учета сезонности"""
        self.logger.info("Создание временных признаков...")

        df = df.copy()

        # Базовые временные признаки
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
        df["Day"] = df["Date"].dt.day
        df["DayOfYear"] = df["Date"].dt.dayofyear

        # Признаки выходных
        df["IsSaturday"] = (df["DayOfWeek"] == 6).astype(int)
        df["IsSunday"] = (df["DayOfWeek"] == 7).astype(int)
        df["IsWeekend"] = ((df["DayOfWeek"] >= 6)).astype(int)

        # Циклическое кодирование для сохранения цикличности
        # Месяц: декабрь (12) должен быть близок к январю (1)
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

        # День недели: воскресенье (7) близко к понедельнику (1)
        df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
        df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

        # День месяца (для учета зарплатных эффектов)
        df["DayOfMonth_sin"] = np.sin(2 * np.pi * df["Day"] / 31)
        df["DayOfMonth_cos"] = np.cos(2 * np.pi * df["Day"] / 31)

        # Сезонные признаки
        seasons = [(3, 5, "IsSpring"), (6, 8, "IsSummer"), (9, 11, "IsAutumn"), (12, 2, "IsWinter")]

        for start, end, name in seasons:
            if start <= end:
                df[name] = ((df["Month"] >= start) & (df["Month"] <= end)).astype(int)
            else:
                # Зима: декабрь (12) или январь-февраль (1-2)
                df[name] = ((df["Month"] >= start) | (df["Month"] <= end)).astype(int)

        # Признаки начала/конца периодов (зарплатные эффекты)
        df["IsMonthStart"] = (df["Day"] <= 5).astype(int)
        df["IsMonthEnd"] = (df["Day"] >= 25).astype(int)
        df["IsYearStart"] = ((df["Month"] == 1) & (df["Day"] <= 7)).astype(int)
        df["IsYearEnd"] = ((df["Month"] == 12) & (df["Day"] >= 25)).astype(int)

        return df

    def create_promo_features(self, df):
        """Создание признаков связанных с акциями и промо-кампаниями"""
        self.logger.info("Создание промо-признаков...")

        df = df.sort_values(["Store", "Date"]).copy()

        # Идентификатор последовательности (меняется при смене Promo 0->1 или 1->0)
        df["PromoSequence"] = df.groupby("Store")["Promo"].transform(lambda x: (x != x.shift()).cumsum())

        # Длина текущей промо-серии
        df["PromoSequenceLength"] = df.groupby(["Store", "PromoSequence"]).cumcount() + 1
        df["PromoSequenceLength"] = np.where(df["Promo"] == 1, df["PromoSequenceLength"], 0)

        # Дни с последней промо-акции
        df["DaysSinceLastPromo"] = self._calculate_days_since_last_promo(df)

        # Начало и конец промо-акции
        df["IsPromoStart"] = ((df["Promo"] == 1) & (df.groupby("Store")["Promo"].shift(1) == 0)).astype(int)
        df["IsPromoEnd"] = ((df["Promo"] == 0) & (df.groupby("Store")["Promo"].shift(1) == 1)).astype(int)

        return df

    def _calculate_days_since_last_promo(self, df):
        """Расчет календарных дней с последней промо-акции"""
        df = df.sort_values(["Store", "Date"])

        result = pd.Series(dtype=float, index=df.index)

        # Для промо-дней результат = 0
        is_promo = df["Promo"] == 1
        result[is_promo] = 0

        # Все промо-даты для merge_asof
        promo_records = df.loc[is_promo, ["Store", "Date"]].copy()
        promo_records = promo_records.rename(columns={"Date": "LastPromoDate"})
        promo_records = promo_records.sort_values(["Store", "LastPromoDate"])

        # Запись без промо
        non_promo = df.loc[~is_promo, ["Store", "Date"]].copy()
        non_promo["_orig_idx"] = non_promo.index
        non_promo = non_promo.sort_values(["Store", "Date"])

        if len(promo_records) > 0 and len(non_promo) > 0:
            # merge_asof - ближайшая промо-дата строго в прошлом
            merged = pd.merge_asof(
                non_promo.sort_values("Date"),
                promo_records.sort_values("LastPromoDate"),
                left_on="Date",
                right_on="LastPromoDate",
                by="Store",
                direction="backward"
            )
            merged = merged.set_index("_orig_idx")
            days_since = (merged["Date"] - merged["LastPromoDate"]).dt.days
            days_since = days_since.fillna(999)
            result.loc[merged.index] = days_since.values
        else:
            result[~is_promo] = 999

        return result

    def create_lag_features(self, df, lags=None):
        """Создание лаговых признаков для учета временных зависимостей"""
        self.logger.info("Создание лаговых признаков...")

        if lags is None:
            lags = self.LAG_DAYS

        if df.empty:
            self.logger.warning("Предупреждение: пустой DataFrame")
            return df

        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

        # Создаём полный календарь для каждого магазина
        store_frames = []

        for store_id in df["Store"].unique():
            store_df = df[df["Store"] == store_id][["Date", "Sales"]].copy()
            store_df = store_df.set_index("Date")

            # Заполняем пропущенные даты (выходные, праздники)
            full_idx = pd.date_range(store_df.index.min(), store_df.index.max(), freq="D")
            store_df = store_df.reindex(full_idx)

            # Сдвинутые продажи для Rolling (не включаем текущий день)
            shifted_sales = store_df["Sales"].shift(1)

            for lag in lags:
                # Лаговые значения продаж
                store_df[f"SalesLag_{lag}"] = store_df["Sales"].shift(lag)

                # Rolling статистики
                store_df[f"RollingMean_{lag}"] = shifted_sales.rolling(window=lag, min_periods=1).mean()
                store_df[f"RollingStd_{lag}"] = shifted_sales.rolling(window=lag, min_periods=1).std()
                store_df[f"RollingMin_{lag}"] = shifted_sales.rolling(window=lag, min_periods=1).min()
                store_df[f"RollingMax_{lag}"] = shifted_sales.rolling(window=lag, min_periods=1).max()

            store_df["Store"] = store_id
            store_df = store_df.reset_index().rename(columns={"index": "Date"})
            store_frames.append(store_df)

        # Объединяем все магазины
        calendar_df = pd.concat(store_frames, ignore_index=True)

        # Возвращаем только исходные даты (без добавленных выходных)
        lag_cols = [c for c in calendar_df.columns if "Lag" in c or "Rolling" in c]
        result = df.merge(
            calendar_df[["Store", "Date"] + lag_cols],
            on=["Store", "Date"],
            how="left"
        )

        # Заполнение пропусков в лаговых признаках
        result = self.fill_lag_missing_values(result)


        na_count = result[lag_cols].isna().sum().sum()
        self.logger.info(f"Лаговые признаки созданы. Осталось NaN: {na_count}")

        return result


    def fill_lag_missing_values(self, df):
        """Корректное заполнение пропусков в лаговых признаках"""
        self.logger.info("Заполнение пропусков в лагах...")

        lag_columns = [col for col in df.columns if "Lag" in col or "Rolling" in col]

        initial_na_count = df[lag_columns].isna().sum().sum()
        self.logger.info(f"Найдено пропусков в лагах {initial_na_count}")

        df = df.sort_values(["Store", "Date"]).copy()

        for col in lag_columns:
            na_count = df[col].isna().sum()
            if na_count > 0:
                if "SalesLag" in col:
                    # Для лагов продаж - expanding mean по магазину
                    df[col] = df.groupby("Store")[col].transform(
                        lambda x: x.fillna(x.expanding(min_periods=1).mean())
                    )
                    # Если остались NaN - заполняем нулём
                    df[col] = df[col].fillna(0)

                elif "RollingMean" in col:
                    # Для скользящего среднего - expanding mean по магазину
                    df[col] = df.groupby("Store")[col].transform(
                        lambda x: x.fillna(x.expanding(min_periods=1).mean())
                    )
                    df[col] = df[col].fillna(0)

                elif "RollingStd" in col:
                    # Для стандартного отклонения - expanding std по магазину
                    df[col] = df.groupby("Store")[col].transform(
                        lambda x: x.fillna(x.expanding(min_periods=1).std())
                    )
                    df[col] = df[col].fillna(0)

                elif "RollingMin" in col:
                    # Для минимума - expanding min по магазину
                    df[col] = df.groupby("Store")[col].transform(
                        lambda x: x.fillna(x.expanding(min_periods=1).min())
                    )
                    df[col] = df[col].fillna(0)

                elif "RollingMax" in col:
                    # Для максимума - expanding max по магазину
                    df[col] = df.groupby("Store")[col].transform(
                        lambda x: x.fillna(x.expanding(min_periods=1).max())
                    )
                    df[col] = df[col].fillna(0)

        final_na_count = df[lag_columns].isna().sum().sum()
        self.logger.info(f"Осталось пропусков: {final_na_count}")

        return df

    def create_holiday_features(self, df):
        """Создание признаков праздников и выходных дней"""
        self.logger.info("Создание праздничных признаков...")

        df = df.copy()

        # Бинарные признаки праздников
        df["IsHoliday"] = (df["StateHoliday"] != "0").astype(int)
        df["IsPublicHoliday"] = (df["StateHoliday"] == "a").astype(int)
        df["IsEaster"] = (df["StateHoliday"] == "b").astype(int)
        df["IsChristmas"] = (df["StateHoliday"] == "c").astype(int)

        # Эффект праздников на соседние дни
        df["WasHolidayYesterday"] = df.groupby("Store")["IsHoliday"].shift(1).fillna(0).astype(int)
        df["IsHolidayTomorrow"] = df.groupby("Store")["IsHoliday"].shift(-1).fillna(0).astype(int)

        # Праздник на прошлой неделе
        df["WasHolidayLastWeek"] = df.groupby("Store")["IsHoliday"].transform(
            lambda x: x.rolling(7, min_periods=1).max().shift(1)
        ).fillna(0).astype(int)

        return df

    def create_store_features(self, df):
        """Создание признаков характеризующих магазины"""
        self.logger.info("Создание признаков магазинов...")

        df = df.copy()

        # One hot encoding типа магазина
        if "StoreType" in df.columns:
            store_type_dummies = pd.get_dummies(df["StoreType"], prefix="StoreType", dtype=int)
            df = pd.concat([df, store_type_dummies], axis=1)

        # One hot encoding ассортимента
        if "Assortment" in df.columns:
            assortment_dummies = pd.get_dummies(df["Assortment"], prefix="Assortment", dtype=int)
            df = pd.concat([df, assortment_dummies], axis=1)

        return df

    def create_competition_features(self, df):
        """Создание признаков конкуренции"""
        self.logger.info("Создание признаков конкуренции...")

        df = df.copy()

        # Месяцы с момента открытия конкурента
        required_cols = ["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth", "Year", "Month"]
        if all(col in df.columns for col in required_cols):
            df["CompetitionOpenMonths"] = (
            (df["Year"] - df["CompetitionOpenSinceYear"]) * 12 +
            (df["Month"] - df["CompetitionOpenSinceMonth"])
            )
            # Отрицательные значения = конкурент еще не открылся
            df["CompetitionOpenMonths"] = df["CompetitionOpenMonths"].clip(lower=0)

            # Бинарный признак: конкурент уже открылся
            df["CompetitionOpen"] = (df["CompetitionOpenMonths"] > 0).astype(int)

        # Категоризация расстояния до конкурента
        if "CompetitionDistance" in df.columns:
            df["CompetitionNear"] = (df["CompetitionDistance"] < 1000).astype(int)
            df["CompetitionMedium"] = (
                (df["CompetitionDistance"] >= 1000) &
                (df["CompetitionDistance"] < 5000)
            ).astype(int)
            df["CompetitionFar"] = (df["CompetitionDistance"] >= 5000).astype(int)

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
        self.logger.info("Начало подготовки финального датасета...")
        self.logger.info(f"Исходные данные: {df.shape[0]} записей, {df.shape[1]} колонок")

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
            df = self.create_competition_features(df)
            df = self.create_lag_features(df)

            # Определяем финальные признаки
            exclude_columns = ["Date", "Sales", "Customers", "Open", "StateHoliday", "StoreType", "Assortment", "PromoInterval", "PromoSequence"]
            feature_columns = [col for col in df.columns if col not in exclude_columns]

            self.feature_names = feature_columns
            self._processed_records = len(df)

            self.logger.info(f"Итоговое количество признаков: {len(feature_columns)}")
            self.logger.info(f"Финальная форма данных: {df.shape}")
            self.logger.info(f"Обработано записей: {self._processed_records}")

            # Возвращаем только признаки + целевую переменную
            final_df = df[feature_columns + ["Sales"]]

            return final_df, feature_columns

        except Exception as e:
            self.logger.error(f"Ошибка подготовки датасета: {e}")
            raise

    def get_feature_statistics(self):
        """Получение статистики по сгенерированным признакам"""
        return {
            "total_features": len(self.feature_names),
            "processed_records": self._processed_records,
            "nan_processed": self.nan_info,
            "feature_categories": {
                "temporal": len([f for f in self.feature_names if any(x in f for x in ["Year", "Month", "Week", "Day", "Season", "Weekend"])]),
                "promo": len([f for f in self.feature_names if "Promo" in f]),
                "holiday": len([f for f in self.feature_names if "Holiday" in f]),
                "lag": len([f for f in self.feature_names if "Lag" in f or "Rolling" in f]),
                "store": len([f for f in self.feature_names if "StoreType" in f or "Assortment" in f]),
                "competition": len([f for f in self.feature_names if "Competition" in f]),
            }
        }


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    cleaned_data = pd.read_csv(DATA_PATH / "processed/cleaned_data.csv", parse_dates=["Date"])

    engineer = FeatureEngineer()
    final_data, feature_names = engineer.prepare_final_dataset(cleaned_data)

    final_data.to_csv(DATA_PATH / "processed/final_dataset.csv", index=False)

    stats = engineer.get_feature_statistics()
    engineer.logger.info(f"Статистика обработки: {stats}")

    engineer.logger.info("Feature engineering успешно завершен")
