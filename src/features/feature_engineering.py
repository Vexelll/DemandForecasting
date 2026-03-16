import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np

from config.settings import DATA_PATH, get_feature_config, setup_logging


class FeatureEngineer:

    REQUIRED_COLUMNS = ["Store", "Date", "Sales", "DayOfWeek", "Promo"]
    VALID_STORE_TYPES = ["a", "b", "c", "d"]
    VALID_ASSORTMENT_TYPES = ["a", "b", "c"]
    EXCLUDE_COLUMNS = [
            "Date", "Sales", "Customers", "Open",
            "StateHoliday", "StoreType", "Assortment",
            "PromoInterval", "PromoSequence"
        ]

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.feature_names = []
        self.nan_info = {}
        self._processed_records = 0
        self.lag_days = get_feature_config().get("lag_days", [1, 7, 14, 28])

    def handle_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """NaN -> осмысленные значения: CompetitionDistance=100k, если нет конкурента, и т.д."""
        self.logger.info("Обработка NaN значений...")

        df = df.copy()
        self.nan_info["before"] = df.isna().sum().sum()

        try:
            # CompetitionDistance NaN = конкурента нет -> ставим 100к (дальше максимум в ~75к)
            if "CompetitionDistance" in df.columns:
                if "CompetitionOpenSinceYear" in df.columns:
                    df["HasCompetition"] = (~df["CompetitionOpenSinceYear"].isna()).astype(int)
                df["CompetitionDistance"] = df["CompetitionDistance"].fillna(100000)

            competition_columns = ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]
            for col in competition_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(0) # 0 = конкурентов нет

            # Promo2 NaN = магазин не участвует в расширенной программе
            promo2_columns = ["Promo2SinceWeek", "Promo2SinceYear"]

            if "Promo2SinceYear" in df.columns:
                df["InPromo2"] = (~df["Promo2SinceYear"].isna()).astype(int)

            for col in promo2_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(0)

            if "PromoInterval" in df.columns:
                df["PromoInterval"] = df["PromoInterval"].fillna("NoPromo")

            # Бинарные: мода по магазину
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

            # Числовые: медиана по магазину (чтобы не смешивать разные масштабы)
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

            # Категории: мода
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

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Year, Month, Week, DayOfWeek + sin/cos кодирование + сезоны + weekend"""
        self.logger.info("Создание временных признаков...")

        df = df.copy()

        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
        df["ISOYear"] = df["Date"].dt.isocalendar().year.astype(int)
        df["Day"] = df["Date"].dt.day
        df["DayOfYear"] = df["Date"].dt.dayofyear

        df["IsSaturday"] = (df["DayOfWeek"] == 6).astype(int)
        df["IsSunday"] = (df["DayOfWeek"] == 7).astype(int)
        df["IsWeekend"] = ((df["DayOfWeek"] >= 6)).astype(int)

        # sin/cos - чтобы декабрь(12) и январь(1) были рядом, а не на разных концах шкалы
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

        df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
        df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

        # День месяца - для зарплатных эффектов (пики в начале/конце месяца)
        df["DayOfMonth_sin"] = np.sin(2 * np.pi * df["Day"] / 31)
        df["DayOfMonth_cos"] = np.cos(2 * np.pi * df["Day"] / 31)

        # Rossman - Германия, сезонность влияет на трафик
        seasons = [(3, 5, "IsSpring"), (6, 8, "IsSummer"), (9, 11, "IsAutumn"), (12, 2, "IsWinter")]

        for start, end, name in seasons:
            if start <= end:
                df[name] = ((df["Month"] >= start) & (df["Month"] <= end)).astype(int)
            else:
                # Зима: декабрь + январь-февраль
                df[name] = ((df["Month"] >= start) | (df["Month"] <= end)).astype(int)

        df["IsMonthStart"] = (df["Day"] <= 5).astype(int)
        df["IsMonthEnd"] = (df["Day"] >= 25).astype(int)
        df["IsYearStart"] = ((df["Month"] == 1) & (df["Day"] <= 7)).astype(int)
        df["IsYearEnd"] = ((df["Month"] == 12) & (df["Day"] >= 25)).astype(int)

        return df

    def create_promo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """PromoSequence, длина серии, дни с последней акции, начало/конец"""
        self.logger.info("Создание промо-признаков...")

        df = df.sort_values(["Store", "Date"]).copy()

        # Нумерация промо-серий: новый ID при каждом переключении 0->1 или 1->0
        df["PromoSequence"] = df.groupby("Store")["Promo"].transform(lambda x: (x != x.shift()).cumsum())

        df["PromoSequenceLength"] = df.groupby(["Store", "PromoSequence"]).cumcount() + 1
        df["PromoSequenceLength"] = np.where(df["Promo"] == 1, df["PromoSequenceLength"], 0)

        df["DaysSinceLastPromo"] = self._calculate_days_since_last_promo(df)

        df["IsPromoStart"] = ((df["Promo"] == 1) & (df.groupby("Store")["Promo"].shift(1) == 0)).astype(int)
        df["IsPromoEnd"] = ((df["Promo"] == 0) & (df.groupby("Store")["Promo"].shift(1) == 1)).astype(int)

        return df

    def _calculate_days_since_last_promo(self, df: pd.DataFrame) -> pd.Series:
        """merge_asof по промо-датам - быстрее чем groupby+apply"""
        df = df.sort_values(["Store", "Date"])

        result = pd.Series(dtype=float, index=df.index)

        # Промо-день -> 0 дней с последней акции, очевидно
        is_promo = df["Promo"] == 1
        result[is_promo] = 0

        promo_records = df.loc[is_promo, ["Store", "Date"]].copy()
        promo_records = promo_records.rename(columns={"Date": "LastPromoDate"})
        promo_records = promo_records.sort_values(["Store", "LastPromoDate"])

        # Для не-промо записей ищем ближайшую промо-дату в прошлом
        non_promo = df.loc[~is_promo, ["Store", "Date"]].copy()
        non_promo["_orig_idx"] = non_promo.index
        non_promo = non_promo.sort_values(["Store", "Date"])

        if len(promo_records) > 0 and len(non_promo) > 0:
            merged = pd.merge_asof(
                non_promo.sort_values(["Date", "Store"]),
                promo_records.sort_values(["LastPromoDate", "Store"]),
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

    def create_lag_features(self, df: pd.DataFrame, lags: list = None) -> pd.DataFrame:
        """Лаги через полный календарь (MultiIndex) - shift() по дням, а не по строкам"""
        self.logger.info("Создание лаговых признаков...")

        if lags is None:
            lags = self.lag_days

        if df.empty:
            self.logger.warning("Предупреждение: пустой DataFrame")
            return df

        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

        # Полный календарь - без него shift(7) сдвинет на 7 строк, а не на 7 дней
        stores = df["Store"].unique()
        date_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
        full_idx = pd.MultiIndex.from_product([stores, date_range], names=["Store", "Date"])

        sales = (
            df.set_index(["Store", "Date"])["Sales"]
            .reindex(full_idx)
            .sort_index()
        )

        # shifted_sales - чтобы rolling не захватил текущий день
        shifted_sales = sales.groupby(level="Store").shift(1)

        lag_df = pd.DataFrame(index=full_idx)

        for lag in lags:
            lag_df[f"SalesLag_{lag}"] = sales.groupby(level="Store").shift(lag)

            grp = shifted_sales.groupby(level="Store")
            lag_df[f"RollingMean_{lag}"] = grp.transform(lambda x: x.rolling(lag, min_periods=1).mean())
            lag_df[f"RollingStd_{lag}"] = grp.transform(lambda x: x.rolling(lag, min_periods=1).std())
            lag_df[f"RollingMin_{lag}"] = grp.transform(lambda x: x.rolling(lag, min_periods=1).min())
            lag_df[f"RollingMax_{lag}"] = grp.transform(lambda x: x.rolling(lag, min_periods=1).max())

        # Мержим только лаговые колонки обратно (без Sales - она уже в df)
        lag_cols = list(lag_df.columns)
        results = df.merge(
            lag_df.reset_index()[["Store", "Date"] + lag_cols],
            on=["Store", "Date"],
            how="left"
        )

        result = self.fill_lag_missing_values(results)

        na_count = result[lag_cols].isna().sum().sum()
        self.logger.info(f"Лаговые признаки созданы. Осталось NaN: {na_count}")

        return result

    def fill_lag_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """expanding-статистики по магазину - чтобы не тянуть данные из будущего"""
        self.logger.info("Заполнение пропусков в лагах...")

        lag_columns = [col for col in df.columns if "Lag" in col or "Rolling" in col]

        initial_na_count = df[lag_columns].isna().sum().sum()
        self.logger.info(f"Найдено пропусков в лагах {initial_na_count}")

        df = df.sort_values(["Store", "Date"]).copy()

        _fill_strategies = {"SalesLag": "mean", "RollingMean": "mean", "RollingStd": "std", "RollingMin": "min", "RollingMax": "max"}

        for col in lag_columns:
            if df[col].isna().any():
                for pattern, agg_func in _fill_strategies.items():
                    if pattern in col:
                        df[col] = df.groupby("Store")[col].transform(
                            lambda x, fn=agg_func: x.fillna(
                                getattr(x.expanding(min_periods=1), fn)()
                            )
                        )
                        # Совсем новый магазин без истории -> 0
                        df[col] = df[col].fillna(0)
                        break

        final_na_count = df[lag_columns].isna().sum().sum()
        self.logger.info(f"Осталось пропусков: {final_na_count}")

        return df

    def create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Бинарные признаки (a/b/c из StateHoliday) + эффекты на соседние дни"""
        self.logger.info("Создание праздничных признаков...")

        df = df.copy()

        df["IsHoliday"] = (df["StateHoliday"] != "0").astype(int)
        df["IsPublicHoliday"] = (df["StateHoliday"] == "a").astype(int)
        df["IsEaster"] = (df["StateHoliday"] == "b").astype(int)
        df["IsChristmas"] = (df["StateHoliday"] == "c").astype(int)

        # shift по строкам, не по календарю - пропущенный день = закрытый = не праздник, ок
        df["WasHolidayYesterday"] = df.groupby("Store")["IsHoliday"].shift(1).fillna(0).astype(int)
        df["IsHolidayTomorrow"] = df.groupby("Store")["IsHoliday"].shift(-1).fillna(0).astype(int)

        df["WasHolidayLastWeek"] = df.groupby("Store")["IsHoliday"].transform(
            lambda x: x.rolling(7, min_periods=1).max().shift(1)
        ).fillna(0).astype(int)

        return df

    def create_store_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encoding StoreType и Assortment"""
        self.logger.info("Создание признаков магазинов...")

        df = df.copy()

        if "StoreType" in df.columns:
            store_type_dummies = pd.get_dummies(df["StoreType"], prefix="StoreType", dtype=int)
            df = pd.concat([df, store_type_dummies], axis=1)

        if "Assortment" in df.columns:
            assortment_dummies = pd.get_dummies(df["Assortment"], prefix="Assortment", dtype=int)
            df = pd.concat([df, assortment_dummies], axis=1)

        return df

    def create_competition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Месяцы с открытия конкурента + категоризация расстояния (near/medium/far)"""
        self.logger.info("Создание признаков конкуренции...")

        df = df.copy()

        required_cols = ["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth", "Year", "Month"]
        if all(col in df.columns for col in required_cols):
            df["CompetitionOpenMonths"] = (
            (df["Year"] - df["CompetitionOpenSinceYear"]) * 12 +
            (df["Month"] - df["CompetitionOpenSinceMonth"])
            )
            # < 0 значит конкурент еще не открылся
            df["CompetitionOpenMonths"] = df["CompetitionOpenMonths"].clip(lower=0)

            df["CompetitionOpen"] = (df["CompetitionOpenMonths"] > 0).astype(int)

        # Категоризация расстояния: <1km, 1-5km, >5km
        if "CompetitionDistance" in df.columns:
            df["CompetitionNear"] = (df["CompetitionDistance"] < 1000).astype(int)
            df["CompetitionMedium"] = (
                (df["CompetitionDistance"] >= 1000) &
                (df["CompetitionDistance"] < 5000)
            ).astype(int)
            df["CompetitionFar"] = (df["CompetitionDistance"] >= 5000).astype(int)

        return df

    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """Проверяет, что Store, Date, Sales и т.д. на месте"""
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")

        if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            raise ValueError("Колонка Date должна содержать данные типа datetime")

    def prepare_final_dataset(self, df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        """Весь пайплайн: NaN -> признаки -> лаги -> fillna -> drop exclude cols"""
        self.logger.info("Подготовка финального датасета...")
        self.logger.info(f"Исходные данные: {df.shape[0]} записей, {df.shape[1]} колонок")

        try:
            self._validate_input_data(df)

            df = self.handle_nan_values(df)

            df = self.create_temporal_features(df)
            df = self.create_promo_features(df)
            df = self.create_holiday_features(df)
            df = self.create_store_features(df)
            df = self.create_competition_features(df)
            df = self.create_lag_features(df)

            feature_columns = [col for col in df.columns if col not in self.EXCLUDE_COLUMNS]

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

    def get_feature_statistics(self) -> Dict[str, Any]:
        """Сколько признаков создано, по категориям"""
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
    setup_logging()

    cleaned_data = pd.read_csv(DATA_PATH / "processed/cleaned_data.csv", parse_dates=["Date"])

    engineer = FeatureEngineer()
    final_data, feature_names = engineer.prepare_final_dataset(cleaned_data)

    final_data.to_csv(DATA_PATH / "processed/final_dataset.csv", index=False)

    stats = engineer.get_feature_statistics()
    engineer.logger.info(f"Статистика обработки: {stats}")

    engineer.logger.info("Feature engineering успешно завершен")
