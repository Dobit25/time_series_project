"""
train_model.py
==============
Module huấn luyện 3 mô hình Dynamic OLS theo chiến lược Ablation Study
cho dự án phân tích chuỗi thời gian Elden Ring.

Chiến lược ngăn chặn rò rỉ dữ liệu (Data Leakage Prevention):
    - Chia Train/Test theo thời gian thực tế (Chronological Split).
    - 30 dòng cuối cùng (theo thời gian) → tập Test.
    - Mọi mô hình CHỈ được huấn luyện trên tập Train.

Các mô hình:
    - Model 1 (Base):        Lag_Player, Discount_Ratio, Is_Weekend, Years_Since_Release
    - Model 2 (Hype/Content): Model 1 + Log_Twitch_Avg, Trend_Index, Is_Major_Update, Is_Minor_Update
    - Model 3 (Interaction):  Model 2 + Interaction_Discount_Time, Interaction_Update_Time

Input:  data/processed/EldenRing_Model_Ready.csv
Output: data/processed/EldenRing_Train.csv
        data/processed/EldenRing_Test.csv
        models/EldenRing_Model1_Base.pkl
        models/EldenRing_Model2_HypeContent.pkl
        models/EldenRing_Model3_Interaction.pkl

Sử dụng:
    python -m src.models.train_model
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

# ==============================================================================
# Cấu hình Logger — tuân thủ quy tắc KHÔNG dùng print()
# ==============================================================================
logger = logging.getLogger(__name__)


# ==============================================================================
# Hằng số đường dẫn (pathlib)
# ==============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
INPUT_PATH: Path = PROJECT_ROOT / "data" / "processed" / "EldenRing_Model_Ready.csv"
OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODEL_DIR: Path = PROJECT_ROOT / "models"

# Số lượng dòng cuối cùng dùng làm tập Test
TEST_SIZE: int = 90


# ==============================================================================
# Định nghĩa biến cho 3 mô hình (Ablation Study)
# ==============================================================================

# Model 1 — Base: Các yếu tố nền tảng
MODEL_1_FEATURES: list[str] = [
    "Lag_Player",
    "Lag_7_Player",
    "Discount_Ratio",
    "Is_Weekend",
    "Years_Since_Release",
]

# Model 2 — Hype & Content: Base + hiệu ứng truyền thông và nội dung
MODEL_2_FEATURES: list[str] = MODEL_1_FEATURES + [
    "Log_Twitch_Avg",
    "Trend_Index",
    "Is_Major_Update",
    "Is_Minor_Update",
]

# Model 3 — Interaction: Model 2 + tương tác phi tuyến
MODEL_3_FEATURES: list[str] = MODEL_2_FEATURES + [
    "Interaction_Discount_Time",
    "Interaction_Update_Time",
]

# Biến phụ thuộc (Y)
TARGET_COL: str = "Log_Player"


# ==============================================================================
# 1. Đọc dữ liệu Model-Ready
# ==============================================================================
def load_model_ready_data(filepath: Path) -> pd.DataFrame:
    """Đọc dữ liệu đã qua Feature Engineering.

    Args:
        filepath: Đường dẫn tới file EldenRing_Model_Ready.csv

    Returns:
        DataFrame với Date làm Index.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
    """
    try:
        if not filepath.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file Model-Ready tại: {filepath}. "
                "Vui lòng chạy build_features.py trước."
            )

        df: pd.DataFrame = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
        logger.info("Đọc dữ liệu Model-Ready thành công: %d dòng, %d cột", *df.shape)
        return df

    except FileNotFoundError as e:
        logger.error("LỖI ĐỌC FILE: %s", e)
        raise
    except Exception as e:
        logger.error("LỖI KHÔNG XÁC ĐỊNH khi đọc dữ liệu: %s", e)
        raise


# ==============================================================================
# 2. Tách Train/Test theo thời gian (Chronological Split)
# ==============================================================================
def chronological_train_test_split(
    df: pd.DataFrame, test_size: int = TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia dữ liệu thành tập Train và Test theo thứ tự thời gian.

    QUAN TRỌNG — Ngăn chặn rò rỉ dữ liệu:
        - Dữ liệu PHẢI được sắp xếp theo thời gian TRƯỚC khi chia.
        - 'test_size' dòng CUỐI CÙNG → tập Test.
        - Phần còn lại → tập Train.

    Args:
        df: DataFrame đã sắp xếp theo thời gian.
        test_size: Số dòng cuối cùng dùng làm tập Test.

    Returns:
        Tuple (train_df, test_df).

    Raises:
        ValueError: Nếu test_size >= tổng số dòng.
    """
    try:
        n_total: int = len(df)
        if test_size >= n_total:
            raise ValueError(
                f"test_size ({test_size}) phải nhỏ hơn tổng số dòng ({n_total})"
            )

        # Đảm bảo dữ liệu đã sắp xếp theo thời gian
        df = df.sort_index()

        train_df: pd.DataFrame = df.iloc[:-test_size].copy()
        test_df: pd.DataFrame = df.iloc[-test_size:].copy()

        logger.info(
            "Chia Train/Test theo thời gian — "
            "Train: %d dòng (%s → %s) | Test: %d dòng (%s → %s)",
            len(train_df),
            train_df.index.min().strftime("%Y-%m-%d"),
            train_df.index.max().strftime("%Y-%m-%d"),
            len(test_df),
            test_df.index.min().strftime("%Y-%m-%d"),
            test_df.index.max().strftime("%Y-%m-%d"),
        )

        return train_df, test_df

    except ValueError as e:
        logger.error("LỖI CHIA DỮ LIỆU: %s", e)
        raise
    except Exception as e:
        logger.error("LỖI KHÔNG XÁC ĐỊNH khi chia Train/Test: %s", e)
        raise


# ==============================================================================
# 3. Lưu tập Train và Test
# ==============================================================================
def save_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Lưu tập Train và Test ra file CSV.

    Args:
        train_df: Tập huấn luyện.
        test_df: Tập kiểm tra.
        output_dir: Thư mục đầu ra.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path: Path = output_dir / "EldenRing_Train.csv"
        test_path: Path = output_dir / "EldenRing_Test.csv"

        train_df.to_csv(train_path, index=True)
        test_df.to_csv(test_path, index=True)

        logger.info("Đã lưu tập Train: %s (%d dòng)", train_path, len(train_df))
        logger.info("Đã lưu tập Test:  %s (%d dòng)", test_path, len(test_df))

    except Exception as e:
        logger.error("LỖI khi lưu Train/Test CSV: %s", e)
        raise


# ==============================================================================
# 4. Huấn luyện mô hình Dynamic OLS với HC3
# ==============================================================================
def train_ols_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_name: str,
) -> RegressionResultsWrapper:
    """Huấn luyện một mô hình OLS với robust standard errors (HC3).

    Bước thực hiện:
        1. Thêm hằng số (intercept) bằng sm.add_constant().
        2. Fit mô hình OLS CHỈ trên tập Train.
        3. Sử dụng cov_type='HC3' để xử lý phương sai sai số thay đổi
           (heteroscedasticity-robust).

    Args:
        train_df: Tập Train.
        feature_cols: Danh sách tên cột biến độc lập (X).
        target_col: Tên cột biến phụ thuộc (Y).
        model_name: Tên định danh của mô hình (dùng cho logging).

    Returns:
        Object kết quả hồi quy OLS đã fit.

    Raises:
        KeyError: Nếu thiếu cột biến.
    """
    try:
        # Kiểm tra cột biến độc lập
        missing_features: list[str] = [
            c for c in feature_cols if c not in train_df.columns
        ]
        if missing_features:
            raise KeyError(
                f"[{model_name}] Thiếu cột biến độc lập: {missing_features}"
            )

        # Kiểm tra biến phụ thuộc
        if target_col not in train_df.columns:
            raise KeyError(
                f"[{model_name}] Thiếu biến phụ thuộc: '{target_col}'"
            )

        logger.info("-" * 60)
        logger.info("Đang huấn luyện %s...", model_name)
        logger.info("Biến độc lập (X): %s", feature_cols)
        logger.info("Biến phụ thuộc (Y): %s", target_col)

        # Chuẩn bị ma trận X (thêm hằng số) và vector Y
        X_train: pd.DataFrame = sm.add_constant(train_df[feature_cols])
        y_train: pd.Series = train_df[target_col]

        # Fit OLS với HC3 robust standard errors
        model: RegressionResultsWrapper = sm.OLS(y_train, X_train).fit(
            cov_type="HC3"
        )

        logger.info(
            "[%s] R² = %.6f | R² hiệu chỉnh = %.6f | AIC = %.2f | BIC = %.2f",
            model_name,
            model.rsquared,
            model.rsquared_adj,
            model.aic,
            model.bic,
        )
        logger.info("[%s] Số quan sát: %d", model_name, int(model.nobs))

        return model

    except KeyError as e:
        logger.error("LỖI BIẾN: %s", e)
        raise
    except Exception as e:
        logger.error("LỖI khi huấn luyện %s: %s", model_name, e)
        raise


# ==============================================================================
# 5. Lưu mô hình đã huấn luyện (.pkl)
# ==============================================================================
def save_model(
    model: RegressionResultsWrapper,
    model_dir: Path,
    filename: str,
) -> Path:
    """Lưu object mô hình OLS đã huấn luyện ra file pickle.

    Args:
        model: Object kết quả OLS đã fit.
        model_dir: Thư mục lưu mô hình.
        filename: Tên file (bao gồm đuôi .pkl).

    Returns:
        Đường dẫn tuyệt đối tới file mô hình đã lưu.
    """
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path: Path = model_dir / filename

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info("Đã lưu mô hình: %s", model_path)
        return model_path

    except Exception as e:
        logger.error("LỖI khi lưu mô hình %s: %s", filename, e)
        raise


# ==============================================================================
# Pipeline chính — Ablation Study
# ==============================================================================
def run_training_pipeline(
    input_path: Path | None = None,
    output_dir: Path | None = None,
    model_dir: Path | None = None,
) -> dict[str, RegressionResultsWrapper]:
    """Chạy toàn bộ pipeline huấn luyện 3 mô hình Dynamic OLS.

    Pipeline gồm các bước:
        1. Đọc dữ liệu Model-Ready.
        2. Chia Train/Test theo thời gian (30 dòng cuối → Test).
        3. Huấn luyện 3 mô hình OLS (Ablation Study) CHỈ trên tập Train.
        4. Lưu tập Train/Test ra CSV và mô hình ra .pkl.

    Args:
        input_path: Đường dẫn file đầu vào (mặc định: INPUT_PATH).
        output_dir: Thư mục lưu CSV (mặc định: OUTPUT_DIR).
        model_dir: Thư mục lưu mô hình (mặc định: MODEL_DIR).

    Returns:
        Dictionary chứa 3 mô hình đã huấn luyện.
    """
    _input: Path = input_path or INPUT_PATH
    _output_dir: Path = output_dir or OUTPUT_DIR
    _model_dir: Path = model_dir or MODEL_DIR

    logger.info("=" * 70)
    logger.info(
        "GIAI ĐOẠN 3: HUẤN LUYỆN MÔ HÌNH DYNAMIC OLS (Ablation Study)"
    )
    logger.info("=" * 70)

    # Bước 1 — Đọc dữ liệu
    df: pd.DataFrame = load_model_ready_data(_input)

    # Bước 2 — Chia Train/Test theo thời gian
    train_df, test_df = chronological_train_test_split(df, test_size=TEST_SIZE)

    # Bước 3 — Lưu tập Train/Test ra CSV
    save_train_test(train_df, test_df, _output_dir)

    # Bước 4 — Huấn luyện 3 mô hình (CHỈ trên tập Train)
    models: dict[str, RegressionResultsWrapper] = {}

    # --- Model 1: Base ---
    model_1 = train_ols_model(
        train_df, MODEL_1_FEATURES, TARGET_COL,
        model_name="Model 1 (Base)"
    )
    save_model(model_1, _model_dir, "EldenRing_Model1_Base.pkl")
    models["model_1"] = model_1

    # --- Model 2: Hype & Content ---
    model_2 = train_ols_model(
        train_df, MODEL_2_FEATURES, TARGET_COL,
        model_name="Model 2 (Hype & Content)"
    )
    save_model(model_2, _model_dir, "EldenRing_Model2_HypeContent.pkl")
    models["model_2"] = model_2

    # --- Model 3: Interaction ---
    model_3 = train_ols_model(
        train_df, MODEL_3_FEATURES, TARGET_COL,
        model_name="Model 3 (Interaction)"
    )
    save_model(model_3, _model_dir, "EldenRing_Model3_Interaction.pkl")
    models["model_3"] = model_3

    logger.info("=" * 70)
    logger.info("HOÀN TẤT — Đã huấn luyện và lưu thành công 3 mô hình OLS")
    logger.info("=" * 70)

    return models


# ==============================================================================
# Entry Point
# ==============================================================================
if __name__ == "__main__":
    # Cấu hình logging cơ bản — dùng UTF-8 để hỗ trợ tiếng Việt trên Windows
    import io

    handler = logging.StreamHandler(
        io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    run_training_pipeline()
