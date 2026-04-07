"""
build_features.py
=================
Module tiền xử lý và tạo biến phái sinh cho dự án phân tích chuỗi thời gian
Elden Ring. Đảm bảo trục thời gian chính xác, tạo biến logarit, biến trễ (lag),
và biến tương tác phục vụ chiến lược Ablation Study (3 mô hình Dynamic OLS).

Input:  data/processed/EldenRing_Final_Merged.csv
Output: data/processed/EldenRing_Model_Ready.csv

Sử dụng:
    python -m src.features.build_features
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ==============================================================================
# Cấu hình Logger — tuân thủ quy tắc KHÔNG dùng print()
# ==============================================================================
logger = logging.getLogger(__name__)


# ==============================================================================
# Hằng số đường dẫn (pathlib)
# ==============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
INPUT_PATH: Path = PROJECT_ROOT / "data" / "processed" / "EldenRing_Final_Merged.csv"
OUTPUT_PATH: Path = PROJECT_ROOT / "data" / "processed" / "EldenRing_Model_Ready.csv"


# ==============================================================================
# 1. Đọc và chuẩn hóa trục thời gian
# ==============================================================================
def load_and_prepare_time_index(filepath: Path) -> pd.DataFrame:
    """Đọc dữ liệu thô, ép kiểu cột Date về datetime, sắp xếp theo thời gian
    và thiết lập Date làm Index.

    Args:
        filepath: Đường dẫn tuyệt đối tới file CSV đầu vào.

    Returns:
        DataFrame đã được sắp xếp theo thời gian với Date là Index.

    Raises:
        FileNotFoundError: Nếu file CSV không tồn tại.
        KeyError: Nếu cột 'Date' không có trong dữ liệu.
    """
    try:
        if not filepath.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file dữ liệu tại: {filepath}"
            )

        df: pd.DataFrame = pd.read_csv(filepath)
        logger.info("Đọc dữ liệu thành công: %d dòng, %d cột", *df.shape)

        # Ép kiểu Date về datetime
        df["Date"] = pd.to_datetime(df["Date"])
        logger.info("Đã ép kiểu cột 'Date' về datetime64")

        # Sắp xếp theo thời gian tăng dần
        df = df.sort_values("Date").reset_index(drop=True)
        logger.info("Đã sắp xếp dữ liệu theo thứ tự thời gian tăng dần")

        # Đặt Date làm Index
        df = df.set_index("Date")
        logger.info("Đã thiết lập cột 'Date' làm Index của DataFrame")

        return df

    except FileNotFoundError as e:
        logger.error("LỖI ĐỌC FILE: %s", e)
        raise
    except KeyError as e:
        logger.error("LỖI CỘT DỮ LIỆU: Không tìm thấy cột %s", e)
        raise
    except Exception as e:
        logger.error("LỖI KHÔNG XÁC ĐỊNH khi đọc dữ liệu: %s", e)
        raise


# ==============================================================================
# 2. Tạo biến Logarit tự nhiên
# ==============================================================================
def create_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các biến logarit tự nhiên để ổn định phương sai.

    Công thức:
        - Log_Player       = ln(Avg_Player + 1)
        - Log_Twitch_Avg   = ln(Twitch_Avg_Viewers + 1)
            (Nếu cột Twitch_Avg_Viewers không tồn tại, giữ nguyên
             Log_Twitch_Avg đã có sẵn trong dữ liệu gốc)

    Args:
        df: DataFrame đã có Index thời gian.

    Returns:
        DataFrame với các cột logarit được cập nhật/tạo mới.
    """
    try:
        # Tạo Log_Player từ Avg_Player
        if "Avg_Player" in df.columns:
            df["Log_Player"] = np.log(df["Avg_Player"] + 1)
            logger.info(
                "Đã tạo biến Log_Player = ln(Avg_Player + 1) — "
                "Min: %.4f, Max: %.4f",
                df["Log_Player"].min(),
                df["Log_Player"].max(),
            )
        else:
            logger.warning(
                "Không tìm thấy cột 'Avg_Player'. Giữ nguyên Log_Player gốc."
            )

        # Tạo Log_Twitch_Avg nếu có cột Twitch_Avg_Viewers
        if "Twitch_Avg_Viewers" in df.columns:
            df["Log_Twitch_Avg"] = np.log(df["Twitch_Avg_Viewers"] + 1)
            logger.info(
                "Đã tạo biến Log_Twitch_Avg = ln(Twitch_Avg_Viewers + 1)"
            )
        else:
            logger.info(
                "Cột 'Twitch_Avg_Viewers' không tồn tại — "
                "Giữ nguyên Log_Twitch_Avg đã có sẵn trong dữ liệu."
            )

        return df

    except Exception as e:
        logger.error("LỖI khi tạo biến logarit: %s", e)
        raise


# ==============================================================================
# 3. Tạo các biến trễ (Lag Features)
# ==============================================================================
def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các biến trễ (lagged variables) từ Log_Player.

    Biến được tạo:
        - Lag_Player: Giá trị Log_Player ở ngày hôm trước (shift 1).
        - Lag_7_Player: Giá trị Log_Player cách 7 ngày (shift 7).
          Dùng để bắt chu kỳ tuần (weekly seasonality).

    Args:
        df: DataFrame với cột Log_Player.

    Returns:
        DataFrame với thêm 2 cột Lag_Player và Lag_7_Player.

    Raises:
        KeyError: Nếu cột Log_Player không tồn tại.
    """
    try:
        if "Log_Player" not in df.columns:
            raise KeyError(
                "Cột 'Log_Player' không tồn tại — "
                "cần chạy create_log_features() trước."
            )

        # Lag 1 ngày — biến AR(1) cho mô hình Dynamic OLS
        df["Lag_Player"] = df["Log_Player"].shift(1)
        logger.info("Đã tạo biến Lag_Player = Log_Player.shift(1)")

        # Lag 7 ngày — bắt hiệu ứng cuối tuần/chu kỳ tuần
        df["Lag_7_Player"] = df["Log_Player"].shift(7)
        logger.info("Đã tạo biến Lag_7_Player = Log_Player.shift(7)")

        return df

    except KeyError as e:
        logger.error("LỖI BIẾN TRỄ: %s", e)
        raise
    except Exception as e:
        logger.error("LỖI KHÔNG XÁC ĐỊNH khi tạo biến trễ: %s", e)
        raise


# ==============================================================================
# 4. Tạo biến tương tác (Interaction Terms) cho Model 3
# ==============================================================================
def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các biến tương tác phục vụ phân tích kinh tế lượng nâng cao.

    Biến được tạo:
        - Interaction_Discount_Time:
            = Discount_Ratio × Years_Since_Release
            Ý nghĩa: Kiểm chứng giả thuyết "Stockpiling / Thịt đông lạnh" —
            liệu hiệu quả giảm giá có suy giảm theo thời gian hay không.

        - Interaction_Update_Time:
            = Is_Major_Update × Years_Since_Release
            Ý nghĩa: Đánh giá hiệu quả cập nhật nội dung khi game đã lão hóa.

    Args:
        df: DataFrame chứa các cột Discount_Ratio, Years_Since_Release,
            và Is_Major_Update.

    Returns:
        DataFrame với thêm 2 cột interaction.

    Raises:
        KeyError: Nếu thiếu cột cần thiết.
    """
    try:
        required_cols: list[str] = [
            "Discount_Ratio",
            "Years_Since_Release",
            "Is_Major_Update",
        ]
        missing: list[str] = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Thiếu các cột cần thiết để tạo biến tương tác: {missing}"
            )

        # Tương tác Giảm giá × Thời gian
        df["Interaction_Discount_Time"] = (
            df["Discount_Ratio"] * df["Years_Since_Release"]
        )
        logger.info(
            "Đã tạo biến Interaction_Discount_Time "
            "= Discount_Ratio × Years_Since_Release"
        )

        # Tương tác Cập nhật lớn × Thời gian
        df["Interaction_Update_Time"] = (
            df["Is_Major_Update"] * df["Years_Since_Release"]
        )
        logger.info(
            "Đã tạo biến Interaction_Update_Time "
            "= Is_Major_Update × Years_Since_Release"
        )

        return df

    except KeyError as e:
        logger.error("LỖI BIẾN TƯƠNG TÁC: %s", e)
        raise
    except Exception as e:
        logger.error("LỖI KHÔNG XÁC ĐỊNH khi tạo biến tương tác: %s", e)
        raise


# ==============================================================================
# 5. Xử lý giá trị NaN do shift() và lưu output
# ==============================================================================
def clean_and_save(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Xóa các dòng NaN phát sinh từ hàm shift() và lưu file CSV.

    Args:
        df: DataFrame đã hoàn tất feature engineering.
        output_path: Đường dẫn lưu file output.

    Returns:
        DataFrame đã được làm sạch NaN.
    """
    try:
        rows_before: int = len(df)
        df = df.dropna()
        rows_after: int = len(df)

        logger.info(
            "Đã xóa %d dòng NaN do shift() tạo ra "
            "(trước: %d → sau: %d dòng)",
            rows_before - rows_after,
            rows_before,
            rows_after,
        )

        # Đảm bảo thư mục cha tồn tại
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Lưu file CSV — giữ Index (Date) để downstream sử dụng
        df.to_csv(output_path, index=True)
        logger.info("Đã lưu dữ liệu Model-Ready tại: %s", output_path)

        return df

    except Exception as e:
        logger.error("LỖI khi xóa NaN hoặc lưu file: %s", e)
        raise


# ==============================================================================
# Pipeline chính
# ==============================================================================
def run_feature_engineering_pipeline(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Chạy toàn bộ pipeline tạo biến phái sinh cho Elden Ring.

    Pipeline gồm 5 bước:
        1. Đọc dữ liệu, ép kiểu Date, sắp xếp theo thời gian, đặt Date làm Index.
        2. Tạo biến Logarit: Log_Player, Log_Twitch_Avg.
        3. Tạo biến trễ: Lag_1_Player (shift 1), Lag_7_Player (shift 7).
        4. Tạo biến tương tác: Interaction_Discount_Time, Interaction_Update_Time.
        5. Xóa NaN do shift() và lưu file output.

    Args:
        input_path:  Đường dẫn file CSV đầu vào (mặc định: INPUT_PATH).
        output_path: Đường dẫn file CSV đầu ra (mặc định: OUTPUT_PATH).

    Returns:
        DataFrame sạch, sẵn sàng đưa vào mô hình hồi quy.
    """
    _input: Path = input_path or INPUT_PATH
    _output: Path = output_path or OUTPUT_PATH

    logger.info("=" * 70)
    logger.info("GIAI ĐOẠN 2: XÂY DỰNG BIẾN PHÁI SINH (Feature Engineering)")
    logger.info("=" * 70)
    logger.info("Input:  %s", _input)
    logger.info("Output: %s", _output)

    # Bước 1 — Đọc và chuẩn hóa trục thời gian
    df: pd.DataFrame = load_and_prepare_time_index(_input)

    # Bước 2 — Tạo biến logarit tự nhiên
    df = create_log_features(df)

    # Bước 3 — Tạo biến trễ (Lag)
    df = create_lag_features(df)

    # Bước 4 — Tạo biến tương tác (Interaction)
    df = create_interaction_features(df)

    # Bước 5 — Xóa NaN và lưu output
    df = clean_and_save(df, _output)

    logger.info("=" * 70)
    logger.info("HOÀN TẤT — Dữ liệu Model-Ready: %d dòng, %d cột", *df.shape)
    logger.info("Danh sách cột: %s", df.columns.tolist())
    logger.info("=" * 70)

    return df


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
    run_feature_engineering_pipeline()
