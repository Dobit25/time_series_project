"""
visualize.py
============
Module trích xuất Business Insights và vẽ biểu đồ chẩn đoán mô hình
cho dự án phân tích chuỗi thời gian Elden Ring.

Chức năng:
    1. Ghép 3 mô hình OLS thành bảng so sánh có dấu sao (*) bằng summary_col.
    2. Trích xuất Business Insights từ hệ số hồi quy Model 3.
    3. Kiểm chứng giả thuyết Stockpiling ("Thịt đông lạnh").
    4. Vẽ biểu đồ Actual vs Fitted (Train + Test prediction).

Input:  Models (.pkl), Train data, Test data
Output: reports/figures/EldenRing_Actual_vs_Fitted.png

Sử dụng:
    python -m src.visualization.visualize
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from statsmodels.regression.linear_model import RegressionResultsWrapper

# ==============================================================================
# Cấu hình Logger — tuân thủ quy tắc KHÔNG dùng print()
# ==============================================================================
logger = logging.getLogger(__name__)


# ==============================================================================
# Hằng số đường dẫn (pathlib)
# ==============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
MODEL_DIR: Path = PROJECT_ROOT / "models"
DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR: Path = PROJECT_ROOT / "reports" / "figures"
OUTPUT_FIGURE: Path = FIGURE_DIR / "EldenRing_Actual_vs_Fitted.png"


# ==============================================================================
# 1. Tải mô hình và dữ liệu
# ==============================================================================
def load_trained_models(model_dir: Path) -> dict[str, RegressionResultsWrapper]:
    """Tải 3 mô hình OLS đã huấn luyện từ file .pkl.

    Args:
        model_dir: Thư mục chứa các file .pkl.

    Returns:
        Dictionary chứa 3 mô hình đã tải.

    Raises:
        FileNotFoundError: Nếu file mô hình không tồn tại.
    """
    try:
        model_files: dict[str, str] = {
            "model_1": "EldenRing_Model1_Base.pkl",
            "model_2": "EldenRing_Model2_HypeContent.pkl",
            "model_3": "EldenRing_Model3_Interaction.pkl",
        }

        models: dict[str, RegressionResultsWrapper] = {}
        for key, filename in model_files.items():
            filepath: Path = model_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy mô hình: {filepath}"
                )
            with open(filepath, "rb") as f:
                models[key] = pickle.load(f)
            logger.info("Đã tải mô hình: %s", filepath.name)

        return models

    except FileNotFoundError as e:
        logger.error("LỖI TẢI MÔ HÌNH: %s", e)
        raise
    except Exception as e:
        logger.error("LỖI KHÔNG XÁC ĐỊNH khi tải mô hình: %s", e)
        raise


def load_train_test_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tải tập Train và Test từ CSV.

    Args:
        data_dir: Thư mục chứa file CSV.

    Returns:
        Tuple (train_df, test_df).
    """
    try:
        train_path: Path = data_dir / "EldenRing_Train.csv"
        test_path: Path = data_dir / "EldenRing_Test.csv"

        train_df: pd.DataFrame = pd.read_csv(
            train_path, parse_dates=["Date"], index_col="Date"
        )
        test_df: pd.DataFrame = pd.read_csv(
            test_path, parse_dates=["Date"], index_col="Date"
        )

        logger.info(
            "Đã tải dữ liệu — Train: %d dòng | Test: %d dòng",
            len(train_df), len(test_df),
        )
        return train_df, test_df

    except Exception as e:
        logger.error("LỖI khi tải dữ liệu Train/Test: %s", e)
        raise


# ==============================================================================
# 2. Bảng so sánh 3 mô hình (summary_col)
# ==============================================================================
def generate_comparison_table(
    models: dict[str, RegressionResultsWrapper],
) -> str:
    """Ghép 3 mô hình OLS thành bảng so sánh có dấu sao (*) significance.

    Sử dụng statsmodels.iolib.summary2.summary_col để tạo bảng regression
    chuẩn học thuật với R², R² adj, AIC, BIC, N, và significance stars.

    Args:
        models: Dictionary chứa 3 mô hình OLS.

    Returns:
        Chuỗi text của bảng so sánh.
    """
    try:
        table = summary_col(
            results=[models["model_1"], models["model_2"], models["model_3"]],
            model_names=[
                "Model 1\n(Base)",
                "Model 2\n(Hype & Content)",
                "Model 3\n(Interaction)",
            ],
            stars=True,
            float_format="%.4f",
            info_dict={
                "N": lambda x: f"{int(x.nobs)}",
                "R²": lambda x: f"{x.rsquared:.4f}",
                "R² adj": lambda x: f"{x.rsquared_adj:.4f}",
                "AIC": lambda x: f"{x.aic:.2f}",
                "BIC": lambda x: f"{x.bic:.2f}",
            },
        )

        table_str: str = table.as_text()
        logger.info("=" * 70)
        logger.info("BẢNG SO SÁNH 3 MÔ HÌNH OLS (Ablation Study)")
        logger.info("=" * 70)
        # Ghi log từng dòng của bảng
        for line in table_str.split("\n"):
            logger.info(line)
        logger.info("=" * 70)

        return table_str

    except Exception as e:
        logger.error("LỖI khi tạo bảng so sánh mô hình: %s", e)
        raise


# ==============================================================================
# 3. Trích xuất Business Insights
# ==============================================================================
def extract_business_insights(
    model_3: RegressionResultsWrapper,
) -> dict[str, float]:
    """Trích xuất và ghi log các Business Insights từ Model 3.

    Phân tích:
        a) Hiệu ứng giảm giá (Discount_Ratio coefficient).
        b) Giả thuyết Stockpiling (Interaction_Discount_Time coefficient).

    Args:
        model_3: Mô hình OLS Model 3 (Interaction) đã fit.

    Returns:
        Dictionary chứa các hệ số quan trọng.
    """
    try:
        params = model_3.params
        pvalues = model_3.pvalues
        insights: dict[str, float] = {}

        logger.info("=" * 70)
        logger.info("BUSINESS INSIGHTS — ELDEN RING (Từ Model 3 Interaction)")
        logger.info("=" * 70)

        # --- Insight 1: Hiệu ứng Giảm giá ---
        if "Discount_Ratio" in params.index:
            coeff_discount: float = params["Discount_Ratio"]
            pval_discount: float = pvalues["Discount_Ratio"]
            insights["Discount_Ratio"] = coeff_discount

            # Diễn giải: Vì Y = Log_Player, hệ số β cho biết
            # thay đổi 1 đơn vị Discount_Ratio (0→1 = 0%→100%)
            # → thay đổi β đơn vị trong Log_Player
            # → %Δ Player ≈ β × 100 nếu tính gần đúng
            logger.info(
                "Business Insight: Cứ giảm giá 1%% thì lượng người chơi "
                "của Elden Ring tăng %.4f%% (hệ số = %.6f, p-value = %.4f)",
                coeff_discount * 100,
                coeff_discount,
                pval_discount,
            )
        else:
            logger.warning("Không tìm thấy hệ số Discount_Ratio trong Model 3.")

        # --- Insight 2: Giả thuyết Stockpiling (Thịt đông lạnh) ---
        if "Interaction_Discount_Time" in params.index:
            coeff_interaction: float = params["Interaction_Discount_Time"]
            pval_interaction: float = pvalues["Interaction_Discount_Time"]
            insights["Interaction_Discount_Time"] = coeff_interaction

            logger.info("-" * 60)
            if coeff_interaction < 0:
                logger.info(
                    "Kiểm chứng Stockpiling — Hệ số Interaction_Discount_Time "
                    "= %.6f (< 0, p-value = %.4f)",
                    coeff_interaction,
                    pval_interaction,
                )
                logger.info(
                    "KẾT LUẬN: Chứng minh thành công giả thuyết Thịt đông lạnh: "
                    "Game để càng lâu, giảm giá càng kém hiệu quả "
                    "do người chơi đã mua găm hàng"
                )
            else:
                logger.info(
                    "Kiểm chứng Stockpiling — Hệ số Interaction_Discount_Time "
                    "= %.6f (>= 0, p-value = %.4f)",
                    coeff_interaction,
                    pval_interaction,
                )
                logger.info(
                    "KẾT LUẬN: KHÔNG xác nhận giả thuyết Thịt đông lạnh. "
                    "Hệ số >= 0 cho thấy hiệu quả giảm giá KHÔNG suy giảm "
                    "theo thời gian, hoặc thậm chí tăng lên."
                )
        else:
            logger.warning(
                "Không tìm thấy hệ số Interaction_Discount_Time trong Model 3."
            )

        logger.info("=" * 70)
        return insights

    except Exception as e:
        logger.error("LỖI khi trích xuất Business Insights: %s", e)
        raise


# ==============================================================================
# 4. Vẽ biểu đồ Actual vs Fitted (Train + Test)
# ==============================================================================
def plot_actual_vs_fitted(
    models: dict[str, RegressionResultsWrapper],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Path = OUTPUT_FIGURE,
) -> None:
    """Vẽ biểu đồ Log_Player thực tế vs Fitted Values của Model 3.

    Chi tiết biểu đồ:
        - Đường thực tế (Actual) Log_Player toàn thời gian.
        - Đường Fitted Values gộp cả đoạn Train (in-sample) và
          đoạn Predict trên Test (out-of-sample).
        - Đường ranh giới dọc (Vertical line) đánh dấu điểm bắt đầu tập Test.
        - Lưu hình ảnh chuẩn dpi=300.

    Args:
        models: Dictionary chứa 3 mô hình OLS.
        train_df: Tập Train.
        test_df: Tập Test.
        output_path: Đường dẫn lưu hình ảnh.
    """
    try:
        model_3: RegressionResultsWrapper = models["model_3"]
        feature_cols: list[str] = [
            "Lag_Player", "Lag_7_Player", "Discount_Ratio", "Is_Weekend",
            "Years_Since_Release", "Log_Twitch_Avg", "Trend_Index",
            "Is_Major_Update", "Is_Minor_Update",
            "Interaction_Discount_Time", "Interaction_Update_Time",
        ]

        # --- Fitted values trên tập Train (In-sample) ---
        X_train: pd.DataFrame = sm.add_constant(
            train_df[feature_cols], has_constant="add"
        )
        fitted_train: pd.Series = model_3.predict(X_train)

        # --- Predicted values trên tập Test (Out-of-sample) ---
        X_test: pd.DataFrame = sm.add_constant(
            test_df[feature_cols], has_constant="add"
        )
        predicted_test: pd.Series = model_3.predict(X_test)

        # --- Ghép Actual toàn bộ ---
        actual_full: pd.Series = pd.concat([
            train_df["Log_Player"], test_df["Log_Player"]
        ])

        # --- Ghép Fitted/Predicted toàn bộ ---
        fitted_full: pd.Series = pd.concat([fitted_train, predicted_test])

        # --- Điểm bắt đầu tập Test ---
        test_start_date: pd.Timestamp = test_df.index.min()

        # --- Vẽ biểu đồ ---
        fig, ax = plt.subplots(figsize=(16, 7))

        # Đường thực tế
        ax.plot(
            actual_full.index,
            actual_full.values,
            color="#2196F3",
            linewidth=1.0,
            alpha=0.85,
            label="Thực tế (Log_Player)",
        )

        # Đường Fitted/Predicted
        ax.plot(
            fitted_full.index,
            fitted_full.values,
            color="#FF5722",
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
            label="Model 3 — Fitted (Train) + Predicted (Test)",
        )

        # Đường ranh giới dọc — Test bắt đầu
        ax.axvline(
            x=test_start_date,
            color="#4CAF50",
            linewidth=2.0,
            linestyle="-.",
            alpha=0.8,
            label=f"Bắt đầu tập Test ({test_start_date.strftime('%Y-%m-%d')})",
        )

        # Vùng shade cho Test
        ax.axvspan(
            test_start_date,
            actual_full.index.max(),
            alpha=0.08,
            color="#4CAF50",
        )

        # Định dạng biểu đồ
        ax.set_title(
            "Elden Ring — Actual vs Fitted/Predicted (Model 3 Interaction)",
            fontsize=15,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("Ngày", fontsize=12)
        ax.set_ylabel("Log(Số người chơi trung bình + 1)", fontsize=12)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Định dạng trục thời gian
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate(rotation=45)

        plt.tight_layout()

        # Lưu hình ảnh
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info("Đã lưu biểu đồ Actual vs Fitted: %s", output_path)

    except Exception as e:
        logger.error("LỖI khi vẽ biểu đồ: %s", e)
        raise


# ==============================================================================
# Pipeline chính
# ==============================================================================
def run_visualization_pipeline(
    model_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    figure_path: Optional[Path] = None,
) -> None:
    """Chạy toàn bộ pipeline trực quan hóa và trích xuất insights.

    Pipeline gồm:
        1. Tải 3 mô hình và dữ liệu Train/Test.
        2. Tạo bảng so sánh 3 mô hình (summary_col có dấu sao).
        3. Trích xuất Business Insights (Discount effect, Stockpiling).
        4. Vẽ và lưu biểu đồ Actual vs Fitted.

    Args:
        model_dir: Thư mục chứa mô hình (mặc định: MODEL_DIR).
        data_dir: Thư mục chứa dữ liệu (mặc định: DATA_DIR).
        figure_path: Đường dẫn lưu hình ảnh (mặc định: OUTPUT_FIGURE).
    """
    _model_dir: Path = model_dir or MODEL_DIR
    _data_dir: Path = data_dir or DATA_DIR
    _figure_path: Path = figure_path or OUTPUT_FIGURE

    logger.info("=" * 70)
    logger.info("GIAI ĐOẠN 4: TRỰC QUAN HÓA & BUSINESS INSIGHTS")
    logger.info("=" * 70)

    # Bước 1 — Tải mô hình và dữ liệu
    models: dict[str, RegressionResultsWrapper] = load_trained_models(_model_dir)
    train_df, test_df = load_train_test_data(_data_dir)

    # Bước 2 — Bảng so sánh 3 mô hình
    generate_comparison_table(models)

    # Bước 3 — Business Insights
    extract_business_insights(models["model_3"])

    # Bước 4 — Vẽ biểu đồ
    plot_actual_vs_fitted(models, train_df, test_df, _figure_path)

    logger.info("=" * 70)
    logger.info("HOÀN TẤT — Pipeline trực quan hóa đã hoàn thành")
    logger.info("=" * 70)


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
    run_visualization_pipeline()
