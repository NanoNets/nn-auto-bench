from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from nnautobench.config.config import MODEL_CONFIGS
from nnautobench.inference.predictor import Predictor
from nnautobench.models import get_model
from nnautobench.utils.common_utils import load_data

load_dotenv()  # Load environment variables at the start of benchmark.py

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_benchmark(
    model_name,
    input_file=None,
    max_workers=16,
    few_shot=0,
    layout="default",
    conf_score_method="prob",
    limit=None,
):
    start_time = datetime.now()
    logger.info(f"Starting benchmark for model: {model_name}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Max workers: {max_workers}")
    logger.info(f"Num of fewshot examples: {few_shot}")
    logger.info(f"Layout: {layout}")
    try:
        model_config = MODEL_CONFIGS[model_name]
        model_class = get_model(model_name)
        if model_class is None:
            logger.error(f"Unknown model: {model_name}")
            raise ValueError(f"Unknown model: {model_name}")

        logger.info(f"Initializing {model_name} model")
        model = model_class(
            model_config["model_name"],
            model_config["api_base"],
        )
        predictor = Predictor(model, conf_score_method)

        logger.info("Loading and filtering data")
        dataset_name = ""
        if input_file:
            df = load_data(input_file)
            if limit is None:
                df_filtered = df
            elif limit > len(df):
                logger.warning(
                    f"{limit} is greater than the number of samples in the dataset. Using the entire dataset",
                )
                df_filtered = df
            else:
                df_filtered = df.head(limit)

            dataset_name = os.path.basename(input_file).split(".")[0]

        else:
            raise ValueError("Either input_file or model_id must be provided")
        logger.info(f"Filtered data shape: {df_filtered.shape}")

        results = []
        logger.info(f"Starting prediction with {max_workers} workers")
        pred_start_time = time.perf_counter()
        assert few_shot == 1, "Only oneshot dataset is supported for now!"
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    predictor.process_single_image,
                    row["image_path"],
                    row["accepted"],
                    few_shot,
                    (
                        []
                        if few_shot == 0
                        else [
                            {
                                "text": row[f"ctx_{i}"],
                                # if i!=1 else row[f'annotated_image_path'] ,
                                "image_path": row[f"ctx_{i}_image_path"],
                                "accepted": row[f"ctx_{i}_accepted"],
                            }
                            for i in range(1, few_shot + 1)
                        ]
                    ),
                    row["content"],  # change this
                    row["Queried_labels"],
                    layout,
                ): i
                for i, (_, row) in enumerate(df_filtered.iterrows())
            }

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(df_filtered),
                desc="Processing images",
            ):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    logger.error(
                        f"Image generated an exception: {exc}",
                        exc_info=True,
                    )

        pred_end_time = time.perf_counter()
        logger.info(
            f"Prediction completed in {pred_end_time - pred_start_time:.2f} seconds",
        )

        logger.info("Saving results to DataFrame")
        df_results = pd.DataFrame(results)

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/benchmark_results_{model_name}_{dataset_name}_{'field'}_{current_time}.jsonl"
        df_results.to_json(output_file, orient="records", lines=True)
        logger.info(
            f"Results saved to {output_file} of shape {df_results.shape}",
        )

        logger.info(f"Results for {model_name}:")
        avg_accuracy = df_results.file_accuracy.mean()
        parsing_accuracy = df_results.parsing_accuracy.mean()

        correct_approved = df_results["correct_approved"].sum()
        incorrect_approved = df_results["incorrect_approved"].sum()
        total_fields = df_results["total_fields"].sum()
        logger.info(
            f"Approval Accuracy: {correct_approved/(correct_approved+incorrect_approved):.4f}, Approval Rate: {(correct_approved + incorrect_approved)/total_fields:.4f}",
        )
        logger.info(f"Parsing Accuracy: {parsing_accuracy:.4f}")
        logger.info(f"Average Accuracy: {avg_accuracy:.4f}")

    finally:
        pass

    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Benchmark completed in {duration}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmark for a specific model",
    )
    parser.add_argument(
        "model_name",
        choices=MODEL_CONFIGS.keys(),
        help="Name of the model to benchmark",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_file", help="Path to the input JSONL file")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Maximum number of worker threads (default: 16)",
    )
    parser.add_argument(
        "--few_shot",
        type=int,
        default=1,
        help="Number of fewshot examples (default: 1 i.e oneshot)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="default",
        help="Layout of the input text (default: default)",
    )
    parser.add_argument(
        "--conf_score_method",
        type=str,
        default="prob",
        help="Confidence score method (default: prob, yes_no, nanonets, consistency)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to process",
    )
    args = parser.parse_args()

    logger.info("Starting benchmark script")
    logger.info(f"Arguments: {args}")

    try:
        run_benchmark(
            args.model_name,
            input_file=args.input_file,
            max_workers=args.max_workers,
            few_shot=args.few_shot,
            layout=args.layout,
            conf_score_method=args.conf_score_method,
            limit=args.limit,
        )
    except Exception as e:
        logger.exception("An error occurred during benchmark execution")

    logger.info("Benchmark script completed")
