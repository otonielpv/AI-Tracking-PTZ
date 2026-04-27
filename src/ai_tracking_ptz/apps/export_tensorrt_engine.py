from __future__ import annotations

import argparse
import logging

from ultralytics import YOLO

from ai_tracking_ptz.logging_utils import configure_logging


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Ultralytics model to TensorRT engine")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to the source .pt model")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference/export image size")
    parser.add_argument("--device", default="cuda:0", help="Export device. Example: cuda:0")
    parser.add_argument("--half", action="store_true", help="Enable FP16 export when supported")
    parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace size in GB")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        model = YOLO(args.model)
        output_path = model.export(
            format="engine",
            imgsz=args.imgsz,
            device=args.device,
            half=args.half,
            workspace=args.workspace,
        )
        LOGGER.info("TensorRT export completed: %s", output_path)
        return 0
    except Exception:
        LOGGER.exception("TensorRT export failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
