from ultralytics import YOLO
from pathlib import Path

def main():
    model = YOLO("yolov8n.pt")

    root_dir = Path(__file__).resolve().parent
    project_dir = root_dir / "training_output"

    model.train(
        data=str(root_dir / "config.yaml"),
        epochs=30,
        imgsz=640,
        project=str(project_dir),  # 절대 경로로 전달
        name="mabi_model",
        exist_ok=True
    )

if __name__ == "__main__":
    main()
