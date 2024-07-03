import cv2
import os
import pandas as pd
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

# Azure Custom Vision configuration
project_id = "your_project_id"
ENDPOINT = "your_endpoint"
prediction_key = "your_prediction_key"

credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, credentials)

# Function to count fish in each frame
def count_fish_in_frame(frame_path, model_name="Iteration5"):
    with open(frame_path, "rb") as test_data:
        results = predictor.detect_image_with_no_store(project_id, model_name, test_data)
    image = cv2.imread(frame_path)
    fish_count = 0
    for prediction in results.predictions:
        if prediction.probability > 0.6:
            fish_count += 1
            box = prediction.bounding_box
            h, w, _ = image.shape
            start_point = (int(box.left * w), int(box.top * h))
            end_point = (int((box.left + box.width) * w), int((box.top + box.height) * h))
            color = (255, 0, 0)
            thickness = 2
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            cv2.putText(image, f'Fish count: {fish_count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    output_path = frame_path.replace("buffer", "output")
    cv2.imwrite(output_path, image)
    return fish_count

# Processing video frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fish_counts = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = f"buffer/frame_{frame_count}.jpg"
        cv2.imwrite(frame_path, frame)
        fish_count = count_fish_in_frame(frame_path)
        fish_counts.append(fish_count)
        frame_count += 1
    cap.release()
    return fish_counts

# Main execution
if __name__ == "__main__":
    video_path = "data/video.mp4"
    fish_counts = process_video(video_path)
    df = pd.DataFrame(fish_counts, columns=["num"])
    df.to_csv("data/fish_counts.csv", index=False)
    print("Fish counting completed and results saved to fish_counts.csv")

