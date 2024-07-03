```markdown
# Fish Counting Project

This project automates fish counting in video footage using Azure Custom Vision and PyTorch for object detection. It leverages various technologies to achieve accurate results:

* **Azure Custom Vision:** Streamlines training and deployment of custom object detection models.
* **Detecto:** A user-friendly PyTorch library for object detection tasks.
* **PyTorch:** A powerful deep learning framework for building and training custom models.
* **OpenCV:** Provides efficient computer vision functionalities for video processing.
* **Pandas:** Facilitates data manipulation and analysis of fish counts.
* **Matplotlib:** Creates informative visualizations for fish count data.

## Project Overview

1. **Data Collection and Annotation**

   - Gathered fish images from reliable sources like Google Images and Creative Commons, ensuring proper licensing.
   - Carefully annotated images using LabelImg to define fish bounding boxes.
   - Uploaded annotated images to Google Drive for easy access in Google Colab.

2. **Model Training**

   - Employed PyTorch and Detecto to train an object detection model tailored to fish identification within Google Colab.
   - Leveraged Azure Custom Vision for additional training and prediction capabilities (if applicable).

3. **Frame Extraction and Counting**

   - Segmented the video into individual frames for analysis.
   - Processed each frame through the trained model to detect and count fish.
   - Visualized detected fish using bounding boxes to verify accuracy.

4. **Data Plotting**

   - Utilized Matplotlib to generate informative plots that depict fish counts over time.
   - Applied various smoothing techniques (moving average, etc.) for enhanced visualization and trend identification.

## Getting Started

### Prerequisites

- Python 3.7+
- Azure SDK (if using Azure Custom Vision)
- PyTorch
- Detecto
- OpenCV
- Pandas
- Matplotlib

### Installation

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/MatthewNader2/Fish-Counting-Project.git](https://github.com/MatthewNader2/Fish-Counting-Project.git)
   cd fish-counting-project
   ```

2. **Install required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Data Annotation and Upload:**

   - Annotate images using tools like LabelImg in Pascal VOC format (or a suitable format for your object detection task).
   - Upload annotated images to a convenient location (e.g., Google Drive).

2. **Model Training:**

   - Train the model using provided scripts or adjust them for your specific dataset.
   - Consider adjusting training parameters for optimal performance.

3. **Extract Fish Counts:**

   ```bash
   python scripts/fish_counting.py
   ```

4. **Plot Results:**

   ```bash
   python scripts/plot_fish_data.py
   ```

### Code Structure

- `data/`: Stores annotated images and CSV files.
- `scripts/`: Contains Python scripts for data preprocessing, model training, and inference (fish counting).
- `notebooks/` (Optional): Can include Jupyter notebooks for exploratory data analysis and model evaluation, if applicable.

**Note:** Replace placeholders like `'Iteration5'` in the code with your specific Azure Custom Vision project details, if applicable.

### Detailed Steps

**Data Collection**

- Images were sourced from credible platforms like Google Images and Creative Commons, ensuring appropriate licensing.
- Frames were extracted from the video footage to create a more comprehensive dataset.
- The images were meticulously annotated using LabelImg to define fish bounding boxes in Pascal VOC format (or a suitable format for your object detection task). This helps the model learn to identify fish in the video frames.
- Annotated images were uploaded to a convenient location (e.g., Google Drive) for later access during model training in Google Colab.

**Model Training**

- PyTorch, a robust deep learning framework, was used in conjunction with the Detecto library to train an object detection model specifically designed to identify fish in video frames.
- The training process was conducted within Google Colab, a cloud-based platform that offers access to powerful computing resources without requiring local machine setup.
- Azure Custom Vision was potentially employed to further enhance model training and prediction capabilities (if applicable). This can involve uploading your annotated images to Azure Custom Vision and letting it train a model in the cloud.

**Extracting Fish Counts**

- The video was meticulously segmented into individual frames for thorough analysis.
- Each frame was processed through the trained object detection model to detect and count fish instances.
- Bounding boxes were created around detected fish for visual verification and to ensure model accuracy. This helps you to
