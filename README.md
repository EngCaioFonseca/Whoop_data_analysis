# Whoop & Hevy Fitness Analysis

![image](https://github.com/user-attachments/assets/5a67a6d5-18cc-4dbc-8227-5ee7398461d8)

![image](https://github.com/user-attachments/assets/d68893e6-ec7f-496f-aaf1-892d141810b8)

![image](https://github.com/user-attachments/assets/33fbf941-3a44-4f69-9cd8-42ba014fdcbb)

![image](https://github.com/user-attachments/assets/26b07ef6-14ab-4442-8c9e-e47205bf8707)

![image](https://github.com/user-attachments/assets/0824f8b8-7c5f-4b59-a305-79c4069d9253)

![image](https://github.com/user-attachments/assets/fc910463-da1e-491a-8b01-576cc3be7d3f)

![image](https://github.com/user-attachments/assets/81ed9be3-802b-4bae-bf93-bf96fba5ec4c)

![image](https://github.com/user-attachments/assets/5ecf4925-219a-4667-a68c-9885fd833b0c)

![image](https://github.com/user-attachments/assets/51ce832d-5333-49c4-b53f-271549801096)

![image](https://github.com/user-attachments/assets/e7c9e232-3d8b-4121-97b4-e0b2dba65185)

![image](https://github.com/user-attachments/assets/39b07678-2908-4b4f-ab5e-88fd8fa9af3d)

![image](https://github.com/user-attachments/assets/44487eef-a3b4-4871-bb3b-682920b9d7e0)

![image](https://github.com/user-attachments/assets/e4d939b5-7b0d-48ca-9f04-548026553353)

![image](https://github.com/user-attachments/assets/ea1c8db3-60a8-40c8-9918-42782de1f5c9)

![image](https://github.com/user-attachments/assets/cf103b44-3ae9-4d8f-b310-7d57f71e472d)

![image](https://github.com/user-attachments/assets/4890d366-2eb2-4753-9328-cec4d62aed3b)

![image](https://github.com/user-attachments/assets/0d9d4eaa-f02d-4fc4-9a82-9041db8b61f3)

![image](https://github.com/user-attachments/assets/6ca4522c-cc34-4f92-aba2-79859aa31f88)

![image](https://github.com/user-attachments/assets/07f85f2a-9f9b-4272-ac9e-a7a9227970be)

![image](https://github.com/user-attachments/assets/43d218ce-485e-475a-b253-3b9562818e55)

![image](https://github.com/user-attachments/assets/decdb900-3164-4620-83d9-933a0e9150c9)


## Overview
This project analyzes personal fitness data from both Whoop (recovery/strain metrics) and Hevy (workout tracking) to provide insights into training patterns, recovery relationships, and overall fitness trends. The analysis combines physiological data from Whoop with detailed workout metrics from Hevy to understand the relationship between recovery, strain, and workout performance.

## Features
- Comprehensive analysis of Whoop metrics (recovery, strain, sleep)
- Detailed workout analysis from Hevy data
- Combined analysis of recovery metrics and workout performance
- Machine learning models for recovery prediction
- Time series analysis of fitness trends
- Anomaly detection in recovery patterns
- Cluster analysis of recovery states
- Personalized recommendations based on current metrics

## Data Sources
### Whoop Data
- `physiological_cycles.csv`: Recovery, strain, and sleep cycle data
- `workouts.csv`: Workout-specific data
- `sleep.csv`: Detailed sleep metrics

### Hevy Data
- `workout_data_hevy.csv`: Detailed workout logs including:
  - Exercise details
  - Sets, reps, and weights
  - RPE (Rate of Perceived Exertion)
  - Workout duration and timing

## Installation

bash
Clone the repository
git clone https://github.com/yourusername/whoop-hevy-analysis.git
Install required packages
pip install -r requirements.txt

## Usage

python
Run the main analysis
python whoop_analysis.py

## Analysis Components

### Whoop Analysis
- Recovery score trends
- Sleep performance metrics
- Strain analysis
- Heart rate variability (HRV) patterns
- Respiratory rate trends

### Hevy Analysis
- Workout frequency patterns
- Exercise type distribution
- Volume progression over time
- RPE distribution
- Set and rep patterns

### Combined Analysis
- Recovery score vs. workout performance
- Strain levels vs. workout volume
- Sleep quality impact on workout performance
- Correlation analysis between recovery and workout metrics

## Visualizations
The analysis generates various visualizations including:
- Time series plots of recovery metrics
- Workout volume progression
- Recovery-performance correlation plots
- Cluster analysis visualizations
- Anomaly detection plots
- Sleep metrics analysis

## Machine Learning Components
- Recovery score prediction
- Workout performance clustering
- Anomaly detection in recovery patterns
- Time series forecasting

## Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

## Project Structure

├── whoop_analysis.py
├── requirements.txt
├── data/
│ ├── physiological_cycles.csv
│ ├── workouts.csv
│ ├── sleep.csv
│ └── workout_data_hevy.csv
├── visualizations/
│ ├── recovery_trends.png
│ ├── workout_analysis.png
│ └── combined_insights.png
└── README.md


## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Whoop for providing detailed recovery and strain metrics
- Hevy for comprehensive workout tracking capabilities
- The Python data science community for excellent tools and libraries

## Contact
Your Name - Caio Fonseca
Project Link: [https://github.com/EngCaioFonseca/whoop-hevy-analysis](https://github.com/EngCaioFonseca/Whoop_data_analysis)


