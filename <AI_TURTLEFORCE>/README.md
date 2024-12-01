# Team <AI_TURTLEFORCE>'s Project

---

# PainSense: AI-Driven Pain Detection and Chatbot Assistance

## Abstract

### Background and Problem Statement
Pain is a critical signal in healthcare, yet its accurate and timely detection remains a challenge, especially in scenarios where individuals are unable to communicate effectively. This project addresses the need for an automated, AI-driven solution to detect, analyze, and assist in managing pain using multimodal inputs like audio and video.

### Impact and Proposed Solution
The proposed system combines computer vision, audio analysis, and natural language processing to:
1. Detect pain-related signals in audio and video.
2. Provide real-time feedback through an AI-powered chatbot.
3. Offer insights to healthcare providers or caregivers for better decision-making.

### Project Outcomes and Deliverables
- **Multimodal Pain Detection**: Combines facial expression recognition and audio sentiment analysis.
- **Interactive Chatbot**: Assists users by answering queries and providing pain-related recommendations.
- **Visualization**: Generates graphs to represent pain levels over time.
- **Flask API and Gradio Interface**: Enables integration and user-friendly interaction.

---

## Features
1. **Audio and Video Analysis**: Detects pain signals using models like FER (Facial Emotion Recognition) and Wav2Vec2 for audio.
2. **Customizable Thresholds**: Configurable thresholds for pain and confidence levels.
3. **Real-Time Feedback**: Provides immediate results and chatbot assistance.
4. **Visualization**: Displays pain scores graphically for better understanding.

---

## System Requirements

- Python 3.8 or later
- Required Libraries (install via `requirements.txt`):
---

## How to Run the Project

Step 1: Clone the Repository
Step 2: Install Dependencies
Step 3: Run the Flask Server
Step 4: Launch Gradio Interface
The Gradio interface will automatically launch in your default browser, or you can access it via the link provided in the terminal.

---

## Instructions

### **Uploading Inputs**
1. Upload an **audio file** (e.g., `.mp3`, `.wav`) or **video file** (e.g., `.mp4`, `.avi`).
2. Click **Submit** to analyze the input.

### **Viewing Results**
- **Pain Detection Result**: Indicates if pain is detected.
- **Average Pain Score**: Displays the overall pain intensity.
- **Graph**: Shows the pain score variation over time.

### **Chatbot Interaction**
- Type your query in the chatbot input box.
- Receive contextual assistance based on pain detection results.

---

## Architecture

### **1. Pain Detection**
- **Facial Expression Recognition (FER)**: Detects emotions related to pain (e.g., anger, fear, sadness) from video frames.
- **Audio Sentiment Analysis**: Analyzes audio for pain-related emotional cues using Wav2Vec2.

### **2. Data Fusion**
Combines results from video and audio analysis using a weighted average.

### **3. Visualization**
- Generates a graph illustrating pain score trends.
- Includes a configurable pain threshold for better insights.

### **4. Chatbot**
- Built using the Groq API for contextual interactions.
- Processes user queries and adapts responses based on pain detection results.

---

## Acknowledgements
This project utilizes:
- [FER](https://github.com/justinshenk/fer) for facial emotion recognition.
- [Wav2Vec2](https://huggingface.co/models) for audio sentiment analysis.
- [Gradio](https://gradio.app/) for interactive UI.
- [Flask](https://flask.palletsprojects.com/) for API backend.

---

## Future Enhancements
1. Extend support for additional languages in the chatbot.
2. Incorporate more robust pain detection models.
3. Enable integration with wearable devices for real-time monitoring.
4. Radiology Report Generation:
   - Develop an AI-driven system to generate detailed radiology reports in English and French for radiologists based on medical images.
   - Simplify these reports into Darija (Moroccan Arabic) for better understanding by patients.
5. Explore multi-modal fusion techniques to improve detection accuracy across diverse inputs.

---
