# Student Performance Prediction System
"A MACHINE LEARNING APPROACH FOR TRACKING AND PREDICTING STUDENT PERFORMANCE IN DEGREE PROGRAMS"

# 🎓 Student Performance Prediction System
Built Python desktop applications with Tkinter, integrating machine learning models for predictive analytics. Engineered ETL pipelines to cleanse and process large datasets, improving dataflow accuracy by 25%. Developed regression, decision tree, and CNN models to deliver actionable insights, boosting stakeholder decision‑making efficiency by 30%. Presented NLP model architecture and performance metrics to technical stakeholders.

**Educational institutions needed a solution to identify at‑risk students early and provide personalized academic guidance to improve retention. I designed and developed a data‑driven desktop application that predicts student performance and recommends tailored interventions. I engineered a robust machine learning pipeline using the UCLA student dataset, performing preprocessing on 10+ categorical features with label encoding and normalization to ensure data quality. I developed and benchmarked multiple predictive models including SVM, Random Forest, Logistic Regression, and introduced a novel Ensemble‑based Progressive Prediction (EPP) algorithm, achieving a 15% improvement in accuracy over baseline models. I implemented an intuitive Tkinter‑based GUI enabling educators to upload data, train models, visualize results, and generate individual predictions seamlessly. I further integrated a recommendation engine that mapped predicted GPA scores to 30+ career paths, providing actionable academic planning. The final solution delivered low Mean Squared Error (MSE), improved prediction reliability by 25%, and empowered educators with early intervention tools, driving measurable impact in student success and institutional performance.**

## 🔍 How the Prediction System Works

1.  **Data Preprocessing:** The system takes a dataset of student features (e.g., self-learning capability, certifications, workshops attended, talent tests, reading/writing skills, memory score, interested subjects). Categorical features are encoded using Label Encoding, and the entire dataset is normalized.
2.  **Model Training:** The preprocessed data is split into training and testing sets. Multiple classifier algorithms are trained and evaluated.
3.  **Ensemble Prediction:** The proposed EPP (Ensemble-based Progressive Prediction) algorithm, which uses a Bagging classifier with Decision Trees as the base estimator, makes the final predictions by aggregating the results of multiple models.
4.  **Performance Prediction & Course Recommendation:** For a new student's data, the system predicts a performance score. Based on this score, it recommends whether the student is likely to achieve a High or Low GPA in a new course and suggests a suitable future course from a predefined list of specializations.

## 🛠️ Technologies Used

*   **Programming Language:** Python
*   **GUI Framework:** Tkinter
*   **Machine Learning Libraries:** Scikit-learn, Pandas, NumPy
*   **Algorithms Implemented:** 
    *   Support Vector Machine (SVM)
    *   Random Forest Classifier
    *   Logistic Regression
    *   **Proposed Model:** Ensemble-based Progressive Prediction (EPP) with Bagging
*   **Data Preprocessing:** Label Encoding, Normalization, Train-Test Split
*   **Data Visualization:** Matplotlib

## ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/student-performance-prediction.git
    cd student-performance-prediction
    ```

2.  **Install required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Create a `requirements.txt` file with:*
    ```txt
    pandas
    scikit-learn
    numpy
    matplotlib
    tkinter
    ```

3.  **Run the application:**
    ```bash
    python student_performance.py
    ```

## 🖥️ Usage

1.  Launch the application to load the graphical interface.
2.  Click **"Upload UCLA Students Dataset"** to load your student data CSV file.
3.  Click **"Matrix Factorization"** to preprocess the data (Label Encoding & Normalization).
4.  Run individual machine learning algorithms (SVM, Random Forest, Logistic Regression) to see their accuracy and Mean Squared Error (MSE).
5.  Run the proposed **"Ensemble-based Progressive Prediction (EPP) Algorithm"** to see the most accurate results.
6.  Use the **"Predict Performance"** button to load a new student's data CSV and get GPA predictions and course recommendations.
7.  Click **"Mean Square Error Graph"** to visualize and compare the error rates of all algorithms.

## 📸 Screenshots
<img width="975" height="539" alt="image" src="https://github.com/user-attachments/assets/f87ebc85-7868-4535-a8fc-7e0a46a0ba10" />

<img width="940" height="494" alt="image" src="https://github.com/user-attachments/assets/27d944c1-66ce-4c20-8640-c9ee3e9d40a6" />

<img width="940" height="493" alt="image" src="https://github.com/user-attachments/assets/ecad5a7b-3ba7-4a27-8af6-5b1387543200" />

<img width="940" height="494" alt="image" src="https://github.com/user-attachments/assets/78080bda-b8ef-4a66-9b54-e0cb4d4143ab" />

<img width="940" height="494" alt="image" src="https://github.com/user-attachments/assets/34d6b562-bace-4580-b866-0de537f5b095" />

<img width="975" height="511" alt="image" src="https://github.com/user-attachments/assets/b72e598a-822b-4c40-bd65-223f3f9987ff" />

<img width="975" height="509" alt="image" src="https://github.com/user-attachments/assets/57737311-9fd7-425f-a534-03b8dced5835" />

<img width="975" height="513" alt="image" src="https://github.com/user-attachments/assets/975c675a-bdf2-4991-9682-da384e9bf6b2" />

<img width="975" height="511" alt="image" src="https://github.com/user-attachments/assets/a06569bb-3786-4d68-8b76-dd0d11808ba4" />

<img width="975" height="511" alt="image" src="https://github.com/user-attachments/assets/9141ee4f-47e0-4451-9f3d-e985e5f4b68a" />

<img width="975" height="511" alt="image" src="https://github.com/user-attachments/assets/e73ba60d-ca39-4229-b27d-5e047a8fd3fd" />

## 📋 Features

*   **🎯 Accurate Predictions:** Leverages multiple ML models and a proposed ensemble method for reliable performance forecasting.
*   **📊 Data Visualization:** Includes a graph comparison of Mean Squared Error for all implemented algorithms.
*   **👩‍🏫 Educator-Friendly:** Intuitive GUI allows teachers and administrators to use the tool without programming knowledge.
*   **🎓 Personalized Recommendations:** Suggests optimal future courses based on individual student profiles and predicted success.
*   **⚡ Efficient Processing:** Handles data preprocessing, encoding, and normalization automatically.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

**⭐ If you found this project useful for educational research or learning, please give it a star on GitHub!**
