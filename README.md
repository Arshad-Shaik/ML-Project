# ML-Project - Student Performance Prediction System
"A MACHINE LEARNING APPROACH FOR TRACKING AND PREDICTING STUDENT PERFORMANCE IN DEGREE PROGRAMS"

# 🎓 Student Performance Prediction System
A comprehensive **Machine Learning desktop application** built with **Python and Tkinter** designed to track, analyze, and predict student academic performance in degree programs. This system leverages ensemble learning techniques to provide accurate GPA predictions and personalized course recommendations, aiding educational institutions in proactive student intervention.

## 📖 Situation-Task-Action-Result (STAR Method)

*   **Situation:** Educational institutions face challenges in identifying at-risk students early and providing personalized academic guidance to improve retention and success rates in degree programs.
*   **Task:** To design and develop an intelligent, data-driven software solution that can accurately predict student performance based on academic and behavioral features, and provide actionable recommendations for courses and interventions.
*   **Action:** 
    *   **Engineered** a robust machine learning pipeline using a **UCLA student dataset**, performing comprehensive data preprocessing including label encoding and normalization for 10+ categorical features.
    *   **Developed and compared** multiple predictive models including **Support Vector Machines (SVM), Random Forest, Logistic Regression**, and a **novel Ensemble-based Progressive Prediction (EPP) algorithm** using Bagging with Decision Trees.
    *   **Designed and implemented** an intuitive **Graphical User Interface (GUI)** using Tkinter, allowing educators to upload data, train models, visualize results, and generate individual student predictions seamlessly.
    *   **Integrated** a recommendation engine that suggests optimal future courses based on predicted GPA scores, mapping outcomes to over 30 distinct career paths and specializations.
*   **Result:** Successfully delivered a fully-functional desktop application that demonstrates the effective use of ensemble machine learning for educational analytics. The system provides educators with a powerful tool to predict student performance with low Mean Squared Error (MSE), enabling early intervention and personalized academic planning to improve student outcomes.

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

## 📋 Features

*   **🎯 Accurate Predictions:** Leverages multiple ML models and a proposed ensemble method for reliable performance forecasting.
*   **📊 Data Visualization:** Includes a graph comparison of Mean Squared Error for all implemented algorithms.
*   **👩‍🏫 Educator-Friendly:** Intuitive GUI allows teachers and administrators to use the tool without programming knowledge.
*   **🎓 Personalized Recommendations:** Suggests optimal future courses based on individual student profiles and predicted success.
*   **⚡ Efficient Processing:** Handles data preprocessing, encoding, and normalization automatically.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/student-performance-prediction/issues).

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

**⭐ If you found this project useful for educational research or learning, please give it a star on GitHub!**
