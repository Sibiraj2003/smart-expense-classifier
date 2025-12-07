Smart Expense Classifier

AI + Machine Learning expense classification with real-time analytics dashboard

This project uses Machine Learning (CatBoost), Django REST Framework, and a React frontend to classify user expenses into categories (Essential, Luxury, Transport), visualize classification performance, and provide monthly & category-wise spending insights through an interactive dashboard.

⭐ Key Features
🧠 AI Expense Classification

CatBoost ML model

Oversampling + class weighting

Balanced recall across classes

Real-world class imbalance handling

📊 Analytics Dashboard (Django)

Recall per class

Confusion matrix

Class distribution

Monthly trend analysis

Category spend distribution

⚛ React Frontend

Expense input form

Instant classification

Async API communication

Responsive UI

📦 Tech Stack
Layer	Technology
Frontend	React, JavaScript
Backend API	Django REST Framework
ML	CatBoost, Scikit-learn
Analytics	Chart.js, Bootstrap
Data	CSV dataset (transaction history)
🧪 Machine Learning

Model: CatBoostClassifier
Model Strategy

Cyclic encoding (Month, Day, Weekday)

Feature pruning

Random oversampling

Class balancing

Custom class weights

Original Dataset
Highly imbalanced toward “Essential” category → caused high accuracy but poor recall for minority classes.

Solution

Oversampling

Auto class weights

Custom soft class weighting

Improved minority recall without collapsing accuracy

📈 Dashboard Visualizations

Confusion Matrix

Class Distribution

Recall Per Class

Monthly Spend Line Chart

Spend by Category

🚀 Getting Started (Local Development)
1️⃣ Clone Repo
git clone https://github.com/<username>/smart-expense-classifier.git
cd smart-expense-classifier

2️⃣ Backend Setup
cd backend
python -m venv venv
source venv/Scripts/activate   # Windows
pip install -r requirements.txt
python manage.py runserver

3️⃣ Frontend Setup
cd frontend
npm install
npm start

🔌 API Usage

POST /predict-expense

{
  "Amount": 1200,
  "Year": 2025,
  "Month": 12,
  ...
}


Response:

{
  "prediction": "Luxury"
}

📍 Project Structure
/frontend          → React app
/backend           → Django + REST API
/ml                → Training scripts + models + metrics

⚡ Deployment Strategy
Backend

Render.com / Railway / Fly.io

gunicorn

whitenoise

collectstatic

Frontend

Vercel

Build → auto deploy

Environment variable for API url

📁 Important Files

ml/model_training.py

ml/output/models/model_refined.pkl

ml/output/plots/metrics.json

dashboard.html (Chart.js)

🎓 What I Learned

Handling imbalanced datasets

Custom model class weighting

Cyclic time encoding

Building analytical dashboards

Full-stack ML application design

Deployment best practices

Perfect for ML + Django full-stack portfolio projects.

🧩 Future Enhancements

JWT user login

User-based spending history

PDF analytics export

Budget alerts

SQL database integration

📸 Screenshots

(LATER>>>...)

![dashboard screenshot]()
![react form]()

🤝 Contributing

Pull requests are welcome.

🧑‍💻 Author

Sibiraj

AI & ML enthusiast

Full Stack developer in progress