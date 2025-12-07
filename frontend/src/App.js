import logo from './logo.svg';
import './App.css';
import React, { useState } from "react";
import 'bootstrap/dist/css/bootstrap.min.css'
import axios from "axios";




function App() {
  const [amount, setAmount] = useState("");
  const [month, setMonth] = useState("");
  const [day, setDay] = useState("");
  const [year, setYear] = useState("");
  const [result, setResult] = useState(null);

  const predict = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:8000/api/predict/", {
        Amount: Number(amount),
        Month: Number(month),
        Day: Number(day),
        Year: Number(year)
      });
      setResult(response.data.prediction);
    } catch (err) {
      console.error(err);
    }
  };

  const getBadgeColor = () => {
    if (result === "Essential") return "success";
    if (result === "Luxury") return "warning";
    if (result === "Transport") return "info";
    return "secondary";
  };

  return (
    <div className="app-bg d-flex align-items-center justify-content-center">

      <div className="card shadow-lg p-4 form-card">

        <h2 className="text-center mb-4 fw-bold text-primary">
          💳 AI Expense Prediction
        </h2>

        <input className="form-control mb-3" placeholder="Amount"
          value={amount} onChange={e => setAmount(e.target.value)} />

        <input className="form-control mb-3" placeholder="Month"
          value={month} onChange={e => setMonth(e.target.value)} />

        <input className="form-control mb-3" placeholder="Day"
          value={day} onChange={e => setDay(e.target.value)} />

        <input className="form-control mb-3" placeholder="Year"
          value={year} onChange={e => setYear(e.target.value)} />

        <button className="btn btn-primary w-100" onClick={predict}>
          🔍 Predict
        </button>
        <br></br>
        <div className="btn btn-secondary me-2">
          <a 
            href="http://127.0.0.1:8000/api/dashboard" 
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-outline-info"
          >
            📊 Open Dashboard
          </a>
        </div>
        {result && (
          <div className="mt-4 alert fade show text-center fs-4">
            Predicted Category:
            <span className={`badge bg-${getBadgeColor()} ms-2 fs-5`}>
              {result}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
