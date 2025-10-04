import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import "./App.css";

// ================= Home Page =================
const Home = () => (
  <div className="homepage">
    <header className="home-header">
      <h1>Insurance Cross Sell Prediction</h1>
      <p>Smart insurance predictions using ML models!</p>
      <Link to="/predict" className="btn-primary">Go to Prediction</Link>
    </header>
    <img src="/insurance.png" alt="Insurance Banner" className="home-image" />
  </div>
);

// ================= Prediction Page =================
const Predict = () => {
  // Default input values that naturally give YES prediction
  const [form, setForm] = useState({
    Gender_Male: "1",
    Gender_Female: "0",
    Age_log: "3.9",
    Driving_License: "1",
    Region_Code_Encoding: "28",
    Previously_Insured: "0",
    Vehicle_Age_Encoding: "1",
    Vehicle_Damage_Encoding: "1",
    Annual_Premium: "40000",
    Policy_Sales_Channel_Encoding: "152",
    Vintage: "30"
  });

  const [result, setResult] = useState(null);
  const [probability, setProbability] = useState(null);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await response.json();
      setResult(data.prediction);
      setProbability(data.probability);
    } catch (error) {
      console.error("Error fetching prediction:", error);
      setResult("Error connecting to server ❌");
      setProbability(null);
    }
  };

  return (
    <div className="main-container">
      <div className="form-section">
        <h1>Prediction Form</h1>
        <form onSubmit={handleSubmit}>
          {Object.keys(form).map((field) => (
            <div className="form-field" key={field}>
              <label>{field.replace("_", " ")}</label>
              <input
                type="text"
                name={field}
                value={form[field]}
                onChange={handleChange}
                required
              />
            </div>
          ))}
          <button type="submit">Predict</button>
        </form>
      </div>

      <div className="result-section">
        {result ? (
          <div
            className="result-card"
            style={{ borderLeftColor: result === "YES" ? "#0FA4AF" : "#964734" }}
          >
            <h2>Prediction Result</h2>
            <p><strong>Status:</strong> {result}</p>
            {probability !== null && (
              <p><strong>Probability:</strong> {(probability * 100).toFixed(2)}%</p>
            )}
          </div>
        ) : (
          <div className="result-card">
            <h2>Awaiting Input</h2>
            <p>Fill in the form and click Predict to see results.</p>
          </div>
        )}
        <Link to="/" className="btn-secondary">Back to Home</Link>
      </div>
    </div>
  );
};

// ================= Main App =================
const App = () => (
  <Router>
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/predict" element={<Predict />} />
    </Routes>
  </Router>
);

export default App;
