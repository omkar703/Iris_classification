# Iris Classification: System Architecture

## 1. System Topology

The Iris Classification application is engineered as a lightweight, low-latency microservice combining a Flask REST API backend with a responsive, standalone client-side frontend.

```
Client (Browser / Automated Sensor)
  │
  ├── HTTP GET  /            ──> Jinja2 Template (templates/index.html)
  ├── HTTP POST /predict     ──> Flask Application (app.py)
  │                                   │
  │                                   ├── Field Alias Normalizer
  │                                   ├── Feature Range Validator
  │                                   └── Scikit-Learn GaussianNB (iris_model.pkl)
  │                                           │
  │                             <─────────────┴── Returns JSON: class, probabilities
  └── HTTP GET  /favicon.ico ──> send_from_directory(static/favicon.ico)
```

---

## 2. API Contract Specification

### Endpoint: `POST /predict`
- **Request Headers:** `Content-Type: application/json`
- **Supported Feature Keys & Aliases:**
  - `sepal length (cm)` or `sepal_length`
  - `sepal width (cm)` or `sepal_width`
  - `petal length (cm)` or `petal_length`
  - `petal width (cm)` or `petal_width`
- **Payload Schema:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```
- **Response Schema (`200 OK`):**
```json
{
  "class_flower": "setosa",
  "prediction": 0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```

---

## 3. Client-Side Standalone Execution Mode

The web frontend (`templates/index.html`) incorporates an embedded client-side Gaussian likelihood inference routine that enables interactive specimen exploration without requiring a live Python runtime (enabling zero-dependency static deployments e.g., on Vercel or GitHub Pages). When a live backend is connected, the UI seamlessly delegates to `POST /predict`.
