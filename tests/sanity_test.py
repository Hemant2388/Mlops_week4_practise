import joblib
import numpy as np

def test_prediction_output():
    model = joblib.load("artifacts/model.joblib")
    
    # Use a simple, known sample
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Likely to predict 'setosa'
    prediction = model.predict(sample)[0]

    # Write predicted species to shared output file
    with open("test_output.txt", "a") as f:
        f.write("\n🧠 Model Prediction\n")
        f.write(f"✅ Predicted species: {prediction}\n")

    # Ensure it's a valid string
    assert isinstance(prediction, str)
