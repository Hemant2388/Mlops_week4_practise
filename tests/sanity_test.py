import joblib
import pandas as pd

def test_model_prediction():
    model = joblib.load("artifacts/model.joblib")
    
    # A sample test input (from setosa class)
    test_sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=[
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
    ])
    
    prediction = model.predict(test_sample)
    
    with open("prediction_output.txt", "w") as f:
        f.write(f"✅ Predicted species: {prediction[0]}\n")
    
    assert prediction[0] in ['setosa', 'versicolor', 'virginica']