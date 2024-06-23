from flask import Flask, request, render_template
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the dataset
data = pd.DataFrame({
    'Usia': ['Muda', 'Dewasa', 'Tua', 'Muda', 'Dewasa', 'Tua', 'Muda', 'Dewasa', 'Tua', 'Muda'],
    'Pendapatan': ['Rendah', 'Tinggi', 'Sedang', 'Sedang', 'Rendah', 'Tinggi', 'Rendah', 'Tinggi', 'Sedang', 'Tinggi'],
    'Status_Pernikahan': ['Single', 'Menikah', 'Menikah', 'Single', 'Menikah', 'Menikah', 'Single', 'Menikah', 'Menikah', 'Single'],
    'Jumlah_Anak': ['Tidak ada', 'Banyak', 'Sedikit', 'Tidak ada', 'Sedikit', 'Banyak', 'Sedikit', 'Tidak ada', 'Sedikit', 'Tidak ada'],
    'Membeli_Rumah': ['Tidak', 'Ya', 'Ya', 'Tidak', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Ya', 'Ya']
})

# Prepare the data for Naive Bayes
X = data.drop(columns=['Membeli_Rumah'])
y = data['Membeli_Rumah']

# Encode categorical variables
label_encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Encode the target variable
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Train the Naive Bayes model
model = CategoricalNB()
model.fit(X, y)

def calculate_percentages(initial_data):
    total = len(initial_data)
    percentages = {}
    
    for column in initial_data.columns:
        column_data = initial_data[column].value_counts().to_dict()
        percentages[column] = []
        
        for value in initial_data[column].unique():
            ya_count = len(initial_data[(initial_data[column] == value) & (initial_data['Membeli_Rumah'] == 'Ya')])
            tidak_count = len(initial_data[(initial_data[column] == value) & (initial_data['Membeli_Rumah'] == 'Tidak')])
            total_count = ya_count + tidak_count
            p_ya = ya_count / total_count if total_count != 0 else 0
            p_tidak = tidak_count / total_count if total_count != 0 else 0
            
            percentages[column].append({
                'index': value,
                'Ya': ya_count,
                'Tidak': tidak_count,
                'P(Ya)': p_ya,
                'P(Tidak)': p_tidak
            })
    
    return percentages

def save_prediction_to_dataset(input_data, prediction):
    global data
    new_data = pd.DataFrame([input_data])
    new_data['Membeli_Rumah'] = prediction
    data = pd.concat([data, new_data], ignore_index=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    percentages = None

    if request.method == 'POST':
        input_data = {
            'Usia': request.form['Usia'],
            'Pendapatan': request.form['Pendapatan'],
            'Status_Pernikahan': request.form['Status_Pernikahan'],
            'Jumlah_Anak': request.form['Jumlah_Anak']
        }

        # Transform input data using label encoders
        input_df = pd.DataFrame([input_data])
        for column in input_df.columns:
            input_df[column] = label_encoders[column].transform(input_df[column])

        # Predict
        prediction_encoded = model.predict(input_df)
        prediction = le_target.inverse_transform(prediction_encoded)[0]

        # Save prediction to dataset
        save_prediction_to_dataset(input_data, prediction)

        # Calculate percentages based on initial_data
        percentages = calculate_percentages(data)
    
    return render_template('index.html', prediction=prediction, percentages=percentages)

@app.route('/dataset')
def dataset():
    initial_data = pd.DataFrame({
        'Usia': ['Muda', 'Dewasa', 'Tua', 'Muda', 'Dewasa', 'Tua', 'Muda', 'Dewasa', 'Tua', 'Muda'],
        'Pendapatan': ['Rendah', 'Tinggi', 'Sedang', 'Sedang', 'Rendah', 'Tinggi', 'Rendah', 'Tinggi', 'Sedang', 'Tinggi'],
        'Status_Pernikahan': ['Single', 'Menikah', 'Menikah', 'Single', 'Menikah', 'Menikah', 'Single', 'Menikah', 'Menikah', 'Single'],
        'Jumlah_Anak': ['Tidak ada', 'Banyak', 'Sedikit', 'Tidak ada', 'Sedikit', 'Banyak', 'Sedikit', 'Tidak ada', 'Sedikit', 'Tidak ada'],
        'Membeli_Rumah': ['Tidak', 'Ya', 'Ya', 'Tidak', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Ya', 'Ya']
    })
    
    data_html = initial_data.to_html(classes='table table-striped table-bordered table-hover')
    percentages = calculate_percentages(initial_data)
    return render_template('dataset.html', data_html=data_html, percentages=percentages)

if __name__ == '__main__':
    app.run(debug=True)
