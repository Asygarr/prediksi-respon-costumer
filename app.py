from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import uuid

app = Flask(__name__)

# Inisialisasi model Decision Tree
data = pd.read_csv('data/campaign_responses.csv')
X = data[data.columns[1:8]]
y = data['responded']

# Mengonversi kolom objek menjadi numerik dengan Label Encoding
label_encoders = {}
for column in ['gender', 'employed', 'marital_status']:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Mengonversi target menjadi numerik dengan Label Encoding
y = LabelEncoder().fit_transform(y)

# Membagi data menjadi set pelatihan dan pengujian
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Melatih model Decision Tree
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# Memuat halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Memuat halaman hasil
@app.route('/result', methods=['POST'])
def result():
    if 'file' not in request.files:
        return render_template('index.html', message='Tidak ada file yang diunggah')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='Tidak ada file yang dipilih')

    if file:
        # Membuat identifier unik acak
        unique_id = str(uuid.uuid4())
        # Menggabungkan identifier unik dengan nama file yang diunggah
        file_path = 'uploads/' + unique_id + '_' + file.filename
        file.save(file_path)

        # Membaca data dari file yang sudah diunggah
        df = pd.read_excel(file_path)

        # Proses prediksi
        X_input = df[['age', 'gender', 'annual_income', 'credit_score', 'employed', 'marital_status', 'no_of_children']]
        for column in ['gender', 'employed', 'marital_status']:
            le = label_encoders[column]
            X_input[column] = le.transform(X_input[column])
        predictions = clf.predict(X_input)

        # Membuat dataframe baru untuk menyimpan hasil prediksi
        result_df = pd.DataFrame({'Name': df['name'], 'Responded': predictions})
        result_df['Responded'] = result_df['Responded'].replace({1: 'yes', 0: 'no'})

        # Membuat file hasil prediksi
        result_unique_id = str(uuid.uuid4())
        result_path = 'result/' + result_unique_id + '_hasil_prediksi.xlsx'
        result_df.to_excel(result_path, index=False)

        # Membaca file hasil prediksi dan mengonversinya ke HTML
        result_html = result_df.to_html(classes='table table-striped', index=False)

        # Mengembalikan halaman hasil dengan tabel HTML dan link untuk mengunduh hasil prediksi
        return render_template('result.html', result_html=result_html, result_path=result_path)

# Mengunduh file hasil prediksi
@app.route('/download/<filename>')
def download_file(filename):
    return send_file('result/' + filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
