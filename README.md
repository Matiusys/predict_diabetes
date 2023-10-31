#Laporan Project Machine Learning

#Nama : Matius Yudika Sitorus
#Nim : 211351079
#Kelas :Malam B

#Domain Proyek
Prediksi diabetes ini boleh digunakan sebagai patokan bagi semua kalangan muda maupun lansia untuk siapa yang terkena penyakit diabet maupun tidak

#Business Understanding
Untuk merancang model data mining yang dapat digunakan web app ini dikembangkan untuk memudahkan pengguna dalam menentukan proses pengobatan selanjutnya tergantung dari hasil output web app ini.

#Problem Statement
- Memungkinkan seorang professional/dokter/ahli gizi bekerja lebih cepat dan tepat, dengan itu lebih banyak pasien akan mendapatkan penanganan langsung. 

#Goals
Memudahkan dokter/ahli dalam menentukan pengobatan selanjutnya bagi pasien yang mengidap/tidak mengidap penyakit diabetes dengan hasil yang dikeluarkan oleh web app.

#Data Understanding
Dataset yang saya gunakan berasal dari kaggle yang berisi prediksi diabetes. Dataset ini merupakan sekumpulan data yang dikumpulkan dari website real.  Datasets ini mengandung 9 Attribut(Kolom) dan 100,000 data(baris) pada saat sebelum pemrosesan data cleasing dan EDA.

#Variabel-Variabel pada Diabetes prediction dataset adalah sebagai berikut :
- Kehamilan: Untuk menyatakan Jumlah kehamilan
- Glukosa: Untuk menyatakan kadar Glukosa dalam darah
- Tekanan Darah: Untuk menyatakan pengukuran tekanan darah
- SkinThickness: Untuk menyatakan ketebalan kulit
- Insulin: Untuk menyatakan kadar Insulin dalam darah
- BMI: Untuk menyatakan indeks massa tubuh
- Silsilah DiabetesFungsi : Untuk menyatakan persentase Diabetes
- Usia: Untuk menyatakan usia
- Hasil: Untuk menyatakan hasil akhir 1 adalah Ya dan 0 adalah Tidak

#Data Preparation

Untuk data preparation ini saya melakukan EDA (Exploratory Data Analysis) terlebih dahulu, lalu melakukan proses data cleansing agar model yang dihasilkan memiliki akurasi yang lebih tinggi.

Sebelum memulai data preparation, mari kita mendownload datasets dari kaggle yang akan kita gunakan. Langkah Pertama adalah memasukan token kaggle,

from google.colab import files
files.upload()

Lalu kita harus membuat folder untuk menampung file dari kaggle yang tadi telah di upload,

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle

Lalu, download datasets menggunakan code dibawah ini,

!kaggle datasets download -d akshaydattatraykhare/diabetes-dataset

Setelah download telah selesai, langkah selanjutnya mengekstrak file zipnya kedalam sebuah folder

Datasets telah diekstrak seharusnya sekarang ada folder yang bernama diabetes_dataset dan di dalamnya terdapat file dengan ektensi .csv,
Langkah selanjutnya adalah mengimport library yang dibutuhkan untuk melaksanakan data Exploration, data visualisation, dan data cleansing,

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc

Selanjutnya, mari baca file .csv yang tadi kita ekstrak, lalu melihat 5 data pertama yang ada pada datasets,

data = pd.read_csv('diabetes-dataset/diabetes.csv')

Lalu untuk melihat jumah data, mean data, data terkecil dan data terbesar bisa dengan kode ini,

data.decribe()

Untuk melihat typedata yang digunakan oleh masing-masing kolom bisa menggunakan kode ini,

data.info()

Selanjutnya kita akan melihat korelasi antar kolomnya,

plt.figure(figsize=(15, 5))
sns.heatmap(df.corr(), annot=True)




Korelasi antar kolom numerik terlihat aman namun saya merasa terlalu banyak data yang tidak berkaitan erat, selanjutnya melihat apakah di dalam datasetsnya terdapat nilai null,

sns.heatmap(data.isnull())








Semuanya merah yang menandakan bahwa datasetsnya tidak memiliki data null di dalamnya, selanjutnya akan melihat apakah ada data duplikasi,

data[data.duplicated()]

Selanjutnya, akan melihat variabel variabel diantara tabel tersebut

df['Outcome'].value_counts()

selanjutnya kita melihat table yang terdapat pada variabel variabel

sns.countplot(data=df_oversampler, x='Outcome')

Dan proses EDA dan data cleaning sudah diselesaikan. Selanjutnya adalah membuat modelnya.

#Modeling

Model Machine Learning yang akan digunakan disini adalah Logistic Regression, langkah pertama yang harus dilakukan adalah memasukan semua library yang akan digunakan pada saat proses pembuatan model,

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc

Lalu Membuat variabel yang akan menampung fitur-fitur dan targetnya,

X_oversample, y_oversample = smote.fit_resample(X_train, y_train)

Langkah Selanjutnya adalah membuat train test split, dengan presentase 30% test dan 70% train,

X_train, X_test, y_train, y_test = train_test_split(df_oversampler[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']], df_oversampler['Outcome'], test_size=0.2, stratify=df_oversampler['Outcome'], random_state=101)

Dan selanjutnya adalah mengimplementasikan model Logistic Regression dan melihat tingkah akurasinya,

smote = SMOTE(random_state = 101)
X_oversample, y_oversample = smote.fit_resample(X_train, y_train)

Score yang kita dapatkan adalah 75% untuk test dan 74% untuk train, lalu akhirnya kita akan diuji dengan data inputan kita sendiri,

input_data = np.array([[2018,6000,245,40]])

prediction = LogisticRegression(input_data)
print(classification_report(y_test, y_pred))

Dan Hasilnya adalah  yang artinya tidak berpotensi pengidap diabetes. Sebelum mengakhiri ini, kita harus ekspor modelnya menggunakan pickle agar nanti bisa digunakan pada media lain.

import pickle
filename = 'Prediction_diabetes.sav'
pickle.dump(LogisticRegression,open('Prediction_diabetes.sav','wb'))

#Evaluation

Pie Chart  yang saya gunakan disini adalah Confusion Pie, karena ianya sangat cocok untuk kasus pengkategorian seperti kasus ini. Dengan membandingkan nilai aktual dengan nilai prediksi, kita bisa melihat jumlah hasil prediksi saat model memprediksi diabetes dan nilai aktual pun diabetes, serta melihat saat model memprediksi diabetes sedangkan data aktualnya tidak diabetes.

Disitu terlihat jelas bahwa model kita berhasil memprediksi nilai diabetes yang sama dengan nilai aktualnya sebanyak

#Deployment