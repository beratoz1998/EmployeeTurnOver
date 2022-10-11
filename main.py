from eda import *
from data_prep import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load_data():
    df = pd.read_csv("Employee.csv") #veri okuma işlemi gerçekleştirdik.
    return df
df = load_data()

### DEĞİŞKEN TANIMI ###
#######################
"""
Education = Eğitim bilgisini göstermektedir.
Joining Year = İşe giriş yılını göstermektedir.
City = Şehir Bilgilerini göstermektedir.
Payment Tier = ÖDEME DÜZEYİ: 1: EN YÜKSEK 2: ORTA DÜZEY 3: EN DÜŞÜK
Age = Yaşı göstermektedir.
Gender = Cinsiyeti Göstermektedir.
EverBenched = HİÇ 1 AY VEYA DAHA FAZLA PROJELERİN DIŞINDA KALDI MI?
Experience in Current Domain = MEVCUT ALAN TECRÜBESİ
Leave or Not = ÇALIŞANIN SONRAKİ 2 YIL İÇİNDE ŞİRKETTEN AYRILIP AYIRMADIĞI
"""

### KEŞİFÇİ VERİ ANALİZİ ###
############################
df.info()
df.isnull().sum() # Boş veri yok
df.shape # (4653,9)
df.describe().T
df.head()

df["City"].value_counts()
# Bangalore 2258
# Pune 1268
# New Delhi 1157

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# 9 değişkenden 1 tane numerik 8 tane kategorik değişkenimiz var. 4 tanesi numerik fakat kategorik değişken.

# Kategorik değişkenlerin incelenmesi
for col in cat_cols:
    cat_summary(df, col)

# Sayısal değişkenlerin incelenmesi
df[num_cols].describe().T

for col in num_cols:
    num_summary(df, col, plot=True)

# Target ile kategorik değişkenlerin incelemesi
for col in cat_cols:
    target_summary_with_cat(df, "LeaveOrNot", col)

rare_analyser(df, "LeaveOrNot", cat_cols)


### FEATURE ENGİNEERİNG ###
###########################

# 22-41 yaş arası 20 farklı yaş grubunu 4 e bölüp bunları erkek ve kadın olarak gruplandırıyoruz.
# 22-26 = 1
# 27-31 = 2
# 32-36 = 3
# 37-41 = 4
df.loc[(df['Gender'] == 'Male') & (22 <= df['Age']) & (df['Age'] <= 26), 'New_Age_Cat'] = 'm_1'
df.loc[(df['Gender'] == 'Male') & (27 <= df['Age']) & (df['Age'] <= 31), 'New_Age_Cat'] = 'm_2'
df.loc[(df['Gender'] == 'Male') & (32 <= df['Age']) & (df['Age'] <= 36), 'New_Age_Cat'] = 'm_3'
df.loc[(df['Gender'] == 'Male') & (37 <= df['Age']) & (df['Age'] <= 41), 'New_Age_Cat'] = 'm_4'
df.loc[(df['Gender'] == 'Female') & (22 <= df['Age']) & (df['Age'] <= 26), 'New_Age_Cat'] = 'f_1'
df.loc[(df['Gender'] == 'Female') & (27 <= df['Age']) & (df['Age'] <= 31), 'New_Age_Cat'] = 'f_2'
df.loc[(df['Gender'] == 'Female') & (32 <= df['Age']) & (df['Age'] <= 36), 'New_Age_Cat'] = 'f_3'
df.loc[(df['Gender'] == 'Female') & (37 <= df['Age']) & (df['Age'] <= 41), 'New_Age_Cat'] = 'f_4'

# 0-7 arası tecrübe 8 farklı yıl grubu mevcut. 4 farklı kategoriye bölüyoruz ve ödeme düzeylerine göre gruplandırıyoruz.
# 0-2 = NewGraduate
# 2-4 = Junior
# 4-6 = Middle
# 6-7 = Senior
df.loc[( 0 <= df['ExperienceInCurrentDomain']) & (1 >= df['ExperienceInCurrentDomain']) & (df['PaymentTier'] == 1 ), 'PaymentExprerience_Cat'] = 'high_newgraduate'
df.loc[( 2 <= df['ExperienceInCurrentDomain']) & (3 >= df['ExperienceInCurrentDomain']) & (df['PaymentTier'] == 1 ), 'PaymentExprerience_Cat'] = 'high_junior'
df.loc[( 4 <= df['ExperienceInCurrentDomain']) & (5 >= df['ExperienceInCurrentDomain']) & (df['PaymentTier'] == 1 ), 'PaymentExprerience_Cat'] = 'high_middle'
df.loc[( 6 <= df['ExperienceInCurrentDomain']) & (7 >= df['ExperienceInCurrentDomain']) & (df['PaymentTier'] == 1 ), 'PaymentExprerience_Cat'] = 'high_senior'
df.loc[( 0 <= df['ExperienceInCurrentDomain']) & (1 >= df['ExperienceInCurrentDomain']) & (df['PaymentTier'] == 2 ), 'PaymentExprerience_Cat'] = 'medium_newgraduate'
df.loc[( 2 <= df['ExperienceInCurrentDomain']) & (3 >= df['ExperienceInCurrentDomain']) & (df['PaymentTier'] == 2 ), 'PaymentExprerience_Cat'] = 'medium_junior'
df.loc[( 4 <= df['ExperienceInCurrentDomain']) & (5 >= df['ExperienceInCurrentDomain']) & (df['PaymentTier'] == 2 ), 'PaymentExprerience_Cat'] = 'medium_middle'
df.loc[( 6 <= df['ExperienceInCurrentDomain']) & (7 >= df['ExperienceInCurrentDomain']) & (df['PaymentTier'] == 2 ), 'PaymentExprerience_Cat'] = 'medium_senior'
df.loc[( 0 <= df['ExperienceInCurrentDomain']) & (1 >= df['ExperienceInCurrentDomain']) & (df['PaymentTier'] == 3 ), 'PaymentExprerience_Cat'] = 'low_newgraduate'
df.loc[( 2 <= df['ExperienceInCurrentDomain']) & (3 >= df['ExperienceInCurrentDomain']) & (df['PaymentTier'] == 3 ), 'PaymentExprerience_Cat'] = 'low_junior'
df.loc[( 4 <= df['ExperienceInCurrentDomain']) & (5 >= df['ExperienceInCurrentDomain']) & (df['PaymentTier'] == 3 ), 'PaymentExprerience_Cat'] = 'low_middle'
df.loc[( 6 <= df['ExperienceInCurrentDomain']) & (7 >= df['ExperienceInCurrentDomain']) & (df['PaymentTier'] == 3 ), 'PaymentExprerience_Cat'] = 'low_senior'

# Yaş gruplarına göre son 1 ayda projede yer alıp almadığını gruplandırdık.
df.loc[(df['EverBenched'] == 'No') & (22 <= df['Age']) & (df['Age'] <= 26), 'New_Age_Bench'] = 'no_1'
df.loc[(df['EverBenched'] == 'No') & (27 <= df['Age']) & (df['Age'] <= 31), 'New_Age_Bench'] = 'no_2'
df.loc[(df['EverBenched'] == 'No') & (32 <= df['Age']) & (df['Age'] <= 36), 'New_Age_Bench'] = 'no_3'
df.loc[(df['EverBenched'] == 'No') & (37 <= df['Age']) & (df['Age'] <= 41), 'New_Age_Bench'] = 'no_4'
df.loc[(df['EverBenched'] == 'Yes') & (22 <= df['Age']) & (df['Age'] <= 26), 'New_Age_Bench'] = 'yes_1'
df.loc[(df['EverBenched'] == 'Yes') & (27 <= df['Age']) & (df['Age'] <= 31), 'New_Age_Bench'] = 'yes_2'
df.loc[(df['EverBenched'] == 'Yes') & (32 <= df['Age']) & (df['Age'] <= 36), 'New_Age_Bench'] = 'yes_3'
df.loc[(df['EverBenched'] == 'Yes') & (37 <= df['Age']) & (df['Age'] <= 41), 'New_Age_Bench'] = 'yes_4'

# Mezuniyet durumlarını maaş kademesine göre gruplandırdık.
df.loc[(df['Education'] == 'Bachelors') & (df['PaymentTier'] == 1 ), 'PaymentExprerience_Cat'] = 'high_bachelors'
df.loc[(df['Education'] == 'Bachelors') & (df['PaymentTier'] == 2 ), 'PaymentExprerience_Cat'] = 'medium_bachelors'
df.loc[(df['Education'] == 'Bachelors') & (df['PaymentTier'] == 3 ), 'PaymentExprerience_Cat'] = 'low_bachelors'
df.loc[(df['Education'] == 'Masters') & (df['PaymentTier'] == 1 ), 'PaymentExprerience_Cat'] = 'high_masters'
df.loc[(df['Education'] == 'Masters') & (df['PaymentTier'] == 2 ), 'PaymentExprerience_Cat'] = 'medium_masters'
df.loc[(df['Education'] == 'Masters') & (df['PaymentTier'] == 3 ), 'PaymentExprerience_Cat'] = 'low_masters'
df.loc[(df['Education'] == 'PHD') & (df['PaymentTier'] == 1 ), 'PaymentExprerience_Cat'] = 'high_PHD'
df.loc[(df['Education'] == 'PHD') & (df['PaymentTier'] == 2 ), 'PaymentExprerience_Cat'] = 'medium_PHD'
df.loc[(df['Education'] == 'PHD') & (df['PaymentTier'] == 3 ), 'PaymentExprerience_Cat'] = 'low_PHD'

# İşe giriş yıllarına göre maaş kademesine göre gruplandırdık.
df.loc[( 2017 == df['JoiningYear']) | (2018 == df['JoiningYear']) & (df['PaymentTier'] == 1 ), 'EmployeeTime'] = 'high_newemployee'
df.loc[( 2016 == df['JoiningYear']) | (2015 == df['JoiningYear']) & (df['PaymentTier'] == 1 ), 'EmployeeTime'] = 'high_midemployee'
df.loc[( 2014 == df['JoiningYear']) | (2013 == df['JoiningYear']) | (2012 == df['JoiningYear']) & (df['PaymentTier'] == 1 ), 'EmployeeTime'] = 'high_oldemployee'
df.loc[( 2017 == df['JoiningYear']) | (2018 == df['JoiningYear']) & (df['PaymentTier'] == 2 ), 'EmployeeTime'] = 'medium_newemployee'
df.loc[( 2016 == df['JoiningYear']) | (2015 == df['JoiningYear']) & (df['PaymentTier'] == 2 ), 'EmployeeTime'] = 'medium_midemployee'
df.loc[( 2014 == df['JoiningYear']) | (2013 == df['JoiningYear']) | (2012 == df['JoiningYear']) & (df['PaymentTier'] == 2 ), 'EmployeeTime'] = 'medium_oldemployee'
df.loc[( 2017 == df['JoiningYear']) | (2018 == df['JoiningYear']) & (df['PaymentTier'] == 3 ), 'EmployeeTime'] = 'low_newemployee'
df.loc[( 2016 == df['JoiningYear']) | (2015 == df['JoiningYear']) & (df['PaymentTier'] == 3 ), 'EmployeeTime'] = 'low_midemployee'
df.loc[( 2014 == df['JoiningYear']) | (2013 == df['JoiningYear']) | (2012 == df['JoiningYear']) & (df['PaymentTier'] == 3 ), 'EmployeeTime'] = 'low_oldemployee'

### AYKIRI DEĞER ###
####################
cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))

### ENCODING ###
################

# Label Encoding
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
                and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

# One-Hot Encoding
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Bağımlı değişken ile verilen threshold değerinin üzerindeki korelasyona sahip değişkenler.
target_correlation_matrix(df, corr_th=0.35, target="LeaveOrNot")

### BASE MODEL KURULUMU ###
############################

y = df["LeaveOrNot"]
X = df.drop(["LeaveOrNot"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True, random_state=1)

lgr = LogisticRegression(random_state=12345)
lgr_model = lgr.fit(X_train, y_train)

# TRAIN ACCURACY
y_pred = lgr_model.predict(X_train)

# Accuracy
accuracy_score(y_train, y_pred)
# 0.8084126496776174

# TEST ACCURACY
y_pred = lgr_model.predict(X_test)
y_prob = lgr_model.predict_proba(X_test)[:, 1]

# Accuracy
accuracy_score(y_test, y_pred)
# 0.7987106017191977

# Precision
precision_score(y_test, y_pred)
# 0.8259587020648967

# Recall
recall_score(y_test, y_pred)
# 0.5577689243027888

# F1
f1_score(y_test, y_pred)
# 0.6658739595719381

print(classification_report(y_test, y_pred))

### MODELLEME ###
#################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=1)

# Tüm Modeller
models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('GBM', GradientBoostingClassifier(random_state=12345))]


# Tüm modellerin test hataları
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    msg = "%s: (%f)" % (name, acc)
    print(msg)
    print(classification_report(y_test, y_pred))

# LR: (0.794844)
# KNN: (0.803437)
# CART: (0.825994)
# RF: (0.820623)
# SVM: (0.815252)
# GBM: (0.847476)


## AUTOMATED HYPERPARAMETER OPTIMIZATION WITH BEST MODEL ##
###########################################################

gbm_model = GradientBoostingClassifier(random_state=17)

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model,
                             gbm_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=True).fit(X, y)

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_,
                                 random_state=17, ).fit(X, y)

cv_results = cross_validate(gbm_final,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])


cv_results['test_accuracy'].mean()
# 0.8521380774378144
cv_results['test_f1'].mean()
# 0.754682110176115
cv_results['test_roc_auc'].mean()
# 0.872948117566699


## FEATURE IMPORTANCE ##
########################

def plot_importance(model, features, num=len(X), save=False):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(gbm_final, X_train)
plot_importance(gbm_final, X)

## Prediction for a New Observation ##
######################################

X.columns
random_user = X.sample(1, random_state=45)
gbm_final.predict(random_user)

