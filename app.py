# Fernando Inomata e Joao Gabriel Fazio
# Identificação de Espécies de Animais
import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Função para carregar os dados
def load_data():
    animal_data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'DadosAnimais.csv')
    class_data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Classes.csv')
    animal_data = pd.read_csv(animal_data_path)
    class_data = pd.read_csv(class_data_path)
    
    # Limpeza de dados
    animal_data.dropna(inplace=True)  # Remover linhas com valores ausentes
    class_data.dropna(inplace=True)
    
    return animal_data, class_data

# Função para engenharia de características
def feature_engineering(animal_data):
    # Exemplo: criar uma nova característica baseada em outras
    animal_data['has_four_or_more_legs'] = (animal_data['legs'] >= 4).astype(int)
    return animal_data

# Função para balanceamento de dados
def balance_data(animal_data):
    class_counts = animal_data['class_type'].value_counts()
    min_count = class_counts.min()
    balanced_data = animal_data.groupby('class_type').apply(lambda x: x.sample(min_count)).reset_index(drop=True)
    return balanced_data

# Função para treinar e avaliar o modelo
def train_and_evaluate(animal_data, class_data):
    animal_data = feature_engineering(animal_data)
    animal_data = balance_data(animal_data)
    
    X = animal_data.drop(columns=['animal_name', 'class_type']).values
    y = animal_data['class_type'].values

    # Normalização
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Definição do modelo
    mlp = MLPClassifier(random_state=42)

    # Definição dos hiperparâmetros para busca em grade
    param_grid = {
        'hidden_layer_sizes': [(64,), (64, 32), (64, 32, 16)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'max_iter': [500, 1000]
    }

    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    best_mlp = grid_search.best_estimator_

    # Avaliação do modelo
    accuracy = cross_val_score(best_mlp, X, y, cv=5, scoring='accuracy').mean()
    
    # Geração da matriz de confusão
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_mlp.fit(X_train, y_train)
    y_pred = best_mlp.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('static/confusion_matrix.png')
    plt.close()

    return accuracy

# Carregar os dados e treinar o modelo ao iniciar o app
animal_data, class_data = load_data()
accuracy = train_and_evaluate(animal_data, class_data)
print(f"Modelo treinado com precisão: {accuracy:.2f}")

@app.route('/')
def index():
    return redirect(url_for('identify'))

@app.route('/identify', methods=['GET', 'POST'])
def identify():
    questions = [
        {"question": "Tem pelo?", "column": "hair"},
        {"question": "Tem pena?", "column": "feathers"},
        {"question": "Bota ovo?", "column": "eggs"},
        {"question": "Dá leite?", "column": "milk"},
        {"question": "Pode voar?", "column": "airborne"},
        {"question": "É aquático?", "column": "aquatic"},
        {"question": "É um predador?", "column": "predator"},
        {"question": "Tem dentes?", "column": "toothed"},
        {"question": "Tem coluna vertebral?", "column": "backbone"},
        {"question": "Ele respira ar?", "column": "breathes"},
        {"question": "É venenoso?", "column": "venomous"},
        {"question": "Tem brânquias?", "column": "fins"},
        {"question": "O animal tem 4 ou mais pernas?", "column": "has_four_or_more_legs"},
        {"question": "Tem cauda?", "column": "tail"},
        {"question": "É um animal doméstico?", "column": "domestic"},
        {"question": "É maior que um gato?", "column": "catsize"},
    ]
    
    if request.method == 'POST':
        answers = {q['column']: request.form[q['column']] for q in questions}
        filtered_animals = filter_animals(answers, animal_data)
        
        if not filtered_animals.empty:
            identified_animal_name = filtered_animals.iloc[0]['animal_name']
            identified_class_number = filtered_animals.iloc[0]['class_type']
            
            identified_class_type = class_data.loc[class_data['Class_Number'] == identified_class_number, 'Class_Type'].values[0]
        else:
            identified_animal_name = "Animal não encontrado"
            identified_class_type = "Não encontrado"

        return render_template('result_identification.html', 
                               identified_class=identified_class_type, 
                               identified_animal=identified_animal_name)

    return render_template('identify.html', questions=questions)

def filter_animals(answers, animal_data):
    filtered_animals = animal_data.copy()
    for question, answer in answers.items():
        if answer == 'yes':
            filtered_animals = filtered_animals[filtered_animals[question] == 1]
        elif answer == 'no':
            filtered_animals = filtered_animals[filtered_animals[question] == 0]
    return filtered_animals

if __name__ == '__main__':
    app.run(debug=True)
