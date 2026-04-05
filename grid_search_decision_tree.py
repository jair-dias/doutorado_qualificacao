# Importação das bibliotecas
print("imports")
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Lendo dados

df = pd.read_excel(r"C:\Users\Usuario\PycharmProjects\Projeto IA\Cópia de Respostas com diagnóstico (.excel).xlsx"r"")
df = df.drop(columns=['Carimbo de data/hora', 'Nome'])
print("read")

# Normalizar colunas idade, peso e altura:
df['Idade'] = df['Idade'].replace([df['Idade']],[preprocessing.normalize([df['Idade']])])

df['Peso'] = df['Peso'].replace([df['Peso']],[preprocessing.normalize([df['Peso']])])

df['Altura'] = df['Altura'].replace([df['Altura']],[preprocessing.normalize([df['Altura']])])

# Tratar colunas 0 ou 1

# Trabalho:
mapeamento = {'Aposentado(b)': 0, 'Ativo(a)': 1}
df['Trabalho'] = df['Trabalho'].map(mapeamento)

# Problema de coluna:
mapeamento = {'Não': 0, 'Sim': 1}
df['Problema de coluna na família'] = df['Problema de coluna na família'].map(mapeamento)

# Perdeu peso:
mapeamento = {'Não': 0, 'Sim': 1}
df['Perdeu peso'] = df['Perdeu peso'].map(mapeamento)

# Cansaço:
mapeamento = {'Não': 0, 'Sim': 1}
df['Cansaço'] = df['Cansaço'].map(mapeamento)

# Urina:
mapeamento = {'Normal pois consigo segurar, mesmo aos esforços como espirrar ou tossir': 0, 'Perco urina sem perceber, mesmo que seja pequenas quantidades': 1}
df['Urina'] = df['Urina'].map(mapeamento)

# Cigarro:
mapeamento = {'Não': 0, 'Sim': 1}
df['Cigarro'] = df['Cigarro'].map(mapeamento)

# Dor de cabeça:
mapeamento = {'Não': 0, 'Sim': 1}
df['Dor de cabeça'] = df['Dor de cabeça'].map(mapeamento)

# Dobrar o corpo:
mapeamento = {'Sim, consigo dobrar': 0, 'Não consigo dobrar quase nada': 1}
df['Dobrar o corpo'] = df['Dobrar o corpo'].map(mapeamento)

# Sexo:
mapeamento = {'M': 0, 'F': 1}
df['sexo'] = df['sexo'].map(mapeamento)

# Tratar colunas que com mais de duas opções, mas só pode marcar uma

colunas = ['Profissão',
           'Levantar peso',
           'Tempo de dor',
           'A dor desce',
           'Dor nas costas e na perna',
           'Exercícios para a dor',
           'Motivação',
           'Rotina de exercícios físicos',
           'Sensibilidade',
           'Pernas fracas',
           'Levantar perna esticada']

for coluna in colunas:
    # Novas colunas com --resposta, e 0 ou 1 como valor
    dummies = pd.get_dummies(df[coluna], prefix=coluna, prefix_sep='--', dtype=int)

    # Junte as novas colunas ao DataFrame
    df = pd.concat([df, dummies], axis=1)

    # Remova a coluna original que já foi transformada
    df = df.drop(coluna, axis=1)

# Tratar coluna com múltiplas opções
coluna_multipla = 'Tem outras doenças'

def limpar_e_separar(texto):
    if not isinstance(texto, str):
        return [] # Retorna lista vazia se for NaN ou não for texto

    # 1. Remove o texto literal '\s*'
    texto_limpo = texto.replace('\\s*', '')

    # 2. Separa os itens pela vírgula
    itens = texto_limpo.split(',')

    # 3. Remove espaços em branco de cada item e remove itens vazios
    itens_finais = [item.strip() for item in itens if item.strip()]
    return itens_finais

# Aplicamos a função de limpeza na coluna
listas_limpas = df[coluna_multipla].apply(limpar_e_separar)

# get_dummies
dummies = pd.get_dummies(listas_limpas.explode())
dummies = dummies.groupby(level=0).max()

prefixo = 'Tem outras doenças --'
dummies = dummies.add_prefix(prefixo)

dummies = dummies.astype(int)

# Junte as novas colunas ao DataFrame
df = pd.concat([df, dummies], axis=1)

# Remova a coluna original que já foi transformada
df = df.drop(coluna_multipla, axis=1)

# Realocando o Target para última coluna e tratando target

last_column = df.pop('diagnóstico')
df.insert(df.shape[1],'diagnóstico',last_column)

mapeamento = {'M54.5': 0, 'M54.4': 1}
df['diagnóstico'] = df['diagnóstico'].map(mapeamento)
print("tratamento finalizado")

# Fisher Exact Test
# Inicializar uma lista para armazenar os resultados
resultados_fisher = []

# Iterar sobre todas as colunas, exceto a última
for coluna in df.columns[3:-1]:
    a = b = c = d = 0
    # Calcular as frequências de cada categoria
    for i in range(df.shape[0]):
        if df[coluna][i] == 1 and df['diagnóstico'][i] == 0:
            a += 1
        elif df[coluna][i] == 1 and df['diagnóstico'][i] == 1:
            b += 1
        elif df[coluna][i] == 0 and df['diagnóstico'][i] == 0:
            c += 1
        elif df[coluna][i] == 0 and df['diagnóstico'][i] == 1:
            d += 1
    # Armazenar os resultados
    data = [[a, b], [c, d]]
    resultado = stats.fisher_exact(data)
    resultados_fisher.append((coluna, resultado[0], resultado[1]))

# Criar DataFrame
df_fisher=pd.DataFrame(resultados_fisher,columns=["Coluna", "Statistic", "P-value"])

# Extrair os valores p e os índices das colunas
p_values = [resultado[2] for resultado in resultados_fisher]
indices_colunas = range(3, len(df.columns)-1)

# Dropando colunas com pvalor>0.05
# Lista para armazenar os índices das colunas a serem removidas
colunas_para_remover = []

# Identificar as colunas com p_values igual a 1
for i, valor_p in enumerate(p_values):
    if valor_p > 0.05:
        colunas_para_remover.append(i)

colunas_para_remover = [x + 3 for x in colunas_para_remover]
# +3 porque as primeiras colunas de df_test não são binárias (peso, altura e idade)

# Remover as colunas do DataFrame
df = df.drop(df.columns[colunas_para_remover], axis=1)
print("fisher finalizado")


# Formatando dados

X = df.drop(columns=['diagnóstico'])
y = [int(target) for target in df['diagnóstico'].values]

# GridSearchCV for Decision Tree Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

param_grid = {'criterion': ['gini', 'entropy', 'log_loss'],
              'splitter': ['best', 'random'],
              'max_depth' : [10, 20, 50],
              'min_samples_split': [0.01, 0.1, 0.2, 0.3, 0.4],
              'min_samples_leaf': [0.01, 0.1, 0.15, 0.2],
              'min_weight_fraction_leaf': [0.0, 0.01, 0.05, 0.1, 0.15, 0.2],
              'max_features': ['sqrt', 'log2'],
              'max_leaf_nodes': [2, 4, 5, 6, 10, 20],
              'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
              'ccp_alpha': [0.1, .01, .001],
              'class_weight': ['balanced', None]
             }
tree_clas = DecisionTreeClassifier()
grid_search_dt = GridSearchCV(estimator=tree_clas, param_grid=param_grid, cv=5, verbose=2, scoring='accuracy',return_train_score=True)
grid_result_dt=grid_search_dt.fit(X_train,y_train)

# DecisionTreeClassifier summarize results
means = grid_result_dt.cv_results_['mean_test_score']
stds = grid_result_dt.cv_results_['std_test_score']

means_train = grid_result_dt.cv_results_['mean_train_score']
stds_train = grid_result_dt.cv_results_['std_train_score']

params = grid_result_dt.cv_results_['params']

for mean, stdev, meantrain, stdtrain, param in zip(means, stds, means_train, stds_train, params):
    print("%f (%f) and train %f (%f) with: %r" % (mean, stdev, meantrain, stdtrain, param))

y_predicted_svc = grid_result_dt.predict(X_test)
print(classification_report(y_test,y_predicted_svc))
print(confusion_matrix(y_test,y_predicted_svc))

print("Best: %f using %s" % (grid_result_dt.best_score_, grid_result_dt.best_params_))


# Salvar resultados em planilha

# Converter o dicionário de resultados para um DataFrame do pandas
results_df = pd.DataFrame(grid_result_dt.cv_results_)

# Ordenar o DataFrame pelos melhores resultados (opcional, mas recomendado)
results_df = results_df.sort_values(by='rank_test_score', ascending=True)

# Definir o nome do arquivo csv de saída
output_filename = "grid_search_decision_tree_results.csv"

# Salvar o DataFrame em um arquivo csv
# O parâmetro index=False evita que o índice do DataFrame seja salvo como uma coluna na planilha
results_df.to_csv(output_filename, index=False)

print(f"Resultados salvos com sucesso no arquivo: {output_filename}")


# não salvou oarquivo porque estava muito grande para excel

#               precision    recall  f1-score   support

#            0       0.87      1.00      0.93        20
#            1       1.00      0.57      0.73         7

#     accuracy                           0.89        27
#    macro avg       0.93      0.79      0.83        27
# weighted avg       0.90      0.89      0.88        27

# [[20  0]
#  [ 3  4]]
# Best: 0.951282 using {'ccp_alpha': 0.001, 'class_weight': None, 'criterion': 'gini', 'max_depth': 20, 'max_features': 'sqrt', 
# 'max_leaf_nodes': 10, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 0.01, 'min_samples_split': 0.1, 'min_weight_fraction_leaf': 0.05, 'splitter': 'random'}