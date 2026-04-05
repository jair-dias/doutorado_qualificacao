import pandas as pd
import ast
import numpy as np

# Carregar o arquivo com tratamento de tipos mistos
print("Carregando arquivo...")
df = pd.read_csv('grid_search_random_forest_results.csv', header=None, low_memory=False)

print(f"✅ Arquivo carregado! Total de combinações: {len(df):,}")
print(f"Shape: {df.shape}")
print(f"Número de colunas: {len(df.columns)}")

# Mostrar amostra das primeiras linhas para entender estrutura
print("\n📋 AMOSTRA DAS PRIMEIRAS LINHAS:")
print("-" * 50)
for i in range(min(5, len(df))):
    print(f"Linha {i}: {df.iloc[i, :5].tolist()}...")

# Baseado na sua estrutura (ajuste conforme necessário):
# 0-3: tempos, 4: ccp_alpha, 5: class_weight, 6: criterion, 7: max_depth,
# 8: max_features, 9: min_samples_leaf, 10: min_samples_split,
# 11: min_weight_fraction_leaf, 12: n_estimators, 13: params_string,
# 14: accuracy, 15: precision, 16: recall, 17: f1, ...

# Primeiro, vamos identificar automaticamente as colunas
print("\n🔍 IDENTIFICANDO COLUNAS AUTOMATICAMENTE...")

# Procurar coluna de parâmetros (contém dicionário)
param_col = None
for i in range(len(df.columns)):
    try:
        sample = str(df.iloc[0, i])
        if '{' in sample and '}' in sample:
            param_col = i
            print(f"✅ Coluna de parâmetros identificada: {i}")
            break
    except:
        continue

# Procurar colunas de métricas (valores entre 0 e 1)
accuracy_col = None
recall_col = None
precision_col = None
f1_col = None

for i in range(len(df.columns)):
    try:
        # Tentar converter para numérico
        val = pd.to_numeric(df.iloc[0, i], errors='coerce')
        if val is not None and 0 <= val <= 1:
            # Usar valores da sua linha de exemplo para identificar
            if 0.92 <= val <= 0.93:  # accuracy = 0.9230769230769231
                accuracy_col = i
                print(f"✅ Coluna de accuracy identificada: {i} (valor={val})")
            elif 0.91 <= val <= 0.92:  # recall = 0.9166666666666666
                recall_col = i
                print(f"✅ Coluna de recall identificada: {i} (valor={val})")
            elif val == 1.0:  # precision = 1.0
                precision_col = i
                print(f"✅ Coluna de precision identificada: {i} (valor={val})")
            elif 0.91 <= val <= 0.92:  # f1 = 0.9166666666666666
                f1_col = i
                print(f"✅ Coluna de f1 identificada: {i} (valor={val})")
    except:
        continue

# Se não encontrou automaticamente, usar índices padrão baseados na sua descrição
if param_col is None:
    param_col = 13

if accuracy_col is None:
    accuracy_col = 14

if recall_col is None:
    recall_col = 16

if precision_col is None:
    precision_col = 15

if f1_col is None:
    f1_col = 17

print(f"\n🎯 COLUNAS SELECIONADAS:")
print(f"   Parâmetros: col{param_col}")
print(f"   Accuracy: col{accuracy_col}")
print(f"   Recall: col{recall_col}")
print(f"   Precision: col{precision_col}")
print(f"   F1-Score: col{f1_col}")

# CONVERTER COLUNAS NUMÉRICAS PARA FLOAT
print("\n🔄 Convertendo colunas numéricas...")

# Converter colunas de métricas para float
for col in [accuracy_col, recall_col, precision_col, f1_col]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Verificar se há valores NaN após conversão
print(f"Valores NaN em accuracy: {df[accuracy_col].isna().sum()}")
print(f"Valores NaN em recall: {df[recall_col].isna().sum()}")

# Remover linhas com valores NaN nas métricas importantes
df_clean = df.dropna(subset=[accuracy_col, recall_col])
print(f"Linhas após limpeza: {len(df_clean):,}")

# Top 5 por Accuracy
print("\n" + "=" * 70)
print("5 MELHORES POR ACCURACY - RANDOM FOREST")
print("=" * 70)

df_sorted_acc = df_clean.sort_values(accuracy_col, ascending=False)
top5_acc = df_sorted_acc.head(5)

for i in range(len(top5_acc)):
    print(f"\n🎯 RESULTADO #{i + 1}")
    print(f"   Accuracy: {top5_acc.iloc[i, accuracy_col]:.4f}")
    print(f"   Recall: {top5_acc.iloc[i, recall_col]:.4f}")
    print(f"   Precision: {top5_acc.iloc[i, precision_col]:.4f}")
    print(f"   F1-Score: {top5_acc.iloc[i, f1_col]:.4f}")

    # Extrair parâmetros
    try:
        params_str = str(top5_acc.iloc[i, param_col])
        params = ast.literal_eval(params_str)
        print("\n   ⚙️ HIPERPARÂMETROS DO RANDOM FOREST:")

        print("   🔧 PARÂMETROS PRINCIPAIS:")
        if 'n_estimators' in params:
            print(f"      n_estimators: {params['n_estimators']} (número de árvores)")
        if 'criterion' in params:
            print(f"      criterion: {params['criterion']}")
        if 'max_depth' in params:
            print(f"      max_depth: {params['max_depth']}")
        if 'max_features' in params:
            print(f"      max_features: {params['max_features']}")

        print("\n   🎯 PARÂMETROS DE DIVISÃO:")
        if 'min_samples_split' in params:
            print(f"      min_samples_split: {params['min_samples_split']}")
        if 'min_samples_leaf' in params:
            print(f"      min_samples_leaf: {params['min_samples_leaf']}")

        print("\n   ⚙️ OUTROS PARÂMETROS:")
        outros = ['ccp_alpha', 'class_weight', 'min_weight_fraction_leaf']
        for param in outros:
            if param in params:
                print(f"      {param}: {params[param]}")

    except Exception as e:
        print(f"\n   ❌ Erro ao ler parâmetros: {e}")
        print(f"   String original: {str(top5_acc.iloc[i, param_col])[:200]}...")

    print("-" * 50)

# Top 5 por Recall
print("\n" + "=" * 70)
print("5 MELHORES POR RECALL - RANDOM FOREST")
print("=" * 70)

df_sorted_rec = df_clean.sort_values(recall_col, ascending=False)
top5_rec = df_sorted_rec.head(5)

for i in range(len(top5_rec)):
    print(f"\n🎯 RESULTADO #{i + 1}")
    print(f"   Recall: {top5_rec.iloc[i, recall_col]:.4f}")
    print(f"   Accuracy: {top5_rec.iloc[i, accuracy_col]:.4f}")
    print(f"   Precision: {top5_rec.iloc[i, precision_col]:.4f}")
    print(f"   F1-Score: {top5_rec.iloc[i, f1_col]:.4f}")

    try:
        params_str = str(top5_rec.iloc[i, param_col])
        params = ast.literal_eval(params_str)
        print(f"\n   n_estimators: {params.get('n_estimators', 'N/A')}")
        print(f"   criterion: {params.get('criterion', 'N/A')}")
        print(f"   max_depth: {params.get('max_depth', 'N/A')}")
    except:
        print("\n   ❌ Erro ao ler parâmetros")

    print("-" * 50)

# Salvar resultados
print("\n💾 SALVANDO RESULTADOS...")

# Criar DataFrames com nomes de colunas apropriados para salvar
top5_acc_df = pd.DataFrame({
    'accuracy': top5_acc.iloc[:, accuracy_col].values,
    'recall': top5_acc.iloc[:, recall_col].values,
    'precision': top5_acc.iloc[:, precision_col].values,
    'f1': top5_acc.iloc[:, f1_col].values,
    'params': top5_acc.iloc[:, param_col].values
})

top5_rec_df = pd.DataFrame({
    'recall': top5_rec.iloc[:, recall_col].values,
    'accuracy': top5_rec.iloc[:, accuracy_col].values,
    'precision': top5_rec.iloc[:, precision_col].values,
    'f1': top5_rec.iloc[:, f1_col].values,
    'params': top5_rec.iloc[:, param_col].values
})

try:
    top5_acc_df.to_csv('melhores_5_random_forest_accuracy.csv', index=False)
    top5_rec_df.to_csv('melhores_5_random_forest_recall.csv', index=False)
    print("✅ Arquivos salvos com sucesso!")
    print("   - melhores_5_random_forest_accuracy.csv")
    print("   - melhores_5_random_forest_recall.csv")
except Exception as e:
    print(f"⚠️ Erro ao salvar arquivos: {e}")

# Estatísticas
print("\n📊 ESTATÍSTICAS GERAIS:")
print(f"Melhor accuracy: {df_clean[accuracy_col].max():.4f}")
print(f"Melhor recall: {df_clean[recall_col].max():.4f}")
print(f"Accuracy média: {df_clean[accuracy_col].mean():.4f}")
print(f"Recall médio: {df_clean[recall_col].mean():.4f}")