import pandas as pd
import ast
import os

# Caminho do arquivo MLP
caminho_arquivo = r'C:\Users\Usuario\PycharmProjects\codigos para busca 5 melhores desempenhos\grid_search_mlp_results.csv'

# Carregar os dados
df = pd.read_csv(caminho_arquivo)
print(f"Arquivo MLP carregado com sucesso! Shape: {df.shape}")

# Usar nomes genéricos
df.columns = [f'col{i}' for i in range(len(df.columns))]

print("\n📋 ESTRUTURA DAS PRIMEIRAS LINHAS:")
print("-" * 50)
for i in range(min(5, len(df.columns))):
    print(f"col{i}: Primeiro valor = {df[f'col{i}'].iloc[0]}")

# IDENTIFICAÇÃO DAS COLUNAS PARA MLP (baseado na estrutura que você mostrou):
# Exemplo: "0.4598665714263916,0.014856417759408295,0.0017679214477539062,6.097354610609228e-05,relu,0.001,80,0.8,0.995,1e-07,"(10, 30, 10)",constant,0.0001,1000,adam,"{'activation': 'relu', ...}",1.0,1.0,1.0,0.8333333333333334,...

# Vamos identificar automaticamente as colunas importantes:
print("\n🔍 IDENTIFICANDO COLUNAS AUTOMATICAMENTE...")

# Procurar coluna de parâmetros (contém dicionário)
param_col = None
for i in range(len(df.columns)):
    sample = str(df[f'col{i}'].iloc[0])
    if '{' in sample and '}' in sample and 'activation' in sample.lower():
        param_col = i
        print(f"✅ Coluna de parâmetros identificada: col{param_col}")
        break

# Procurar colunas de métricas (valores entre 0 e 1)
accuracy_col = None
precision_col = None
recall_col = None
f1_col = None

for i in range(len(df.columns)):
    try:
        val = df[f'col{i}'].iloc[0]
        # Verificar se é float entre 0 e 1
        if isinstance(val, (int, float)) and 0 <= val <= 1:
            # Tentar identificar pelo valor da primeira linha
            if val == 1.0:  # Na sua linha, accuracy é 1.0
                if accuracy_col is None:
                    accuracy_col = i
                    print(f"✅ Coluna de accuracy identificada: col{accuracy_col} (valor={val})")
            elif abs(val - 0.8333333333333334) < 0.0001:  # Recall na sua linha
                if recall_col is None:
                    recall_col = i
                    print(f"✅ Coluna de recall identificada: col{recall_col} (valor={val})")
            elif val == 1.0 and i != accuracy_col:  # Outro 1.0 pode ser precision
                if precision_col is None:
                    precision_col = i
                    print(f"✅ Coluna de precision identificada: col{precision_col} (valor={val})")
    except:
        continue

# Se não encontrou todas, usar índices padrão baseados na sua descrição
if param_col is None:
    param_col = 15  # Baseado na estrutura: params_string está na coluna 15

if accuracy_col is None:
    accuracy_col = 16  # Accuracy após params_string

if precision_col is None:
    precision_col = 17  # Precision após accuracy

if recall_col is None:
    recall_col = 18  # Recall após precision

if f1_col is None:
    f1_col = 19  # F1 após recall

print(f"\n📊 COLUNAS SELECIONADAS:")
print(f"   Parâmetros: col{param_col}")
print(f"   Accuracy: col{accuracy_col}")
print(f"   Precision: col{precision_col}")
print(f"   Recall: col{recall_col}")
print(f"   F1-Score: col{f1_col}")

# Ordenar pela coluna de accuracy
df_sorted = df.sort_values(f'col{accuracy_col}', ascending=False)
top_5 = df_sorted.head(5)

print("\n" + "=" * 80)
print("5 MELHORES RESULTADOS DO GRID SEARCH - MLP (Multilayer Perceptron)")
print("=" * 80)

for i, (index, row) in enumerate(top_5.iterrows(), 1):
    print(f"\n🎯 RESULTADO #{i}")
    print(f"📊 Accuracy: {row[f'col{accuracy_col}']:.4f}")
    print(f"🎯 Precision: {row[f'col{precision_col}']:.4f}")
    print(f"🔍 Recall: {row[f'col{recall_col}']:.4f}")
    print(f"⭐ F1-Score: {row[f'col{f1_col}']:.4f}")

    print("\n⚙️ HIPERPARÂMETROS DO MLP:")
    try:
        params = ast.literal_eval(row[f'col{param_col}'])

        # Organizar por categorias para melhor visualização
        print("   📈 ARQUITETURA DA REDE:")
        if 'hidden_layer_sizes' in params:
            print(f"      hidden_layer_sizes: {params['hidden_layer_sizes']}")
        if 'activation' in params:
            print(f"      activation: {params['activation']}")
        if 'solver' in params:
            print(f"      solver: {params['solver']}")
        if 'max_iter' in params:
            print(f"      max_iter: {params['max_iter']}")

        print("\n   ⚡ PARÂMETROS DE TREINAMENTO:")
        if 'learning_rate' in params:
            print(f"      learning_rate: {params['learning_rate']}")
        if 'learning_rate_init' in params:
            print(f"      learning_rate_init: {params['learning_rate_init']}")
        if 'batch_size' in params:
            print(f"      batch_size: {params['batch_size']}")
        if 'alpha' in params:
            print(f"      alpha (regularização L2): {params['alpha']}")

        print("\n   🔧 PARÂMETROS AVANÇADOS:")
        if 'beta_1' in params:
            print(f"      beta_1: {params['beta_1']}")
        if 'beta_2' in params:
            print(f"      beta_2: {params['beta_2']}")
        if 'epsilon' in params:
            print(f"      epsilon: {params['epsilon']}")

        # Mostrar outros parâmetros não categorizados
        outros_params = {k: v for k, v in params.items()
                         if k not in ['hidden_layer_sizes', 'activation', 'solver', 'max_iter',
                                      'learning_rate', 'learning_rate_init', 'batch_size', 'alpha',
                                      'beta_1', 'beta_2', 'epsilon']}
        if outros_params:
            print("\n   📝 OUTROS PARÂMETROS:")
            for key, value in outros_params.items():
                print(f"      {key}: {value}")

    except Exception as e:
        print(f"   ❌ Erro ao ler parâmetros: {e}")
        print(f"   String original: {row[f'col{param_col}'][:200]}...")

    print("-" * 80)

# Mostrar também os 5 melhores em formato de tabela resumida
print("\n" + "=" * 80)
print("RESUMO DOS 5 MELHORES RESULTADOS MLP")
print("=" * 80)

resumo = top_5[[f'col{accuracy_col}', f'col{precision_col}',
                f'col{recall_col}', f'col{f1_col}',
                f'col{param_col}']].copy()
resumo.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Hiperparâmetros']
print(resumo.to_string(index=False))

# Salvar os melhores resultados
try:
    output_path = r'C:\Users\Usuario\PycharmProjects\codigos para busca 5 melhores desempenhos\melhores_5_resultados_mlp.csv'
    top_5.to_csv(output_path, index=False)
    print(f"\n💾 Resultados salvos em: {output_path}")
except Exception as e:
    print(f"\n⚠️ Não foi possível salvar o arquivo: {e}")

# Estatísticas adicionais
print(f"\n📈 ESTATÍSTICAS DO GRID SEARCH MLP:")
print(f"   Total de combinações testadas: {len(df):,}")
print(f"   Melhor accuracy: {df[f'col{accuracy_col}'].max():.4f}")
print(f"   Pior accuracy: {df[f'col{accuracy_col}'].min():.4f}")
print(f"   Accuracy média: {df[f'col{accuracy_col}'].mean():.4f}")
print(f"   Melhor recall: {df[f'col{recall_col}'].max():.4f}")
print(f"   Pior recall: {df[f'col{recall_col}'].min():.4f}")
print(f"   Recall médio: {df[f'col{recall_col}'].mean():.4f}")

# Análise adicional: parâmetros mais frequentes nos top 5
print(f"\n🔍 ANÁLISE DOS HIPERPARÂMETROS MAIS FREQUENTES NOS TOP 5:")

parametros_comuns = {}
for _, row in top_5.iterrows():
    try:
        params = ast.literal_eval(row[f'col{param_col}'])
        for key, value in params.items():
            if key not in parametros_comuns:
                parametros_comuns[key] = []
            parametros_comuns[key].append(str(value))
    except:
        continue

print("   Parâmetro | Valor mais comum | Frequência nos Top 5")
print("   " + "-" * 50)
for key, values in parametros_comuns.items():
    from collections import Counter

    counter = Counter(values)
    valor_mais_comum, frequencia = counter.most_common(1)[0]
    print(f"   {key:15} | {str(valor_mais_comum)[:20]:20} | {frequencia}/5 ({frequencia / 5 * 100:.0f}%)")