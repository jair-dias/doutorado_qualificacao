import pandas as pd
import ast
import numpy as np

# Carregar o arquivo com tratamento de tipos mistos
print("Carregando arquivo de resultados do SVM...")
df = pd.read_csv('grid_search_svm_results.csv', low_memory=False)

print(f"✅ Arquivo carregado! Total de combinações: {len(df):,}")
print(f"Shape: {df.shape}")
print(f"Número de colunas: {len(df.columns)}")
print(f"Nomes das colunas: {list(df.columns)}")

# Mostrar amostra das primeiras linhas para entender estrutura
print("\n📋 AMOSTRA DAS PRIMEIRAS LINHAS:")
print("-" * 50)
print(df.head(3))

# Identificar colunas relevantes
print("\n🔍 IDENTIFICANDO COLUNAS RELEVANTES...")

# Coluna de parâmetros (contém dicionário)
param_col = 'params' if 'params' in df.columns else None
if param_col is None:
    for col in df.columns:
        if df[col].astype(str).str.contains('{').any():
            param_col = col
            break

print(f"✅ Coluna de parâmetros: {param_col}")

# Identificar colunas de métricas
# No arquivo CSV do SVM, as métricas de teste são: mean_test_score, std_test_score, rank_test_score
# E temos as métricas de treino também

# Coluna de accuracy (no SVM, mean_test_score representa a acurácia)
accuracy_col = 'mean_test_score' if 'mean_test_score' in df.columns else None

# Para recall, precisamos identificar as colunas split0_test_score, split1_test_score, etc.
# Vamos calcular o recall médio se tivermos essas colunas
split_cols = [col for col in df.columns if 'split' in col and 'test_score' in col]
print(f"✅ Colunas de splits encontradas: {len(split_cols)} colunas")

# Se não tivermos uma coluna específica para recall, usaremos mean_test_score como proxy
# No seu caso, parece que mean_test_score é a acurácia média
recall_col = accuracy_col  # Usaremos a mesma coluna como proxy, a menos que tenha recall específico

print(f"\n🎯 COLUNAS SELECIONADAS:")
print(f"   Parâmetros: {param_col}")
print(f"   Accuracy (mean_test_score): {accuracy_col}")
print(f"   Recall (usando accuracy como proxy): {recall_col}")

# CONVERTER COLUNAS NUMÉRICAS PARA FLOAT
print("\n🔄 Convertendo colunas numéricas...")

# Converter colunas de métricas para float
if accuracy_col:
    df[accuracy_col] = pd.to_numeric(df[accuracy_col], errors='coerce')

# Verificar se há valores NaN após conversão
print(f"Valores NaN em accuracy: {df[accuracy_col].isna().sum() if accuracy_col else 'N/A'}")

# Remover linhas com valores NaN nas métricas importantes
df_clean = df.copy()
if accuracy_col:
    df_clean = df_clean.dropna(subset=[accuracy_col])
print(f"Linhas após limpeza: {len(df_clean):,}")

# Top 5 por Accuracy
print("\n" + "=" * 70)
print("5 MELHORES POR ACCURACY - SVM")
print("=" * 70)

if accuracy_col:
    df_sorted_acc = df_clean.sort_values(accuracy_col, ascending=False)
    top5_acc = df_sorted_acc.head(5)

    for i in range(len(top5_acc)):
        print(f"\n🎯 RESULTADO #{i + 1}")
        print(f"   Accuracy (mean_test_score): {top5_acc.iloc[i][accuracy_col]:.4f}")

        # Mostrar também o desvio padrão
        if 'std_test_score' in df.columns:
            print(f"   Desvio padrão: {top5_acc.iloc[i]['std_test_score']:.4f}")

        # Mostrar rank
        if 'rank_test_score' in df.columns:
            print(f"   Rank: {int(top5_acc.iloc[i]['rank_test_score'])}")

        # Tempos de execução
        if 'mean_fit_time' in df.columns:
            print(f"   Tempo médio de treino: {top5_acc.iloc[i]['mean_fit_time']:.4f}s")

        # Extrair parâmetros
        try:
            params_str = str(top5_acc.iloc[i][param_col])
            params = ast.literal_eval(params_str)
            print("\n   ⚙️ HIPERPARÂMETROS DO SVM:")

            print("   🔧 PARÂMETROS PRINCIPAIS:")
            if 'C' in params:
                print(f"      C: {params['C']} (parâmetro de regularização)")
            if 'kernel' in params:
                print(f"      kernel: {params['kernel']}")
            if 'gamma' in params:
                print(f"      gamma: {params['gamma']}")
            if 'class_weight' in params:
                print(f"      class_weight: {params['class_weight']}")

            print("\n   ⚙️ OUTROS PARÂMETROS:")
            outros = ['shrinking', 'probability', 'tol', 'cache_size', 'max_iter']
            for param in outros:
                if param in params:
                    print(f"      {param}: {params[param]}")

            # Mostrar também os parâmetros das colunas individuais
            print("\n   📊 PARÂMETROS DAS COLUNAS:")
            for col in ['param_C', 'param_kernel', 'param_gamma', 'param_class_weight']:
                if col in df.columns and not pd.isna(top5_acc.iloc[i][col]):
                    print(f"      {col}: {top5_acc.iloc[i][col]}")

        except Exception as e:
            print(f"\n   ❌ Erro ao ler parâmetros: {e}")
            print(f"   String original: {str(top5_acc.iloc[i][param_col])[:200]}...")

        print("-" * 50)
else:
    print("❌ Coluna de accuracy não encontrada!")

# Top 5 por Recall (usando a mesma coluna como proxy)
print("\n" + "=" * 70)
print("5 MELHORES POR RECALL - SVM (usando accuracy como proxy)")
print("=" * 70)

if recall_col:
    df_sorted_rec = df_clean.sort_values(recall_col, ascending=False)
    top5_rec = df_sorted_rec.head(5)

    for i in range(len(top5_rec)):
        print(f"\n🎯 RESULTADO #{i + 1}")
        print(f"   Recall/Accuracy: {top5_rec.iloc[i][recall_col]:.4f}")

        if 'std_test_score' in df.columns:
            print(f"   Desvio padrão: {top5_rec.iloc[i]['std_test_score']:.4f}")

        # Mostrar scores individuais dos splits
        split_scores = []
        for col in split_cols:
            if col in df.columns and not pd.isna(top5_rec.iloc[i][col]):
                split_scores.append(top5_rec.iloc[i][col])

        if split_scores:
            print(f"   Scores dos splits: {[f'{s:.3f}' for s in split_scores]}")

        try:
            params_str = str(top5_rec.iloc[i][param_col])
            params = ast.literal_eval(params_str)
            print(f"\n   ⚙️ Parâmetros principais:")
            print(f"      C: {params.get('C', 'N/A')}")
            print(f"      kernel: {params.get('kernel', 'N/A')}")
            print(f"      gamma: {params.get('gamma', 'N/A')}")
            print(f"      class_weight: {params.get('class_weight', 'N/A')}")

            # Mostrar também da coluna param_C se existir
            if 'param_C' in df.columns:
                print(f"      param_C: {top5_rec.iloc[i]['param_C']}")

        except Exception as e:
            print(f"\n   ❌ Erro ao ler parâmetros: {e}")

        print("-" * 50)
else:
    print("❌ Coluna de recall não encontrada!")

# Análise por tipo de kernel
print("\n" + "=" * 70)
print("ANÁLISE POR TIPO DE KERNEL")
print("=" * 70)

if 'param_kernel' in df.columns:
    kernels = df_clean['param_kernel'].unique()
    print(f"Kernels encontrados: {list(kernels)}")

    for kernel in kernels:
        kernel_df = df_clean[df_clean['param_kernel'] == kernel]
        if len(kernel_df) > 0:
            print(f"\n📊 Kernel: {kernel}")
            print(f"   Número de combinações: {len(kernel_df)}")
            print(f"   Melhor accuracy: {kernel_df[accuracy_col].max():.4f}" if accuracy_col else "")
            print(f"   Accuracy média: {kernel_df[accuracy_col].mean():.4f}" if accuracy_col else "")

            # Top 3 para este kernel
            top_kernel = kernel_df.sort_values(accuracy_col, ascending=False).head(
                3) if accuracy_col else kernel_df.head(3)
            for idx, row in top_kernel.iterrows():
                print(f"   - C={row.get('param_C', 'N/A')}, gamma={row.get('param_gamma', 'N/A')}, "
                      f"accuracy={row[accuracy_col]:.4f}" if accuracy_col else "")

# Salvar resultados
print("\n💾 SALVANDO RESULTADOS...")

if accuracy_col:
    # Criar DataFrames com nomes de colunas apropriados para salvar
    top5_acc_df = pd.DataFrame()
    for i, row in top5_acc.iterrows():
        acc_data = {
            'accuracy': row[accuracy_col],
            'rank': row.get('rank_test_score', ''),
            'std_test_score': row.get('std_test_score', ''),
            'mean_fit_time': row.get('mean_fit_time', ''),
            'params': row[param_col] if param_col else ''
        }

        # Adicionar parâmetros individuais
        param_cols = ['param_C', 'param_kernel', 'param_gamma', 'param_class_weight', 'param_shrinking']
        for pcol in param_cols:
            if pcol in df.columns:
                acc_data[pcol] = row[pcol]

        top5_acc_df = pd.concat([top5_acc_df, pd.DataFrame([acc_data])], ignore_index=True)

if recall_col:
    top5_rec_df = pd.DataFrame()
    for i, row in top5_rec.iterrows():
        rec_data = {
            'recall_accuracy': row[recall_col],
            'rank': row.get('rank_test_score', ''),
            'std_test_score': row.get('std_test_score', ''),
            'mean_fit_time': row.get('mean_fit_time', ''),
            'params': row[param_col] if param_col else ''
        }

        # Adicionar parâmetros individuais
        param_cols = ['param_C', 'param_kernel', 'param_gamma', 'param_class_weight', 'param_shrinking']
        for pcol in param_cols:
            if pcol in df.columns:
                rec_data[pcol] = row[pcol]

        top5_rec_df = pd.concat([top5_rec_df, pd.DataFrame([rec_data])], ignore_index=True)

try:
    if accuracy_col and 'top5_acc_df' in locals():
        top5_acc_df.to_csv('melhores_5_svm_accuracy.csv', index=False)
        print("✅ Arquivo salvo: melhores_5_svm_accuracy.csv")

    if recall_col and 'top5_rec_df' in locals():
        top5_rec_df.to_csv('melhores_5_svm_recall.csv', index=False)
        print("✅ Arquivo salvo: melhores_5_svm_recall.csv")
except Exception as e:
    print(f"⚠️ Erro ao salvar arquivos: {e}")

# Estatísticas
print("\n📊 ESTATÍSTICAS GERAIS DO SVM:")
if accuracy_col:
    print(f"Melhor accuracy: {df_clean[accuracy_col].max():.4f}")
    print(f"Accuracy média: {df_clean[accuracy_col].mean():.4f}")
    print(f"Desvio padrão médio: {df_clean['std_test_score'].mean():.4f}" if 'std_test_score' in df.columns else "")
    print(
        f"Número de combinações únicas com accuracy máxima: {len(df_clean[df_clean[accuracy_col] == df_clean[accuracy_col].max()])}")

# Análise dos melhores parâmetros
print("\n🔍 ANÁLISE DOS MELHORES PARÂMETROS:")

if accuracy_col and len(df_clean) > 0:
    best_idx = df_clean[accuracy_col].idxmax()
    best_row = df_clean.loc[best_idx]

    print(f"Melhor accuracy encontrada: {best_row[accuracy_col]:.4f}")

    if param_col:
        try:
            best_params = ast.literal_eval(str(best_row[param_col]))
            print(f"\n⚙️ Melhores hiperparâmetros do SVM:")
            for key, value in best_params.items():
                print(f"   {key}: {value}")
        except:
            print(f"\nParâmetros: {best_row[param_col]}")

    print(f"\n📈 Performance detalhada:")
    for col in ['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time']:
        if col in df.columns:
            print(f"   {col}: {best_row[col]:.4f}")

print("\n✅ Análise do SVM concluída!")