import pandas as pd
import ast
import os

# Caminho do arquivo
caminho_arquivo = r'C:\Users\Usuario\PycharmProjects\codigos para busca 5 melhores desempenhos\grid_search_decision_tree_results.csv'

# Carregar os dados
df = pd.read_csv(caminho_arquivo)
print(f"Arquivo carregado com sucesso! Shape: {df.shape}")

# Usar nomes genéricos
df.columns = [f'col{i}' for i in range(len(df.columns))]

# IDENTIFICAÇÃO MANUAL DAS COLUNAS (baseado na sua saída):
# col16 = Accuracy (0.9230769230769232)
# col15 = Parâmetros (dicionário com hiperparâmetros)
# col17 = Precision (1.0)
# col18 = Recall (0.9166666666666666)
# col19 = F1-Score (0.9166666666666666)

# Ordenar pela coluna de accuracy (col16)
df_sorted = df.sort_values('col16', ascending=False)
top_5 = df_sorted.head(5)

print("\n" + "=" * 70)
print("5 MELHORES RESULTADOS DO GRID SEARCH - DECISION TREE")
print("=" * 70)

for i, (index, row) in enumerate(top_5.iterrows(), 1):
    print(f"\n🎯 RESULTADO #{i}")
    print(f"📊 Accuracy: {row['col16']:.4f}")
    print(f"🎯 Precision: {row['col17']:.4f}")
    print(f"🔍 Recall: {row['col18']:.4f}")
    print(f"⭐ F1-Score: {row['col19']:.4f}")

    print("\n⚙️ HIPERPARÂMETROS:")
    try:
        params = ast.literal_eval(row['col15'])
        for key, value in params.items():
            print(f"   {key}: {value}")
    except:
        print(f"   {row['col15']}")

    print("-" * 70)

# Mostrar também os 5 melhores em formato de tabela resumida
print("\n" + "=" * 70)
print("RESUMO DOS 5 MELHORES RESULTADOS")
print("=" * 70)

resumo = top_5[['col16', 'col17', 'col18', 'col19', 'col15']].copy()
resumo.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Hiperparâmetros']
print(resumo.to_string(index=False))

# Salvar os melhores resultados
try:
    top_5.to_csv(
        r'C:\Users\Usuario\PycharmProjects\codigos para busca 5 melhores desempenhos\melhores_5_resultados_decision.csv',
        index=False)
    print(f"\n💾 Resultados salvos em: melhores_5_resultados_decision.csv")
except Exception as e:
    print(f"\n⚠️ Não foi possível salvar o arquivo: {e}")

# Estatísticas adicionais
print(f"\n📈 ESTATÍSTICAS DO GRID SEARCH:")
print(f"   Total de combinações testadas: {len(df):,}")
print(f"   Melhor accuracy: {df['col16'].max():.4f}")
print(f"   Pior accuracy: {df['col16'].min():.4f}")
print(f"   Accuracy média: {df['col16'].mean():.4f}")