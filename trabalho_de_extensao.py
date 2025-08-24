# -*- coding: utf-8 -*-
"""
Análise Estatística para Previsão de Demanda - Açaí do Bairro
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import os

# Configuração inicial
np.random.seed(42)
plt.style.use('default')  # Alterado para o estilo padrão para evitar problemas
dias = 60  # Número de dias

# Mostrar diretório atual
print(f"Diretório atual: {os.getcwd()}")

# 1. SIMULAÇÃO DE DADOS HISTÓRICOS (60 dias)
datas = pd.date_range(start='2024-01-01', periods=dias)

# Variáveis explicativas (TODAS com o mesmo tamanho: dias=60)
temperatura = np.random.normal(28, 3, dias)
fim_de_semana = np.array([1 if data.weekday() >= 5 else 0 for data in datas])
feriados = np.random.choice([0, 1], size=dias, p=[0.9, 0.1])
ruido = np.random.normal(0, 5, dias)  # Ruído com tamanho correto

# Geração da demanda
demanda = 50 + 2.5 * temperatura + 15 * fim_de_semana + 10 * feriados + ruido

# Criando DataFrame
dados = pd.DataFrame({
    'Data': datas,
    'Demanda': demanda,
    'Temperatura': temperatura,
    'Fim_de_Semana': fim_de_semana,
    'Feriado': feriados
})

print("=== DADOS SIMULADOS ===")
print(dados.head())
print("\nEstatísticas descritivas:")
print(dados.describe())

# 2. ANÁLISE EXPLORATÓRIA
print("\n=== ANÁLISE EXPLORATÓRIA ===")

# Correlações
correlacao = dados[['Demanda', 'Temperatura', 'Fim_de_Semana', 'Feriado']].corr()
print("\nMatriz de correlação:")
print(correlacao)

# Gráficos de análise exploratória
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Demanda ao longo do tempo
axes[0, 0].plot(dados['Data'], dados['Demanda'])
axes[0, 0].set_title('Demanda Diária de Açaí')
axes[0, 0].set_ylabel('Quantidade Vendida')

# Distribuição da demanda
axes[0, 1].hist(dados['Demanda'], bins=15, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Distribuição da Demanda')
axes[0, 1].set_xlabel('Quantidade')
axes[0, 1].set_ylabel('Frequência')

# Demanda vs Temperatura
axes[1, 0].scatter(dados['Temperatura'], dados['Demanda'], alpha=0.7)
axes[1, 0].set_title('Relação: Demanda vs Temperatura')
axes[1, 0].set_xlabel('Temperatura (°C)')
axes[1, 0].set_ylabel('Demanda')

# Demanda por tipo de dia
fim_semana_demanda = dados[dados['Fim_de_Semana'] == 1]['Demanda']
dia_util_demanda = dados[dados['Fim_de_Semana'] == 0]['Demanda']
axes[1, 1].boxplot([dia_util_demanda, fim_semana_demanda], 
                  labels=['Dia Útil', 'Fim de Semana'])
axes[1, 1].set_title('Demanda por Tipo de Dia')
axes[1, 1].set_ylabel('Demanda')

plt.tight_layout()
caminho_grafico = os.path.join(os.getcwd(), 'analise_exploratoria.png')
plt.savefig(caminho_grafico, dpi=300)
plt.close()
print(f"Gráfico salvo em: {caminho_grafico}")

# 3. TESTE DE HIPÓTESE: Diferença de demanda entre dias úteis e fins de semana
print("\n=== TESTE DE HIPÓTESE ===")
t_stat, p_value = stats.ttest_ind(dia_util_demanda, fim_semana_demanda)
print(f"Teste t para diferença de médias:")
print(f"Média dias úteis: {dia_util_demanda.mean():.2f}")
print(f"Média fins de semana: {fim_semana_demanda.mean():.2f}")
print(f"Estatística t: {t_stat:.4f}, Valor p: {p_value:.4f}")

if p_value < 0.05:
    print("Há diferença significativa na demanda entre dias úteis e fins de semana (p < 0.05)")
else:
    print("Não há diferença significativa na demanda entre dias úteis e fins de semana")

# 4. MODELO DE REGRESSÃO LINEAR
print("\n=== MODELO DE REGRESSÃO LINEAR ===")

# Preparando os dados
X = dados[['Temperatura', 'Fim_de_Semana', 'Feriado']]
y = dados['Demanda']

# Adicionando constante para o intercepto
X = sm.add_constant(X)

# Ajustando o modelo
modelo = sm.OLS(y, X).fit()

# Resultados do modelo
print(modelo.summary())

# Previsões
dados['Previsao'] = modelo.predict(X)

# Métricas de avaliação
r2 = r2_score(y, dados['Previsao'])
rmse = np.sqrt(mean_squared_error(y, dados['Previsao']))

print(f"\nMétricas de avaliação:")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# 5. SIMULAÇÃO DE PREVISÃO PARA OS PRÓXIMOS 7 DIAS
print("\n=== PREVISÃO PARA PRÓXIMOS 7 DIAS ===")

# Criando dados futuros
proximos_7_dias = pd.date_range(start=datas[-1] + pd.Timedelta(days=1), periods=7)
temp_futura = np.random.normal(28, 3, 7)
fds_futuro = [1 if data.weekday() >= 5 else 0 for data in proximos_7_dias]
feriados_futuro = [0] * 7  # Supondo nenhum feriado

# DataFrame para previsão
futuro = pd.DataFrame({
    'const': 1,
    'Temperatura': temp_futura,
    'Fim_de_Semana': fds_futuro,
    'Feriado': feriados_futuro
})

# Fazendo previsões
previsoes = modelo.predict(futuro)

resultado_previsao = pd.DataFrame({
    'Data': proximos_7_dias,
    'Temperatura_Prevista': temp_futura,
    'Fim_de_Semana': fds_futuro,
    'Demanda_Prevista': previsoes
})

print(resultado_previsao.round(2))

# 6. RECOMENDAÇÕES PARA A AÇAÍTERIA
print("\n=== RECOMENDAÇÕES ===")
media_demanda = dados['Demanda'].mean()
desvio_padrao = dados['Demanda'].std()

print(f"Demanda média histórica: {media_demanda:.2f} unidades/dia")
print(f"Desvio padrão: {desvio_padrao:.2f} unidades")

# Calculando estoque ideal considerando 95% de confiança
estoque_ideal = media_demanda + 1.645 * desvio_padrao  # Percentil 95 da distribuição normal
print(f"Estoque ideal para atender 95% da demanda: {estoque_ideal:.2f} unidades")

# Economia estimada com redução de desperdício
desperdicio_atual = 0.2 * media_demanda  # Supondo 20% de desperdício atual
desperdicio_previsto = 0.05 * media_demanda  # Meta de 5% de desperdício
economia_diaria = desperdicio_atual - desperdicio_previsto
economia_mensal = economia_diaria * 30

print(f"\nEconomia estimada com redução de desperdício:")
print(f"Redução de {desperdicio_atual:.2f} para {desperdicio_previsto:.2f} unidades/dia")
print(f"Economia mensal estimada: {economia_mensal:.2f} unidades")

# 7. SALVANDO RESULTADOS
try:
    caminho_dados = os.path.join(os.getcwd(), 'dados_acai_analisados.csv')
    caminho_previsao = os.path.join(os.getcwd(), 'previsao_proximos_dias.csv')
    
    dados.to_csv(caminho_dados, index=False)
    print(f"Arquivo salvo: {caminho_dados}")
    
    resultado_previsao.to_csv(caminho_previsao, index=False)
    print(f"Arquivo salvo: {caminho_previsao}")
    
    # Verificar se os arquivos foram criados
    for caminho in [caminho_dados, caminho_previsao, caminho_grafico]:
        if os.path.exists(caminho):
            print(f"✓ {caminho} foi criado com sucesso!")
        else:
            print(f"✗ {caminho} não foi criado!")
            
except Exception as e:
    print(f"Erro ao salvar arquivos: {e}")

print("\n=== ANÁLISE CONCLUÍDA ===")