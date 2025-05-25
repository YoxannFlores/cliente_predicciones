from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import os
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

app = Flask(__name__)

# ------------------- CARGA DE DATOS -------------------
archivo = pd.ExcelFile("Base_clientes_dividos.xlsx", engine='openpyxl')
pf_con_ae = pd.read_excel(archivo, sheet_name="PFCAE")
pf_sin_ae = pd.read_excel(archivo, sheet_name="Base_clientes_dividos")
clientes = pd.concat([pf_con_ae, pf_sin_ae], ignore_index=True)

transacciones = pd.read_excel("base_transacciones_final(in)(version 1).xlsx", engine='openpyxl')
transacciones['fecha'] = pd.to_datetime(transacciones['fecha'])

# ------------------- FUNCIONES -------------------

def buscar_por_id(cliente_id):
    datos_cliente = clientes[clientes['id'] == cliente_id][[
        'id', 'fecha_nacimiento', 'fecha_alta', 'id_municipio', 'tipo_persona']]
    transacciones_cliente = transacciones[transacciones['id'] == cliente_id][[
        'fecha', 'comercio', 'giro_comercio', 'tipo_venta', 'monto']]
    return datos_cliente, transacciones_cliente

def filtrar_transacciones(cliente_id):
    df_cliente = transacciones[transacciones['id'] == cliente_id].copy()
    df_cliente['fecha'] = pd.to_datetime(df_cliente['fecha'])
    df_cliente['dia_pago'] = df_cliente['fecha'].dt.day
    df_cliente['mes'] = df_cliente['fecha'].dt.to_period('M')
    df_cliente['mes_completo'] = df_cliente['fecha'].dt.strftime('%B %Y')
    return df_cliente

def clasificar_gastos(df, tolerancia_dia=2, tolerancia_monto=1.0, min_frecuencia=3):
    clasificacion = []
    agrupado = df.groupby('comercio')
    for comercio, grupo in agrupado:
        if len(grupo) < 2:
            tipo = 'Anormal'
        else:
            dia_mas_comun = grupo['dia_pago'].mode().iloc[0]
            monto_mas_comun = grupo['monto'].mode().iloc[0]
            dias_dentro_rango = (abs(grupo['dia_pago'] - dia_mas_comun) <= tolerancia_dia).sum()
            montos_dentro_rango = (abs(grupo['monto'] - monto_mas_comun) <= tolerancia_monto).sum()
            if dias_dentro_rango >= min_frecuencia and montos_dentro_rango >= min_frecuencia:
                tipo = 'Gasto fijo'
            elif dias_dentro_rango >= min_frecuencia:
                tipo = 'Gasto frecuente'
            elif montos_dentro_rango >= min_frecuencia:
                tipo = 'Poco frecuente'
            else:
                tipo = 'Anormal'
        clasificacion.append({
            'comercio': comercio,
            'tipo': tipo,
            'veces': len(grupo),
            'monto_promedio': grupo['monto'].mean()
        })
    return pd.DataFrame(clasificacion)

def clasificar_gastos_con_etiqueta(df, tolerancia_dia=2, tolerancia_monto=1.0, min_frecuencia=3):
    etiquetas = []
    agrupado = df.groupby('comercio')
    for comercio, grupo in agrupado:
        tipo = 'Anormal'
        if len(grupo) >= 2:
            dia_mas_comun = grupo['dia_pago'].mode().iloc[0]
            monto_mas_comun = grupo['monto'].mode().iloc[0]
            dias_dentro_rango = (abs(grupo['dia_pago'] - dia_mas_comun) <= tolerancia_dia)
            montos_dentro_rango = (abs(grupo['monto'] - monto_mas_comun) <= tolerancia_monto)
            if dias_dentro_rango.sum() >= min_frecuencia and montos_dentro_rango.sum() >= min_frecuencia:
                tipo = 'Gasto fijo'
            elif dias_dentro_rango.sum() >= min_frecuencia:
                tipo = 'Gasto frecuente'
            elif montos_dentro_rango.sum() >= min_frecuencia:
                tipo = 'Poco frecuente'
        etiquetas += [{'index': i, 'tipo': tipo} for i in grupo.index]
    etiquetas_df = pd.DataFrame(etiquetas).set_index('index')
    return df.join(etiquetas_df)

def generar_graficas(cliente_id):
    df_cliente = filtrar_transacciones(cliente_id)

    # Gráfica de líneas
    gasto_tiempo = df_cliente.groupby(df_cliente['fecha'].dt.to_period('M'))['monto'].sum()
    gasto_tiempo.index = gasto_tiempo.index.to_timestamp()
    plt.figure(figsize=(6, 4))
    gasto_tiempo.plot(marker='o')
    plt.title('Histórico de Gastos Mensuales')
    plt.xlabel('Fecha')
    plt.ylabel('Gasto (MXN)')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('static/grafica_lineal.png')
    plt.close()

    # Gráfico de pastel
    top = df_cliente['comercio'].value_counts().nlargest(5)
    otros = df_cliente['comercio'].value_counts().iloc[5:].sum()
    top['Otros'] = otros
    plt.figure(figsize=(5, 5))
    top.plot.pie(autopct='%1.1f%%', startangle=90)
    plt.ylabel('')
    plt.title('Comercios más Frecuentados')
    plt.tight_layout()
    plt.savefig('static/grafica_pastel.png')
    plt.close()

def correr_modelo(cliente_id):
    df_cliente = filtrar_transacciones(cliente_id)
    transacciones_por_mes = df_cliente.groupby(df_cliente['fecha'].dt.to_period('M')).size()
    promedio_transacciones = transacciones_por_mes.mean()

    if promedio_transacciones <= 5:
        freq = 'M'
    elif promedio_transacciones <= 15:
        freq = '15D'
    else:
        freq = 'W'

    df_prophet = df_cliente[['fecha', 'monto']].copy()
    df_prophet = df_prophet.resample(freq, on='fecha').sum().reset_index()
    df_prophet.rename(columns={'fecha': 'ds', 'monto': 'y'}, inplace=True)
    df_prophet = df_prophet[df_prophet['y'] > 0]

    if len(df_prophet) < 6 or df_prophet['y'].std() < 14:
        return {"error": "No se puede predecir por poca actividad o variabilidad"}

    modelo = Prophet()
    modelo.fit(df_prophet)
    future = modelo.make_future_dataframe(periods=1, freq=freq)
    forecast = modelo.predict(future)

    ultimo_real = df_prophet['y'].iloc[-1]
    siguiente_predicho = max(0, forecast['yhat'].iloc[-1])
    variacion = ((siguiente_predicho - ultimo_real) / ultimo_real) * 100

    df_cv = cross_validation(modelo, initial='180 days', period='30 days', horizon='30 days')
    df_perf = performance_metrics(df_cv)

    return {
        "prediccion": round(siguiente_predicho, 2),
        "variacion": round(variacion, 2),
        "coverage": round(df_perf['coverage'].mean() * 100, 2),
        "ultimo": round(ultimo_real, 2)
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        cliente_id = request.form['cliente_id']
        resultado = correr_modelo(cliente_id)
        generar_graficas(cliente_id)
        return render_template('resultado.html', **resultado, cliente_id=cliente_id)
    return render_template('index_2.html')

if __name__ == '__main__':
    app.run(debug=True)

