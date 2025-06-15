from flask import Flask, jsonify, request, send_from_directory
import os
from flask_cors import CORS
import math
from scipy.stats import norm, poisson, kstest, chisquare
import numpy as np 

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')
CORS(app) # Permitir CORS para todas las rutas
# Configurar la carpeta estática para servir archivos de React
# Ruta para servir los archivos estáticos de React (después de la construcción)
@app.route('/')
def serve_react_app():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/hello')
def hello():
    return jsonify(message="Hola desde el Backend de Flask!")

@app.route('/api/generate_distribution_data', methods=['POST'])
def generate_distribution_data():
    data = request.get_json()
    distribution_type = data.get('distributionType')
    
    labels = []
    probabilities = []

    if distribution_type == 'poisson':
        try:
            lambda_val = float(data.get('lambda'))
            # Generar k desde 0 hasta un valor razonable (ej. lambda * 3 o 15), ajustado para asegurar visibilidad de la cola
            max_k = max(15, math.ceil(lambda_val * 3) + 2) 
            for k in range(0, max_k + 1):
                prob = poisson.pmf(k, lambda_val) # Usamos scipy.stats.poisson.pmf
                labels.append(str(k))
                probabilities.append(prob)
        except (ValueError, TypeError):
            return jsonify({"error": "Parámetros de Poisson inválidos"}), 400

    elif distribution_type == 'normal':
        try:
            mean_val = float(data.get('mean'))
            std_dev_val = float(data.get('stdDev'))
            
            min_x = mean_val - 4 * std_dev_val
            max_x = mean_val + 4 * std_dev_val
            num_points = 100 
            
            for i in range(num_points + 1):
                x = min_x + (max_x - min_x) * i / num_points
                prob = norm.pdf(x, loc=mean_val, scale=std_dev_val)
                labels.append(f"{x:.2f}")
                probabilities.append(prob)
        except (ValueError, TypeError):
            return jsonify({"error": "Parámetros de Normal inválidos"}), 400

    else:
        return jsonify({"error": "Tipo de distribución no soportado"}), 400

    return jsonify({
        "labels": labels,
        "data": probabilities
    })

# --- RUTAS API PARA PRUEBAS DE BONDAD DE AJUSTE ---

@app.route('/api/run_goodness_of_fit_test', methods=['POST'])
def run_goodness_of_fit_test():
    data = request.get_json()
    test_type = data.get('testType')
    observed_data = data.get('observedData') # [345, 310, ..., 0]
    distribution_type = data.get('distributionType') # 'poisson' o 'normal'
    
    # Parámetros de la distribución teórica
    lambda_val = data.get('lambda')
    mean_val = data.get('mean')
    std_dev_val = data.get('stdDev')

    results = {
        "testType": test_type,
        "distributionType": distribution_type,
        "statistic": None,
        "pValue": None,
        "conclusion": "No se pudo realizar la prueba.",
        "details": {}
    }

    # Suma total de los abandonos observados (para escalar las probabilidades teóricas)
    total_observed_count = sum(observed_data)
    if total_observed_count == 0:
        results["conclusion"] = "No hay abandonos observados para analizar (suma total es 0)."
        return jsonify(results)

    # Convertir a numpy array para facilitar operaciones
    observed_counts_np = np.array(observed_data)
    
    # --- Generación de frecuencias esperadas y agrupación para Chi-cuadrado ---
    
    expected_probabilities_for_bins = [] # Probabilidad teórica para cada "bin" (semestre)
    
    if distribution_type == 'poisson':
        if lambda_val is None:
            results["conclusion"] = "Lambda para Poisson no proporcionado."
            return jsonify(results)
        lambda_val = float(lambda_val)
        
        # Asumimos que los "semestres" 1 a 10 corresponden a los conteos k=1 a k=10
        # El último bin (semestre 10) debe incluir la probabilidad de X >= 10
        
        for k_val in range(1, len(observed_data)): # Bins para k=1 a k=9
            expected_probabilities_for_bins.append(poisson.pmf(k_val, lambda_val))
        
        # Último bin: P(X >= len(observed_data)) (ej. P(X >= 10) para el 10mo semestre)
        # 1 - poisson.cdf(k-1, lambda) es P(X >= k)
        expected_probabilities_for_bins.append(1 - poisson.cdf(len(observed_data) - 1, lambda_val))
        
    elif distribution_type == 'normal':
        if mean_val is None or std_dev_val is None:
            results["conclusion"] = "Media o Desviación Estándar para Normal no proporcionados."
            return jsonify(results)
        mean_val = float(mean_val)
        std_dev_val = float(std_dev_val)

        # Calculamos la probabilidad de cada bin (semestre) bajo la curva Normal
        # Asumimos que los semestres son rangos (ej. semestre 1 = 0.5 a 1.5, semestre 2 = 1.5 a 2.5, etc.)
        # Y el último bin es 9.5 a infinito
        
        # Probabilidad acumulada hasta el límite inferior del primer semestre
        # P(X < 0.5) o P(X <= 0) si consideramos el primer semestre como el inicio.
        # Para ser consistentes con la discretización de semestres
        prev_cdf = norm.cdf(0.5, loc=mean_val, scale=std_dev_val) # P(X <= 0.5)
        
        for i in range(1, len(observed_data)): # Semestres 1 a 9
            current_upper_bound = i + 0.5 # Límite superior del bin (ej. para sem 1 es 1.5)
            current_cdf = norm.cdf(current_upper_bound, loc=mean_val, scale=std_dev_val)
            expected_probabilities_for_bins.append(current_cdf - prev_cdf)
            prev_cdf = current_cdf
        
        # Última categoría (10mo semestre y más allá)
        expected_probabilities_for_bins.append(1 - prev_cdf) # P(X > 9.5)
        
    else:
        results["conclusion"] = "Tipo de distribución no soportado para la prueba."
        return jsonify(results)

    # --- NORMALIZACIÓN CRUCIAL PARA CHI-CUADRADO ---
    # Asegurarse de que la suma de las probabilidades esperadas sea 1.0 (o muy cercana)
    # y luego escalarlas al total de observaciones.
    sum_expected_probs = np.sum(expected_probabilities_for_bins)
    if sum_expected_probs <= 0: # Si la suma es cero o negativa, hay un problema con los parámetros.
        results["conclusion"] = "Suma de probabilidades esperadas es cero o negativa. Parámetros de distribución podrían ser inadecuados."
        return jsonify(results)

    # Normalizar las probabilidades para que sumen 1, y luego escalar por el total de observados.
    normalized_expected_probs = np.array(expected_probabilities_for_bins) / sum_expected_probs
    expected_counts_raw = normalized_expected_probs * total_observed_count
    
    # --- Agrupación de categorías para Chi-cuadrado ---
    # Esta función agrupará los datos para asegurar que las frecuencias esperadas >= 5
    # (o un mínimo configurable, 5 es el estándar de Cochran)
    def group_categories(obs_counts, exp_counts, min_expected=5):
        grouped_obs = []
        grouped_exp = []
        current_obs_group = 0
        current_exp_group = 0
        
        # Asegurarse de que exp_counts no contenga NaNs o Infs antes de agrupar
        exp_counts = np.nan_to_num(exp_counts, nan=0.0, posinf=0.0, neginf=0.0)

        for i in range(len(obs_counts)):
            current_obs_group += obs_counts[i]
            current_exp_group += exp_counts[i]
            
            # Si es la última categoría O el grupo acumulado ya es >= min_expected
            # y no estamos en la última categoría (para no crear un grupo solo con la última)
            # O estamos en la última categoría y necesitamos cerrar el grupo
            if (current_exp_group >= min_expected and i < len(obs_counts) - 1) or \
               (i == len(obs_counts) - 1):
                
                # Si el último grupo acumulado aún es < min_expected, intentamos fusionarlo con el anterior
                # Esto es una lógica más compleja para evitar grupos muy pequeños al final.
                # Simplificaremos: si al final un grupo es pequeño, lo combinamos con el penúltimo.
                
                # Añadir el grupo actual si es válido
                if current_exp_group > 0: # Solo añadir si hay algo en el grupo
                    grouped_obs.append(current_obs_group)
                    grouped_exp.append(current_exp_group)
                current_obs_group = 0
                current_exp_group = 0
            
            # Caso especial: Si es la última categoría y el current_exp_group es menor que min_expected
            # y ya hay grupos, intenta fusionar con el último grupo existente.
            # Esto puede llevar a un bucle infinito si hay solo un grupo y es menor que min_expected.
            # Una estrategia más simple y robusta es agrupar solo hacia adelante o hacia atrás.
            # La implementación actual agrupa hacia adelante.

        # Post-processing para asegurar que los últimos grupos cumplan min_expected
        # Si el último grupo es menor que min_expected, se fusiona con el penúltimo.
        # Esto podría ser un problema si solo hay un grupo pequeño.
        # En ese caso, la prueba Chi-cuadrado no es apropiada.
        if len(grouped_exp) > 1 and grouped_exp[-1] < min_expected:
            grouped_exp[-2] += grouped_exp[-1]
            grouped_obs[-2] += grouped_obs[-1]
            grouped_exp.pop()
            grouped_obs.pop()
        elif len(grouped_exp) == 1 and grouped_exp[0] < min_expected:
            # Si solo hay un grupo y es muy pequeño, la prueba Chi-cuadrado no es fiable.
            # Podríamos optar por no realizar la prueba o avisar.
            # Para este caso, dejaremos que chisquare falle si no hay suficientes datos,
            # pero el frontend recibirá el error.
            pass # No hacer nada, dejar que el flujo continúe

        return np.array(grouped_obs), np.array(grouped_exp)


    # Aplicar la agrupación
    # Aquí es donde `expected_counts_raw` (ya escalado al total de observados) se usa.
    grouped_observed_np, grouped_expected_np = group_categories(observed_counts_np, expected_counts_raw)
    
    # Verificar que no haya bins vacíos o con valores esperados muy bajos después de la agrupación
    # También asegurarse de que las sumas sigan siendo consistentes después de la agrupación.
    # SciPy chisquare tiene sus propias validaciones para sumas, pero es bueno ser proactivo.
    
    # Si después de agrupar, la suma de esperados difiere mucho de la suma de observados,
    # algo salió mal en la agrupación o en los datos originales.
    if not np.isclose(np.sum(grouped_observed_np), np.sum(grouped_expected_np)):
        results["conclusion"] = f"Error interno: La suma de frecuencias observadas ({np.sum(grouped_observed_np)}) no coincide con la suma de esperadas ({np.sum(grouped_expected_np)}) después de la agrupación. Diferencia porcentual: {abs(np.sum(grouped_observed_np) - np.sum(grouped_expected_np)) / np.sum(grouped_observed_np) * 100:.2f}%"
        return jsonify(results)

    # Asegurar que no haya ceros en las frecuencias esperadas después de agrupar para evitar NaNs
    # Esto ya lo hace group_categories con nan_to_num, pero es una doble verificación.
    if np.any(grouped_expected_np <= 0):
        # Reemplazar 0s con un valor muy pequeño (ej. 1e-10) para evitar división por cero en chisquare si esto ocurre
        # Esto es un workaround, la agrupación debería evitarlo idealmente.
        grouped_expected_np = np.maximum(grouped_expected_np, 1e-10)


    # --- Ejecutar la Prueba Seleccionada ---
    if test_type == 'chi_square':
        try:
            # chisquare directamente compara observados y esperados
            # Si el error "sum of observed must agree with sum of expected" persiste,
            # forzar la normalización de chisquare (correct=False) puede ayudar, pero no es la solución ideal.
            # La solución ideal es que tus datos estén correctos antes.
            stat, p_value = chisquare(f_obs=grouped_observed_np, f_exp=grouped_expected_np)
            
            # Los grados de libertad son (número de bins agrupados - 1)
            # Si se estimó un parámetro de la distribución (lambda, media, std_dev) a partir de los datos,
            # se resta un grado de libertad adicional por cada parámetro estimado.
            # Aquí, los parámetros son dados por el usuario, no estimados de 'observed_data',
            # por lo que no se restan grados de libertad adicionales por los parámetros.
            df = len(grouped_observed_np) - 1 
            
            # Asegurar que el estadístico y p-valor no sean NaN si hay problemas con los datos
            if np.isnan(stat) or np.isnan(p_value):
                 results["conclusion"] = "Error de cálculo: Estadístico o P-valor Chi-cuadrado resultó en NaN. Datos inadecuados para la prueba."
                 return jsonify(results)

            results["statistic"] = round(stat, 4)
            results["pValue"] = round(p_value, 4)
            results["details"]["degrees_of_freedom"] = df
            results["details"]["grouped_observed_counts"] = grouped_observed_np.tolist()
            results["details"]["grouped_expected_counts"] = grouped_expected_np.tolist()

            alpha = 0.05
            if p_value < alpha:
                results["conclusion"] = f"Se rechaza la hipótesis nula (H0). Los datos observados NO se ajustan a una distribución {distribution_type.capitalize()} con los parámetros dados (p-valor = {p_value:.4f} < {alpha})."
            else:
                results["conclusion"] = f"No se rechaza la hipótesis nula (H0). Los datos observados PUEDEN ajustarse a una distribución {distribution_type.capitalize()} con los parámetros dados (p-valor = {p_value:.4f} >= {alpha})."
        except Exception as e:
            import traceback
            results["conclusion"] = f"Error al ejecutar la prueba Chi-cuadrado: {str(e)}"
            results["details"]["error_message"] = str(e)
            results["details"]["traceback"] = traceback.format_exc() # Para depuración


    elif test_type == 'kolmogorov_smirnov':
        try:
            # Generar muestra_sintetica para K-S
            synthetic_sample = []
            for i, count in enumerate(observed_data): 
                if count > 0: 
                    synthetic_sample.extend([i + 1] * count) 
            
            if not synthetic_sample: 
                results["conclusion"] = "No hay datos en la muestra observada para la prueba K-S."
                return jsonify(results)

            synthetic_sample_np = np.array(synthetic_sample)

            if distribution_type == 'poisson':
                stat, p_value = kstest(synthetic_sample_np, lambda x: poisson.cdf(x, lambda_val))
            elif distribution_type == 'normal':
                stat, p_value = kstest(synthetic_sample_np, 'norm', args=(mean_val, std_dev_val))
            
            if np.isnan(stat) or np.isnan(p_value):
                 results["conclusion"] = "Error de cálculo: Estadístico o P-valor K-S resultó en NaN. Datos inadecuados para la prueba."
                 return jsonify(results)

            results["statistic"] = round(stat, 4)
            results["pValue"] = round(p_value, 4)
            results["details"]["sample_size_ks"] = len(synthetic_sample_np)
            
            alpha = 0.05
            if p_value < alpha:
                results["conclusion"] = f"Se rechaza la hipótesis nula (H0). Los datos observados NO se ajustan a una distribución {distribution_type.capitalize()} con los parámetros dados (p-valor = {p_value:.4f} < {alpha})."
            else:
                results["conclusion"] = f"No se rechaza la hipótesis nula (H0). Los datos observados PUEDEN ajustarse a una distribución {distribution_type.capitalize()} con los parámetros dados (p-valor = {p_value:.4f} >= {alpha})."
        except Exception as e:
            import traceback
            results["conclusion"] = f"Error al ejecutar la prueba Kolmogorov-Smirnov: {str(e)}"
            results["details"]["error_message"] = str(e)
            results["details"]["traceback"] = traceback.format_exc() 
        
    else:
        results["conclusion"] = "Tipo de prueba no soportado o error en parámetros."
        return jsonify(results)

    return jsonify(results)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host='0.0.0.0', port=port)