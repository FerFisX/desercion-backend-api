from flask import Flask, jsonify, request
import os
from flask_cors import CORS
import math
from scipy.stats import norm, poisson, kstest, chisquare
import numpy as np 
import logging
import sys

# Configurar el logging para que los errores se vean en Render
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

# --- Configuración de CORS ---
# Obtener la URL del frontend de las variables de entorno de Render
# Esto es CRUCIAL para que el frontend pueda comunicarse con el backend.
# Si no se define FRONTEND_URL en Render, usará localhost para desarrollo.
frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:5173") 

# Lista de orígenes permitidos
allowed_origins = [frontend_url]
# Añadir localhost para desarrollo (Vite usa 5173, Create React App usa 3000)
allowed_origins.extend(["http://localhost:5173", "http://localhost:3000"])

# Configurar CORS para permitir solicitudes desde el frontend solo a las rutas API
# Esto es más seguro que CORS(app) sin restricciones.
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

# Eliminar la ruta para servir la app de React desde Flask.
# Render sirve el frontend por separado como un Static Site.
# @app.route('/')
# def serve_react_app():
#    return send_from_directory(app.static_folder, 'index.html')

@app.route('/')
def hello_world_backend():
    """Endpoint simple para verificar que el backend está vivo."""
    app.logger.info("Solicitud recibida en el endpoint /")
    return jsonify(message="¡Backend de Deserción funcionando!")

@app.route('/api/generate_distribution_data', methods=['POST'])
def generate_distribution_data():
    """Genera datos para visualizar distribuciones teóricas."""
    try:
        data = request.get_json()
        if not data:
            app.logger.warning("generate_distribution_data: No JSON data provided.")
            return jsonify({"error": "No JSON data provided"}), 400

        distribution_type = data.get('distributionType')
        
        labels = []
        probabilities = []

        if distribution_type == 'poisson':
            try:
                lambda_val = float(data.get('lambda'))
                if lambda_val <= 0:
                    return jsonify({"error": "Lambda debe ser un valor positivo para Poisson"}), 400
                
                # Generar k desde 0 hasta un valor razonable (ej. lambda * 3 o 15), ajustado para asegurar visibilidad de la cola
                max_k = max(15, math.ceil(lambda_val * 3) + 2) 
                for k in range(0, max_k + 1):
                    prob = poisson.pmf(k, lambda_val)
                    labels.append(str(k))
                    probabilities.append(prob)
            except (ValueError, TypeError) as e:
                app.logger.error(f"generate_distribution_data (Poisson): Error de parámetro: {e}")
                return jsonify({"error": f"Parámetros de Poisson inválidos: {e}"}), 400

        elif distribution_type == 'normal':
            try:
                mean_val = float(data.get('mean'))
                std_dev_val = float(data.get('stdDev'))
                if std_dev_val <= 0:
                    return jsonify({"error": "Desviación estándar debe ser positiva para Normal"}), 400
                
                min_x = mean_val - 4 * std_dev_val
                max_x = mean_val + 4 * std_dev_val
                num_points = 100 
                
                for i in range(num_points + 1):
                    x = min_x + (max_x - min_x) * i / num_points
                    prob = norm.pdf(x, loc=mean_val, scale=std_dev_val)
                    labels.append(f"{x:.2f}")
                    probabilities.append(prob)
            except (ValueError, TypeError) as e:
                app.logger.error(f"generate_distribution_data (Normal): Error de parámetro: {e}")
                return jsonify({"error": f"Parámetros de Normal inválidos: {e}"}), 400

        else:
            app.logger.warning(f"generate_distribution_data: Tipo de distribución no soportado: {distribution_type}")
            return jsonify({"error": "Tipo de distribución no soportado"}), 400

        return jsonify({
            "labels": labels,
            "data": probabilities
        })
    except Exception as e:
        app.logger.error(f"Error general en generate_distribution_data: {e}", exc_info=True)
        return jsonify({"error": f"Error interno del servidor al generar distribución: {e}"}), 500

# --- RUTAS API PARA PRUEBAS DE BONDAD DE AJUSTE ---

@app.route('/api/run_goodness_of_fit_test', methods=['POST'])
def run_goodness_of_fit_test():
    """Realiza pruebas de bondad de ajuste (Chi-cuadrado, Kolmogorov-Smirnov)."""
    try:
        data = request.get_json()
        if not data:
            app.logger.warning("run_goodness_of_fit_test: No JSON data provided.")
            return jsonify({"error": "No JSON data provided"}), 400

        test_type = data.get('testType')
        observed_data = data.get('observedData')
        distribution_type = data.get('distributionType')
        
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

        if not isinstance(observed_data, list) or not all(isinstance(x, (int, float)) for x in observed_data):
            results["conclusion"] = "Los datos observados no son una lista válida de números."
            return jsonify(results), 400

        total_observed_count = sum(observed_data)
        if total_observed_count == 0:
            results["conclusion"] = "No hay abandonos observados para analizar (suma total es 0)."
            return jsonify(results)

        observed_counts_np = np.array(observed_data)
        expected_probabilities_for_bins = []
        
        # --- Generación de frecuencias esperadas basadas en la distribución teórica ---
        if distribution_type == 'poisson':
            if lambda_val is None:
                results["conclusion"] = "Lambda para Poisson no proporcionado."
                return jsonify(results), 400
            try:
                lambda_val = float(lambda_val)
                if lambda_val <= 0:
                    return jsonify({"error": "Lambda debe ser un valor positivo para Poisson"}), 400

                # Para Poisson, los bins corresponden a k=0, k=1, k=2...
                # Asumimos que observed_data[0] es para k=0, observed_data[1] para k=1, etc.
                # El último bin agrupa "k o más".
                for k_val in range(len(observed_data) - 1): # Bins para k=0 a k=len-2
                    expected_probabilities_for_bins.append(poisson.pmf(k_val, lambda_val))
                
                # Último bin: P(X >= last_k_value)
                expected_probabilities_for_bins.append(1 - poisson.cdf(len(observed_data) - 2, lambda_val)) # Cuidado con el índice
                
                # Corrección para cuando observed_data es pequeño y cdf podría ser de un k-1 no existente
                # Si tienes datos para 10 semestres (índices 0-9), el último bin es k=9 o más.
                # P(X >= 9) = 1 - P(X <= 8) = 1 - poisson.cdf(8, lambda_val)
                # Si tu observed_data[i] es para el semestre 'i' (empezando en 1), ajusta los índices (i-1).
                # La lógica actual asume observed_data[i] corresponde a k=i.
                # Si `observed_data` es `[semestre_1, semestre_2, ...]` y representa conteos para k=1, k=2, etc.
                # La forma en que lo tienes definido en el frontend como `[345, 310, ...]` sugiere que el índice 0 es el primer bin.
                # Si el primer bin es para k=0, entonces la lógica de range(len(observed_data)-1) está bien.
                # Y el último sería P(X >= len(observed_data)-1) = 1 - poisson.cdf(len(observed_data)-2, lambda_val)
                # O simplificando, si tus bins son directos k=0, 1, 2...
                # Y el último es para "k_max y más allá".
                
                # REVISIÓN DE BINS: Si tu `observed_data` representa `[count_for_k0, count_for_k1, ..., count_for_k_n]`:
                # Entonces el último bin debe acumular el resto de la probabilidad.
                calculated_probs = [poisson.pmf(k, lambda_val) for k in range(len(observed_data) - 1)]
                calculated_probs.append(1 - poisson.cdf(len(observed_data) - 2, lambda_val))
                expected_probabilities_for_bins = calculated_probs

            except (ValueError, TypeError) as e:
                app.logger.error(f"run_goodness_of_fit_test (Poisson): Error de parámetro: {e}")
                results["conclusion"] = f"Parámetros de Poisson inválidos: {e}"
                return jsonify(results), 400
            
        elif distribution_type == 'normal':
            if mean_val is None or std_dev_val is None:
                results["conclusion"] = "Media o Desviación Estándar para Normal no proporcionados."
                return jsonify(results), 400
            try:
                mean_val = float(mean_val)
                std_dev_val = float(std_dev_val)
                if std_dev_val <= 0:
                    return jsonify({"error": "Desviación estándar debe ser positiva para Normal"}), 400

                # Para Normal, necesitamos probabilidades para intervalos (bins).
                # Asumimos que observed_data[i] corresponde al i-ésimo bin.
                # Necesitas definir los límites de tus bins.
                # Si tus observed_data representan semestres (1, 2, ..., N),
                # podrías definir bins como [0.5, 1.5), [1.5, 2.5), ..., [N-0.5, Inf)
                
                # Ejemplo de bins para semestres [1..10]:
                # bin 0: (0, 1], bin 1: (1, 2], ... bin 9: (9, Inf)
                # O si `observed_data` es el conteo para el valor `i`:
                # bin 0 es para valor 0, bin 1 para valor 1, etc.
                # Esto es crucial para que la Chi-cuadrado sea correcta.
                
                # Implementación con bins discretos, asumiendo que observed_data[i] es para valor i
                # y el último bin es para 'i_max o más'.
                calculated_probs = []
                for i in range(len(observed_data) - 1):
                    # Asumimos que cada índice 'i' representa el valor 'i'
                    # Calculamos la probabilidad puntual para este valor o un rango muy pequeño alrededor de él
                    # Una mejor práctica sería definir rangos si la normal es continua.
                    # Para una aproximación discreta para la Normal:
                    # P(X=i) ≈ P(i-0.5 < X <= i+0.5) = CDF(i+0.5) - CDF(i-0.5)
                    # Y el último bin sería P(X > i_max-0.5)
                    prob = norm.cdf(i + 0.5, loc=mean_val, scale=std_dev_val) - norm.cdf(i - 0.5, loc=mean_val, scale=std_dev_val)
                    calculated_probs.append(prob)
                
                # Último bin (acumula la cola derecha)
                calculated_probs.append(1 - norm.cdf(len(observed_data) - 1 - 0.5, loc=mean_val, scale=std_dev_val)) # X > (last_index - 0.5)
                expected_probabilities_for_bins = calculated_probs

            except (ValueError, TypeError) as e:
                app.logger.error(f"run_goodness_of_fit_test (Normal): Error de parámetro: {e}")
                results["conclusion"] = f"Parámetros de Normal inválidos: {e}"
                return jsonify(results), 400
        
        elif distribution_type == 'exponential':
            if lambda_val is None:
                results["conclusion"] = "Lambda para Exponencial no proporcionado."
                return jsonify(results), 400
            try:
                lambda_val = float(lambda_val)
                if lambda_val <= 0:
                    return jsonify({"error": "Lambda debe ser un valor positivo para Exponencial"}), 400

                # Similar a la Normal, la Exponencial es continua. Necesitas bins.
                # Asumimos que observed_data[i] representa el i-ésimo bin de tiempo.
                # Por ejemplo, el bin i podría ser el tiempo entre i-1 y i.
                calculated_probs = []
                for i in range(len(observed_data) - 1):
                    # P(i-1 < X <= i) = CDF(i) - CDF(i-1)
                    prob = expon.cdf(i + 1, scale=1/lambda_val) - expon.cdf(i, scale=1/lambda_val)
                    calculated_probs.append(prob)
                
                # Último bin (acumula la cola derecha)
                calculated_probs.append(1 - expon.cdf(len(observed_data) - 1, scale=1/lambda_val))
                expected_probabilities_for_bins = calculated_probs

            except (ValueError, TypeError) as e:
                app.logger.error(f"run_goodness_of_fit_test (Exponential): Error de parámetro: {e}")
                results["conclusion"] = f"Parámetros de Exponencial inválidos: {e}"
                return jsonify(results), 400

        else:
            app.logger.warning(f"run_goodness_of_fit_test: Tipo de distribución no soportado para la prueba: {distribution_type}")
            results["conclusion"] = "Tipo de distribución no soportado para la prueba."
            return jsonify(results), 400

        # --- Normalización y Escalamiento de Frecuencias Esperadas ---
        sum_expected_probs = np.sum(expected_probabilities_for_bins)
        if sum_expected_probs <= 0 or np.isnan(sum_expected_probs) or np.isinf(sum_expected_probs):
            app.logger.error(f"Suma de probabilidades esperadas es inválida: {sum_expected_probs}")
            results["conclusion"] = "Suma de probabilidades esperadas es cero, negativa o inválida. Parámetros de distribución o cálculo de bins podrían ser inadecuados."
            return jsonify(results), 400

        normalized_expected_probs = np.array(expected_probabilities_for_bins) / sum_expected_probs
        expected_counts_raw = normalized_expected_probs * total_observed_count
        
        # --- Función de agrupación de categorías (Criterio de Cochran) ---
        def group_categories(obs_counts, exp_counts, min_expected=5):
            grouped_obs = []
            grouped_exp = []
            current_obs_group = 0
            current_exp_group = 0
            
            # Asegurarse de que exp_counts no contenga NaNs o Infs
            exp_counts = np.nan_to_num(exp_counts, nan=0.0, posinf=0.0, neginf=0.0)

            for i in range(len(obs_counts)):
                current_obs_group += obs_counts[i]
                current_exp_group += exp_counts[i]
                
                # Si el grupo acumulado ya cumple el mínimo o si es la última categoría
                # Y si el grupo no está vacío (para evitar grupos con 0 sumas, aunque nan_to_num ayuda)
                if (current_exp_group >= min_expected and i < len(obs_counts) - 1) or \
                   (i == len(obs_counts) - 1 and current_exp_group > 0): # Asegura que el último grupo no sea 0
                    grouped_obs.append(current_obs_group)
                    grouped_exp.append(current_exp_group)
                    current_obs_group = 0
                    current_exp_group = 0
            
            # Post-procesamiento para el caso donde el ÚLTIMO grupo quedó por debajo del mínimo
            # Esto es para evitar un solo bin final con un valor muy bajo que chisquare rechazaría.
            # Solo fusiona si hay al menos dos grupos para fusionar.
            if len(grouped_exp) > 1 and grouped_exp[-1] < min_expected:
                grouped_exp[-2] += grouped_exp[-1]
                grouped_obs[-2] += grouped_obs[-1]
                grouped_exp.pop()
                grouped_obs.pop()
            elif len(grouped_exp) == 0: # Si después de todo no se formaron grupos válidos
                 app.logger.error("group_categories: No se pudieron formar grupos válidos. Posiblemente muy pocos datos o parámetros erróneos.")
                 raise ValueError("No se pudieron formar grupos válidos para la prueba Chi-cuadrado.")


            # Final check: Ensure all expected values are positive
            if np.any(np.array(grouped_exp) <= 0):
                app.logger.warning("group_categories: Frecuencias esperadas cero o negativas después de agrupación. Ajustando a un valor mínimo.")
                # Reemplazar ceros o negativos con un valor muy pequeño para evitar errores de división por cero
                grouped_exp = [max(e, 1e-9) for e in grouped_exp]
            
            return np.array(grouped_obs), np.array(grouped_exp)

        # Aplicar la agrupación
        try:
            grouped_observed_np, grouped_expected_np = group_categories(observed_counts_np.tolist(), expected_counts_raw.tolist())
        except ValueError as e:
            results["conclusion"] = f"Error en la agrupación de categorías para Chi-cuadrado: {e}"
            return jsonify(results), 400

        # --- Ejecutar la Prueba Seleccionada ---
        if test_type == 'chi_square':
            try:
                # Comprobación final de la suma de los grupos
                if not np.isclose(np.sum(grouped_observed_np), np.sum(grouped_expected_np), atol=0.1): # atol para tolerancia
                    # Si las sumas no son cercanas, hay un problema grave con el cálculo o la agrupación.
                    app.logger.error(f"Chi-cuadrado: Suma de observados ({np.sum(grouped_observed_np)}) difiere de esperados ({np.sum(grouped_expected_np)}) después de agrupación.")
                    results["conclusion"] = "Error de consistencia de datos: La suma de frecuencias observadas no coincide con la suma de esperadas después de la agrupación."
                    return jsonify(results), 400

                stat, p_value = chisquare(f_obs=grouped_observed_np, f_exp=grouped_expected_np)
                
                df = len(grouped_observed_np) - 1 
                # Si df es 0 o negativo, la prueba no es válida (muy pocas categorías)
                if df <= 0:
                    results["conclusion"] = "Grados de libertad insuficientes para la prueba Chi-cuadrado. Agrupe más datos."
                    return jsonify(results), 400

                if np.isnan(stat) or np.isnan(p_value):
                    results["conclusion"] = "Error de cálculo: Estadístico o P-valor Chi-cuadrado resultó en NaN. Datos inadecuados para la prueba."
                    return jsonify(results), 400

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
                app.logger.error(f"Error al ejecutar la prueba Chi-cuadrado: {e}", exc_info=True)
                results["conclusion"] = f"Error al ejecutar la prueba Chi-cuadrado: {str(e)}"
                results["details"]["error_message"] = str(e)
                results["details"]["traceback"] = traceback.format_exc()
                return jsonify(results), 500

        elif test_type == 'kolmogorov_smirnov':
            try:
                # La prueba K-S espera una muestra continua. 
                # Convertir datos discretos (conteo por semestre) a una muestra para K-S
                synthetic_sample = []
                for i, count in enumerate(observed_data): 
                    # Cada 'i' representa el "valor" o "semestre"
                    if count > 0: 
                        synthetic_sample.extend([i] * int(count)) # Extendemos el valor 'i' 'count' veces
                
                if not synthetic_sample: 
                    results["conclusion"] = "No hay datos en la muestra observada para la prueba K-S."
                    return jsonify(results), 400

                synthetic_sample_np = np.array(synthetic_sample)

                if distribution_type == 'poisson':
                    # Para K-S con Poisson, se compara la CDF empírica con la CDF de Poisson
                    stat, p_value = kstest(synthetic_sample_np, lambda x: poisson.cdf(x, lambda_val))
                elif distribution_type == 'normal':
                    # Para K-S con Normal, se usa 'norm' con args (media, std_dev)
                    stat, p_value = kstest(synthetic_sample_np, 'norm', args=(mean_val, std_dev_val))
                elif distribution_type == 'exponential':
                    # Para K-S con Exponencial, se usa 'expon' con args (loc, scale)
                    # scale = 1/lambda para la exponencial
                    stat, p_value = kstest(synthetic_sample_np, 'expon', args=(0, 1/lambda_val)) # loc=0 para Exp estándar
                else:
                    results["conclusion"] = "Tipo de distribución no soportado para K-S."
                    return jsonify(results), 400
                
                if np.isnan(stat) or np.isnan(p_value):
                    results["conclusion"] = "Error de cálculo: Estadístico o P-valor K-S resultó en NaN. Datos inadecuados para la prueba."
                    return jsonify(results), 400

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
                app.logger.error(f"Error al ejecutar la prueba Kolmogorov-Smirnov: {e}", exc_info=True)
                results["conclusion"] = f"Error al ejecutar la prueba Kolmogorov-Smirnov: {str(e)}"
                results["details"]["error_message"] = str(e)
                results["details"]["traceback"] = traceback.format_exc() 
                return jsonify(results), 500
            
        else:
            app.logger.warning(f"run_goodness_of_fit_test: Tipo de prueba no soportado: {test_type}")
            results["conclusion"] = "Tipo de prueba no soportado o error en parámetros."
            return jsonify(results), 400

        return jsonify(results)

    except Exception as e:
        import traceback
        app.logger.error(f"Error general en run_goodness_of_fit_test: {e}", exc_info=True)
        return jsonify({"error": f"Error interno del servidor en la prueba de ajuste: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) 
    # En producción (Render), gunicorn manejará la ejecución.
    # Esta línea solo se ejecuta si corres el script directamente (ej. python app.py)
    app.run(host='0.0.0.0', port=port, debug=True) 
    # debug=True es útil para desarrollo local, pero en producción, gunicorn lo maneja.