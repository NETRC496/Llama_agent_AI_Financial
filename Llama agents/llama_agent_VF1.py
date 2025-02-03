import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from llama_agents import (
    AgentService,
    HumanService,
    AgentOrchestrator,
    CallableMessageConsumer,
    ControlPlaneServer,
    ServerLauncher,
    SimpleMessageQueue,
    QueueMessage,
)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

#Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("OPEN_AI_KEY")

def StockAns (stock: str) -> str :
    # Ticker del activo
    activo = str(stock)  # Ticker del activo
    datos = yf.Ticker(activo).history(period= 'max')

    # 2. Calcular los retornos promedio (Average) y la desviación estándar (std)
    datos['Average'] = datos[['High', 'Low']].mean(axis=1).pct_change()
    datos['std'] = datos['Average'].rolling(window=5).std()

    # Eliminar valores NaN para los ajustes
    datos.dropna(inplace=True)
    
    #Función para empezar analisis de datos sobre la acción solicitada
    def Model_Po_Ex (stock): 
    #This function generate the answer to the question "What is the risk and return of stock x?"
     # 1. Descargar datos históricos del activo
        try: 
            #filtrar datos
            datos_filtrados = datos[datos['Average'] > 0]

            # --- MODELO POLINOMIAL ---
            # Ajustar un modelo polinomial de grado 2
            X_poly = np.column_stack((datos['Average'], datos['Average']**2))  # [x, x^2]
            X_poly = sm.add_constant(X_poly)  # Añadir intercepto
            modelo_poly = sm.OLS(datos['std'], X_poly).fit()

            # Coeficientes del modelo polinomial
            alpha_poly = modelo_poly.params[0]
            beta_1_poly = modelo_poly.params[1]
            beta_2_poly = modelo_poly.params[2]

            # Predicciones del modelo polinomial
            datos['std_pred_poly'] = alpha_poly + beta_1_poly * datos['Average'] + beta_2_poly * datos['Average']**2

            # --- MODELO EXPONENCIAL ---
            # Definir la función exponencial
            def modelo_exponencial(x, alpha, beta):
                    return alpha * np.exp(beta * x)
            # Ajustar el modelo exponencial
            popt, _ = curve_fit(modelo_exponencial, datos['Average'], datos['std'], maxfev=10000)
            alpha_exp, beta_exp = popt

            # Predicciones del modelo exponencial
            datos['std_pred_exp'] = modelo_exponencial(datos['Average'], alpha_exp, beta_exp)

            # Crear un diccionario con los datos
            data1 = { 'Average': datos['Average'].tolist(), 
                    'std_pred_poly': datos['std_pred_poly'].tolist(), 
                    'std_pred_exp': datos['std_pred_exp'].tolist()}

            # Función para encontrar la intersección entre dos funciones
            def interseccion(x):
                std_poly = alpha_poly + beta_1_poly * x + beta_2_poly * x**2
                std_exp = alpha_exp * np.exp(beta_exp * x)
                return std_poly - std_exp

            # Valores iniciales para buscar las raíces
            valores_iniciales = [-0.05, 0.05]  # Suponiendo dos intersecciones

            # Resolver las raíces
            intersecciones = [fsolve(interseccion, x0)[0] for x0 in valores_iniciales]

            # Calcular las desviaciones estándar en las intersecciones
            resultados = [(x, alpha_poly + beta_1_poly * x + beta_2_poly * x**2) for x in intersecciones]

            # Mostrar los resultados
            Intersección = {}
            for i, (x, y) in enumerate(resultados):
                Intersección = {'Average': f"{x:.4f}", 'Risk': f"{y:.4f}"}
                # Crear un diccionario con las intersecciones
                intersecciones_dict = {
                    'Average': [x for x, y in resultados],
                    'Risk': [y for x, y in resultados]
                    }
            return intersecciones_dict
        except ValueError as e:
            return "Error en la ejecución de la función" + str(e)

    def Model_Po_Ex_2(stock):
        try:
            # --- MODELO POLINOMIAL ---
            # Ajustar un modelo polinomial de grado 2
            X_poly = np.column_stack((datos['Average'], datos['Average']**2))  # [x, x^2]
            X_poly = sm.add_constant(X_poly)  # Añadir intercepto
            modelo_poly = sm.OLS(datos['std'], X_poly).fit()

            # Coeficientes del modelo polinomial
            alpha_poly = modelo_poly.params[0]
            beta_1_poly = modelo_poly.params[1]
            beta_2_poly = modelo_poly.params[2]

            # Predicciones del modelo polinomial
            datos['std_pred_poly'] = alpha_poly + beta_1_poly * datos['Average'] + beta_2_poly * datos['Average']**2

            # --- MODELO EXPONENCIAL DE GRADO 2 ---
            # Definir la función exponencial de grado 2
            def modelo_exponencial_grado2(x, alpha, beta1, beta2):
                return alpha * np.exp(beta1 * x + beta2 * x**2)

            # Ajustar el modelo exponencial de grado 2
            popt, _ = curve_fit(modelo_exponencial_grado2, datos['Average'], datos['std'], maxfev=10000)
            alpha_exp2, beta1_exp2, beta2_exp2 = popt

            # Predicciones del modelo exponencial de grado 2
            datos['std_pred_exp2'] = modelo_exponencial_grado2(datos['Average'], alpha_exp2, beta1_exp2, beta2_exp2)

            # Función para encontrar la intersección entre dos funciones
            def interseccion(x):
                std_poly = alpha_poly + beta_1_poly * x + beta_2_poly * x**2
                std_exp2 = alpha_exp2 * np.exp(beta1_exp2 * x + beta2_exp2 * x**2)
                return std_poly - std_exp2

            # Valores iniciales para buscar las raíces
            valores_iniciales_2g = [-0.05, 0.05]  # Suponiendo dos intersecciones

            # Resolver las raíces
            intersecciones_2g = [fsolve(interseccion, x0)[0] for x0 in valores_iniciales_2g]
            # Calcular las desviaciones estándar en las intersecciones
            resultados_2g = [(x, alpha_poly + beta_1_poly * x + beta_2_poly * x**2) for x in intersecciones_2g]

            # Mostrar los resultados
            Intersección_2g = {}
            for i, (x, y) in enumerate(resultados_2g):
                    Intersección_2g = {'Average': f"{x:.4f}", 'Risk': f"{y:.4f}"}
            # Crear un diccionario con las intersecciones
            intersecciones_dict2g = {
                 'Average': [x for x, y in resultados_2g],
                 'Risk': [y for x, y in resultados_2g]
                }
            #Mostrar resultados
            return intersecciones_dict2g
        except ValueError as e:
                return "Error en la ejecución de la función" + str(e)
    def Model_log(stock):

        try:
            #Modelo 3
            #Convertir a logaritmos los datos de std
            datos['log_std'] = np.log(datos['std'])
            datos['log_Average'] = np.log(datos['Average'].replace(0, np.nan).dropna())
            # --- MODELO EXPONENCIAL CON LOGARITMOS ---

            # Ajustar el modelo exponencial transformado con logaritmos
            X_exp_log = sm.add_constant(datos['Average'])  # Log(σ) = Log(α) + β * Average
            modelo_exp_log = sm.OLS(datos['log_std'], X_exp_log).fit()

            # Coeficientes del modelo exponencial con logaritmos
            beta_exp_log = modelo_exp_log.params[1]

            # Predicciones del modelo exponencial con logaritmos
            datos['log_std_pred_exp'] = modelo_exp_log.params[0] + beta_exp_log * datos['Average']

            # Generar valores de x
            x_vals = np.linspace(datos['Average'].min(), datos['Average'].max(), 500)

            # Calcular valores de y para el modelo exponencial con logaritmos
            log_std_pred_exp = modelo_exp_log.params[0] + beta_exp_log * x_vals
            std_pred_exp = np.exp(log_std_pred_exp)  # Transformación inversa para obtener valores positivos

            # Filtrar valores de x y y que no superen el rango de 0.025 y sean positivos
            rango_mask_exp_log = (std_pred_exp > 0) & (std_pred_exp <= 0.0455)

            # Crear diccionario con los datos filtrados
            datos_filtrados = {
                'Average': x_vals[rango_mask_exp_log],
                'Risk': std_pred_exp[rango_mask_exp_log]
                }

                # Convertir a DataFrame
            df_filtrado = pd.DataFrame(datos_filtrados)

                # Calcular máximos y mínimos, acotación de la ecuación.
            max_average = df_filtrado['Average'].max()
            min_average = df_filtrado['Average'].min()
            max_risk = df_filtrado['Risk'].max()
            min_risk = df_filtrado['Risk'].min()

            # Mostrar resultados
            # Crear diccionario con los resultados
            resultados_log = {
                'Average': [max_average,min_average],
                'Risk': [max_risk, min_risk]
                }
            return resultados_log
        except ValueError as e:
            return "Error en la ejecución del modelo: " + str(e)
    # Comparar los modelos y mostrar el mejor resultado
    def compare_models():
        try:
            # Ejecutar los modelos y capturar sus resultados
            print("Ejecutando Model_Po_Ex...")
            print("\nEjecutando Model_Po_Ex_2...")
            print("\nEjecutando Model_log...")
            print("Generando resultados...")

            #Ver los resultados de los modelos ejecutados
            resultados_modelo_1 = Model_Po_Ex(stock)
            resultados_modelo_2 = Model_Po_Ex_2(stock)
            resultados_modelo_3 = Model_log(stock)

            #Convertir los modelos ejecutados en dataframe para procesarlos
            modelo_1 = pd.DataFrame(resultados_modelo_1)
            modelo_2 = pd.DataFrame(resultados_modelo_2)
            modelo_3 = pd.DataFrame(resultados_modelo_3)

           #Clasificar y especificar los valores de los 3 dataframe
           #Clasificación de variable del modelo 1
            return_op_model_1 = modelo_1

            #datax1_return_model_1_Av = return_op_model_1.iloc[0,0] #X1 Average de modelo 1 (X1-1)
            #datax2_return_model_1_Av = return_op_model_1.iloc[1,0] #X2 Average de modelo 1 (X2-1)
            #datay1_return_model_1_Risk = return_op_model_1.iloc[0,1] #Y1 Risk de modelo 1 (Y1-1)
            #datay2_return_model_1_Risk = return_op_model_1.iloc[1,1] #Y2 Risk de modelo 1 (Y2-1)

            #Clasificación de variable del modelo 2
            return_op_model_2 = modelo_2

            #datax1_return_model_2_Av = return_op_model_2.iloc[0,0] #X1 Average de modelo 2 (X1-2)
            #datax2_return_model_2_Av = return_op_model_2.iloc[1,0] #X2 Average de modelo 2 (X2-2)
            #datay1_return_model_2_Risk = return_op_model_2.iloc[0,1] #Y1 Risk de modelo 2 (Y1-2)
            #datay2_return_model_2_Risk = return_op_model_2.iloc[1,1] #Y2 Risk de modelo 2 (Y2-2)

            #Clasificación de variable del modelo 3       
            return_op_model_3 = modelo_3

            #datax1_return_model_3_Av = return_op_model_3.iloc[0,0] #X1 Average de modelo 3 (X1-3)
            #datax2_return_model_3_Av = return_op_model_3.iloc[1,0] #X2 Average de modelo 3 (X2-3)
            #datay1_return_model_3_Risk = return_op_model_3.iloc[0,1] #Y1 Average de modelo 3 (Y1-3)
            #datay2_return_model_3_Risk = return_op_model_3.iloc[1,1] #Y2 Average de modelo 3 (Y2-3)


            #Clasificación de comparación de datos

            # X1-1 > X2-1
            # X1-1 < X2-1
            def CC_Data_1 (return_op_model_1):
                if return_op_model_1.iloc[0,0] > return_op_model_1.iloc[1,0]:
                    BM1_1 = return_op_model_1.iloc[0,0]
                    return BM1_1
                elif return_op_model_1.iloc[0,0] < return_op_model_1.iloc[1,0]:
                    BM2_1 = return_op_model_1.iloc[1,0]
                    return BM2_1
                #En caso de igualdad, poco probable
                else:
                    return (BM1_1 or BM2_1)
                
            #X1-2 > X2-2
            #X1-2 < x2-2
            def CC_Data_2(return_op_model_2):
                if return_op_model_2.iloc[0,0] > return_op_model_2.iloc[1,0]:
                    BM1_2 = return_op_model_2.iloc[0,0]
                    return BM1_2
                elif return_op_model_2.iloc[0,0] < return_op_model_2.iloc[1,0]:
                    BM2_2 = return_op_model_2.iloc[1,0]
                    return BM2_2
                #En caso de igualdad, poco probable
                else:
                    return (BM1_2 or BM2_2)
                
            #X1-3 > X2-3
            #X1-3 < x2-3
            def CC_Data_3(return_op_model_3):
                if return_op_model_3.iloc[0,0] > return_op_model_3.iloc[1,0]:
                    BM1_3 = return_op_model_3.iloc[0,0]
                    return BM1_3
                elif return_op_model_3.iloc[0,0] < return_op_model_3.iloc[1,0]:
                    BM2_3 = return_op_model_3.iloc[1,0]
                    return BM2_3
                #En caso de igualdad, poco probable
                else:
                    return (BM1_3 or BM2_3)
                
            #De los resultados de las anteriores funciones, se hace el ultimo proceso para sacar el mejor rendimiento
            def CC_Data():
                #Primera comparación
                def FCC_Data():
                    if CC_Data_1(return_op_model_1) > CC_Data_2(return_op_model_2):
                        BFCM1 = CC_Data_1(return_op_model_1)
                        return BFCM1
                    elif CC_Data_1(return_op_model_1) < CC_Data_2(return_op_model_2):
                        BFCM2 = CC_Data_2(return_op_model_2)
                        return BFCM2
                    #En caso de igualdad, poco probable
                    else:
                        return (BFCM1 or BFCM2)
                def SCC_Data():
                    if FCC_Data() > CC_Data_3(return_op_model_3):
                        BSCM1 = FCC_Data()
                        return BSCM1
                    elif FCC_Data() < CC_Data_3(return_op_model_3):
                        BSCM2 = CC_Data_3(return_op_model_3)
                        return BSCM2 
                    #En caso de igualdad, poco probable 
                    else:
                        return (BSCM1 or BSCM2)
                return SCC_Data()
            
            def EV_Data():
                def EVC_Data():
                    if CC_Data() == CC_Data_1(return_op_model_1):
                        data_1 = CC_Data_1(return_op_model_1)
                        BM1_1 = return_op_model_1.iloc[0,0]
                        BM2_1 = return_op_model_1.iloc[1,0]
                        data_1 == BM1_1 or data_1 == BM2_1
                        return data_1
                    elif CC_Data() == CC_Data_2(return_op_model_2):
                        data_2 = CC_Data_2(return_op_model_2)
                        BM1_2 = return_op_model_2.iloc[0,0]
                        BM2_2 = return_op_model_2.iloc[1,0]
                        data_2 == BM1_2 or data_2 == BM2_2
                        return data_2
                    else:
                        data_3 = CC_Data_3(return_op_model_3)
                        BM1_3 = return_op_model_3.iloc[0,0]
                        BM2_3 = return_op_model_3.iloc[1,0]
                        data_3 == BM1_3 or data_3 == BM2_3
                        return data_3
                    
                def EVCR_data(): #Busqueda de la variable del mejor resultado en los modelos 

                    def EVCRM1_data():#Busqueda en el primer modelo
                        if EVC_Data() == return_op_model_1.iloc[0,0]: #X1-1 (Primera variable del primer modelo)
                            YM1_1 = return_op_model_1.iloc[0,1] #Prueba de Y1-1
                            return YM1_1
                        elif EVC_Data() == return_op_model_1.iloc[1,0]: #X2-1 (segundavariable del primer modelo)
                            YM2_1 = return_op_model_1.iloc[1,1] #Prueba de Y2-1
                            return YM2_1
                        else: 
                            print("Error en encontrar el riesgo para el rendimiento asociado")       
                    def EVCRM2_data(): #Busqueda en el segundo modelo
                        if EVC_Data() == return_op_model_2.iloc[0,0]: #X1-2 (Primera variable del segundo modelo)
                            YM1_2 = return_op_model_2.iloc[0,1] #Prueba de Y1-2
                            return YM1_2
                        elif EVC_Data() == return_op_model_2.iloc[1,0]: #X2-2 (Segunda variable del segundo modelo)
                            YM2_2 = return_op_model_2.iloc[1,1] #Prueba de Y2-2
                            return YM2_2
                        else: 
                            print("Error en encontrar el riesgo para el rendimiento asociado")
                    def EVCRM3_data(): #Busqueda de la variable del mejor resultado
                        if EVC_Data() == return_op_model_3.iloc[0,0]: #X1-3 (Primera variable del tercer modelo)
                            YM1_3 = return_op_model_3.iloc[0,1] #Prueba de Y1-3
                            return YM1_3
                        elif EVC_Data() == return_op_model_3.iloc[1,0]: #X2-3 (Segunda variable del tercer modelo)
                            YM2_3 = return_op_model_3.iloc[1,1] #Prueba de Y2-3
                            return YM2_3
                        else: 
                            print("Error en encontrar el riesgo para el rendimiento asociado")
                    return EVCRM3_data()  
                return print("El rendimiento asociado a la acción es la siguiete",EVC_Data(), "El riesgo asociado a la acción es la siguiente",EVCR_data())
            return EV_Data()
        except Exception as e:
            print("Error al comparar los modelos: " + str(e))
    return print(compare_models())

#Crear herramientas de las funciones
Financial_Analysis_tool = FunctionTool.from_defaults(fn=StockAns)

#Crear workers y agentes
Worker1 = FunctionCallingAgentWorker.from_tools(Financial_Analysis_tool)
Worker2 = FunctionCallingAgentWorker.from_tools([], llm=OpenAI)
Agent1 = Worker1.as_agent()
Agent2 = Worker2.as_agent()

#Crear nuestro multiagente framework components
message_queue = SimpleMessageQueue()
queue_client = message_queue.client

control_plane = ControlPlaneServer(
    message_queue=queue_client,
    orchestrator = AgentOrchestrator(llm=OpenAI),
)
Agent_server_1 = AgentService(
    agent=Agent1,
    message_queue=queue_client,
    description="Brinda información sobre los activos, sus rendimientos y riesgos de inversión.",
    service_name="Financial_Analysis_Agent",
    host="127.0.0.1",
    port=8002,
)
human_service=AgentService(
    agent=Agent2,
    message_queue= queue_client,
    description="Respuestas a preguntas generales",
    host="127.0.0.1",
    port=8003,
)
#Human Consumer Adicional
def handle_result(message:QueueMessage) -> None:
    print("Resultado:", message.data)

human_consumer = CallableMessageConsumer(handler=handle_result, message_type="human")

#Lanzar los servers
launcher = ServerLauncher(
    [Agent_server_1, human_service],
    control_plane,
    message_queue,
    additional_consumers=[human_consumer],
)
import nest_asyncio
import asyncio

nest_asyncio.apply()

async def main():
    await launcher.launch_servers()
asyncio.run(main())
