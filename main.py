import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.metrics import r2_score

# Configuración visual general
sns.set_theme(style='whitegrid')
pd.set_option('display.width', 200)  # evita que las tablas se corten con "..."
pd.set_option('display.max_columns', None)

# Colores de las gráficas nuevas (paleta categórica apta para daltonismo)
AZUL, NARANJA, AQUA = '#2a78d6', '#eb6834', '#1baf7a'

# ==============================================================================
# PARÁMETROS QUE PUEDES AJUSTAR
# ==============================================================================
RUTA_CSV = 'student_performance copy.csv'
# RUTA_CSV = 'student_performance.csv'

# EDITA esta pregunta si tu grupo planteó otra (debe usar las 2 variables cuantitativas)
PREGUNTA = (
    '¿En qué medida las horas de estudio autónomo semanales explican el puntaje'
    ' total de los estudiantes y esa relación cambia según su nivel de'
    ' asistencia?'
)
JUSTIFICACION_BASE = (
    'Registra por estudiante sus horas de estudio, asistencia, participación,'
    ' puntaje y calificación; permite relacionar dos variables cuantitativas'
    ' (horas y puntaje) y cruzarlas con variables cualitativas (calificación y'
    ' nivel de asistencia).'
)

# Decisión de limpieza sobre los atípicos (ver sección 1):
#   False -> se CONSERVAN (se asume que son estudiantes reales)
#   True  -> se ELIMINAN (se asume que son errores de captura)
ELIMINAR_ATIPICOS = False

# Mejora mínima del R² ajustado para preferir el modelo cuadrático sobre el lineal
UMBRAL_MEJORA_R2AJ = 0.02

vars_cuanti = ['weekly_self_study_hours', 'total_score']
vars_cuali = ['grade', 'nivel_asistencia']


# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================
def limites_iqr(s):
  q1, q3 = s.quantile([0.25, 0.75])
  iqr = q3 - q1
  return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def moda_txt(s):
  frec = s.value_counts()
  if frec.iloc[0] == 1:
    return 'no hay (todos los valores son distintos)'
  modas = frec[frec == frec.iloc[0]].index.sort_values()
  return ', '.join(f'{m:.2f}' for m in modas[:3]) + f' (se repite {frec.iloc[0]} veces)'


def clase_modal(s):
  cuentas, bordes = np.histogram(s, bins='sturges')
  i = cuentas.argmax()
  return f'[{bordes[i]:.1f}, {bordes[i + 1]:.1f}]'


def fuerza(r):
  a = abs(r)
  if a < 0.2:
    return 'muy débil'
  if a < 0.4:
    return 'débil'
  if a < 0.6:
    return 'moderada'
  if a < 0.8:
    return 'fuerte'
  return 'muy fuerte'


def r2_ajustado(r2, n, k):
  return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def texto_p(p):
  return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


def metricas(d):
  """Resumen de un escenario de limpieza: tamaño, media, desviación y R²."""
  d = d[vars_cuanti].dropna()
  x, y = d[vars_cuanti[0]].to_numpy(), d[vars_cuanti[1]].to_numpy()
  return {
      'n': len(d),
      'media_horas': x.mean(),
      'desv_horas': x.std(ddof=1),
      'media_puntaje': y.mean(),
      'desv_puntaje': y.std(ddof=1),
      'R': np.corrcoef(x, y)[0, 1],
      'R2_lineal': r2_score(y, np.poly1d(np.polyfit(x, y, 1))(x)),
      'R2_cuadratico': r2_score(y, np.poly1d(np.polyfit(x, y, 2))(x)),
  }


# ==============================================================================
# 0. PREGUNTA, BASE DE DATOS Y VARIABLES (Criterio 1 - 5%)
# ==============================================================================
df = pd.read_csv(RUTA_CSV)

print('=== 0. PREGUNTA DE INVESTIGACIÓN Y BASE DE DATOS ===')
print('Pregunta:', PREGUNTA)
print('Base de datos:', JUSTIFICACION_BASE)

diccionario = pd.DataFrame(
    [
        ['student_id', 'Identificador del estudiante', 'Cualitativa nominal (no se analiza)'],
        ['weekly_self_study_hours', 'Horas de estudio autónomo por semana', 'Cuantitativa continua'],
        ['attendance_percentage', 'Porcentaje de asistencia a clase', 'Cuantitativa continua'],
        ['class_participation', 'Participación en clase (escala numérica)', 'Cuantitativa continua'],
        ['total_score', 'Puntaje total (0 - 100)', 'Cuantitativa continua'],
        ['grade', 'Calificación en letras (derivada del puntaje)', 'Cualitativa ordinal'],
        ['nivel_asistencia', 'Asistencia categorizada: Baja / Media / Alta (se crea desde attendance_percentage)', 'Cualitativa ordinal'],
    ],
    columns=['variable', 'descripción', 'tipo'],
)
print('\n--- DESCRIPCIÓN DE VARIABLES ---')
print(diccionario.to_string(index=False))
print('\nVariables cuantitativas del estudio:', vars_cuanti)
print('Variables cualitativas del estudio :', vars_cuali)

print('\n=== INFORMACIÓN GENERAL Y TIPOS DE DATOS ===')
print(df.info())
print('\n=== PRIMERAS FILAS ===')
print(df.head())

# ==============================================================================
# 1. LIMPIEZA DE DATOS Y ANÁLISIS DE IMPACTO (Criterio 2 - 10%)
# ==============================================================================
print('\n=== 1. LIMPIEZA DE DATOS ===')
df_original = df.copy()  # copia sin tocar, para comparar ANTES vs DESPUÉS
resumen_antes = df_original[vars_cuanti].agg(['count', 'mean', 'median', 'std'])

# --- 1.1 Valores imposibles (errores de captura) -> se convierten en nulos ---
print('\n--- 1.1 Valores imposibles ---')
reglas = {
    'weekly_self_study_hours': (0, 168),  # una semana tiene 168 horas
    'attendance_percentage': (0, 100),
    'total_score': (0, 100),
}
for col, (lo, hi) in reglas.items():
  malos = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
  print(f'{col}: {malos.sum()} valores fuera de [{lo}, {hi}] -> se tratan como nulos')
  df.loc[malos, col] = np.nan

# Segunda variable cualitativa: se crea categorizando la asistencia.
# include_lowest=True evita que una asistencia de exactamente 0% quede como nulo.
if 'nivel_asistencia' not in df.columns:
  df['nivel_asistencia'] = pd.cut(
      df['attendance_percentage'],
      bins=[0, 80, 90, 100],
      labels=['Baja (<=80%)', 'Media (80-90%)', 'Alta (>90%)'],
      include_lowest=True,
  )

# --- 1.2 Nulos ---
print('\n--- 1.2 Nulos ---')
cols_analisis = list(dict.fromkeys(vars_cuanti + vars_cuali + ['attendance_percentage']))
print('Nulos por columna (variables del análisis):\n', df[cols_analisis].isnull().sum())

filas_con_nulos = df[cols_analisis].isnull().any(axis=1)
pct_nulos = filas_con_nulos.mean() * 100

# Se eliminan las filas incompletas para usar la MISMA muestra en todas las secciones
df_sin_nulos = df.dropna(subset=cols_analisis).copy()
print(f'\nFilas con al menos un nulo: {filas_con_nulos.sum()} ({pct_nulos:.2f}% de {len(df)})')
if pct_nulos < 5:
  print(
      'DECISIÓN: ELIMINAR filas con nulos. Justificación: afectan a menos del 5%'
      ' de los datos, así que imputar aportaría más sesgo que información y se'
      ' mantiene la misma muestra en todos los análisis.'
  )
else:
  print(
      'ATENCIÓN: los nulos superan el 5% de las filas. Eliminarlos puede sesgar la'
      ' muestra; considera imputar (p. ej., con la mediana) y justifícalo.'
  )

# --- 1.3 Atípicos (regla del rango intercuartílico) ---
print('\n--- 1.3 Atípicos (IQR) ---')
mask_atipicos = pd.Series(False, index=df_sin_nulos.index)
for col in vars_cuanti:
  lim_inf, lim_sup = limites_iqr(df_sin_nulos[col])
  es_atipico = (df_sin_nulos[col] < lim_inf) | (df_sin_nulos[col] > lim_sup)
  mask_atipicos |= es_atipico
  valores = sorted(df_sin_nulos.loc[es_atipico, col].round(2).tolist())
  print(
      f'Atípicos en {col}: {es_atipico.sum()} (límites [{lim_inf:.2f}, {lim_sup:.2f}])'
      f' -> valores: {valores[:10]}'
  )

df_sin_atipicos = df_sin_nulos[~mask_atipicos].copy()

# Efecto de los atípicos sobre los resultados (se mide siempre, sin importar la decisión)
escenarios = pd.DataFrame({
    'Con atípicos (sin nulos)': metricas(df_sin_nulos),
    'Sin atípicos': metricas(df_sin_atipicos),
}).T
print('\n--- EFECTO DE LOS ATÍPICOS EN LOS RESULTADOS ---')
print(escenarios.round(3))
esc_con, esc_sin = escenarios.iloc[0], escenarios.iloc[1]
print(
    f'Al quitar los atípicos: R² lineal {esc_con.R2_lineal:.3f} -> {esc_sin.R2_lineal:.3f};'
    f' desv. est. de horas {esc_con.desv_horas:.2f} -> {esc_sin.desv_horas:.2f};'
    f' desv. est. de puntaje {esc_con.desv_puntaje:.2f} -> {esc_sin.desv_puntaje:.2f}.'
)

if ELIMINAR_ATIPICOS:
  df_limpio = df_sin_atipicos
  print(
      f'\nDECISIÓN: ELIMINAR {mask_atipicos.sum()} filas con atípicos.'
      ' Justificación: se asumen errores de captura que distorsionan la media, la'
      ' desviación y el ajuste de los modelos (ver tabla de efecto).'
  )
else:
  df_limpio = df_sin_nulos
  print(
      f'\nDECISIÓN: CONSERVAR los {mask_atipicos.sum()} atípicos. Justificación:'
      ' están dentro de los rangos posibles, así que se consideran estudiantes'
      ' reales con desempeño extremo y no errores; eliminarlos reduciría la'
      ' muestra y ocultaría casos genuinos. Su efecto se reporta en la tabla de'
      ' arriba. (Si al revisarlos alguno resulta ser un error de captura, pon'
      ' ELIMINAR_ATIPICOS = True.)'
  )

resumen_despues = df_limpio[vars_cuanti].agg(['count', 'mean', 'median', 'std'])

print('\n--- COMPARACIÓN ANTES VS DESPUÉS DE LIMPIEZA (Para sustentar) ---')
print('ANTES:\n', resumen_antes.round(2))
print('\nDESPUÉS:\n', resumen_despues.round(2))
print(
    f'\nBase final: {len(df_limpio)} de {len(df_original)} filas'
    f' ({len(df_limpio) / len(df_original) * 100:.1f}% conservado).'
)

# ==============================================================================
# 2. COMPARACIÓN DESCRIPTIVA Y GRÁFICOS (Criterio 3 - 15%)
# ==============================================================================
print('\n=== 2. MEDIDAS DE TENDENCIA CENTRAL, POSICIÓN Y DISPERSIÓN ===')
cvs = {}
for col in vars_cuanti:
  s = df_limpio[col]
  q1, q2, q3 = s.quantile([0.25, 0.50, 0.75])
  cv = (s.std() / s.mean()) * 100
  cvs[col] = cv
  print(f'\nVariable: {col}')
  print(
      f'  Tendencia central -> Media: {s.mean():.2f} | Mediana: {q2:.2f} |'
      f' Moda: {moda_txt(s)} | Clase modal: {clase_modal(s)}'
  )
  print(
      f'  Posición          -> Mín: {s.min():.2f} | Q1: {q1:.2f} | Q2:'
      f' {q2:.2f} | Q3: {q3:.2f} | P90: {s.quantile(0.90):.2f} | Máx:'
      f' {s.max():.2f}'
  )
  print(
      f'  Dispersión        -> Rango: {s.max()-s.min():.2f} | IQR:'
      f' {q3-q1:.2f} | Var: {s.var():.2f} | Desv.Est: {s.std():.2f} | CV:'
      f' {cv:.2f}%'
  )

  # Interpretación automática
  asim = s.skew()
  if abs(asim) < 0.5:
    forma = 'aproximadamente simétrica'
  elif asim < 0:
    forma = 'asimétrica a la izquierda (cola hacia valores bajos)'
  else:
    forma = 'asimétrica a la derecha (cola hacia valores altos)'
  het = 'baja' if cv < 15 else 'moderada' if cv < 30 else 'alta'
  print(
      f'  Interpretación    -> Distribución {forma} (asimetría = {asim:.2f}; media {s.mean():.2f} vs mediana {q2:.2f});'
      f' variabilidad relativa {het} (CV = {cv:.1f}%). El 50% central de los'
      f' datos está entre {q1:.2f} y {q3:.2f}.'
  )
  pct_max = (s == s.max()).mean() * 100
  if pct_max >= 5:
    print(
        f'  Efecto techo      -> {pct_max:.1f}% de las observaciones valen'
        f' exactamente el máximo ({s.max():.2f}).'
    )

mayor_cv = max(cvs, key=cvs.get)
print(
    f'\nComparación: la mayor variabilidad relativa está en {mayor_cv}'
    f' (CV = {cvs[mayor_cv]:.1f}%) frente a'
    f' {min(cvs, key=cvs.get)} (CV = {min(cvs.values()):.1f}%).'
)

# Gráficos: Boxplots individuales + Gráfico de puntos
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.boxplot(y=df_limpio['weekly_self_study_hours'], ax=axes[0], color='#4C72B0')
axes[0].set_title('Distribución de Horas de Estudio Semanal')
axes[0].set_ylabel('Horas por semana')

sns.boxplot(y=df_limpio['total_score'], ax=axes[1], color='#55A868')
axes[1].set_title('Distribución del Puntaje Total')
axes[1].set_ylabel('Puntaje (0 - 100)')

sns.scatterplot(
    data=df_limpio,
    x='weekly_self_study_hours',
    y='total_score',
    ax=axes[2],
    alpha=0.6,
)
axes[2].set_title('Relación: Horas de Estudio vs. Puntaje Total')
axes[2].set_xlabel('Horas de estudio semanal')
axes[2].set_ylabel('Puntaje total')

plt.tight_layout()
plt.show()

# ==============================================================================
# 3 y 4. CORRELACIÓN Y COMPARACIÓN DE MODELOS: LINEAL VS CUADRÁTICO (Criterios 4 y 5 - 30%)
# ==============================================================================
x = df_limpio['weekly_self_study_hours'].values
y = df_limpio['total_score'].values
n = len(x)

# Modelo 1: Lineal (y = m*x + b)
coef_lin = np.polyfit(x, y, 1)
poly_lin = np.poly1d(coef_lin)
y_pred_lin = poly_lin(x)
r2_lin = r2_score(y, y_pred_lin)
r_lin = np.corrcoef(x, y)[0, 1]  # Coeficiente de correlación de Pearson (R)

# Modelo 2: Cuadrático (y = a*x^2 + b*x + c)
coef_cuad = np.polyfit(x, y, 2)
poly_cuad = np.poly1d(coef_cuad)
y_pred_cuad = poly_cuad(x)
r2_cuad = r2_score(y, y_pred_cuad)
r_cuad = np.sqrt(r2_cuad)  # Correlación múltiple R para el ajuste polinómico

# Criterios para comparar: el R² del cuadrático SIEMPRE es >= al del lineal (tiene
# un parámetro más), así que se usa también el R² ajustado y una prueba F.
r2aj_lin = r2_ajustado(r2_lin, n, 1)
r2aj_cuad = r2_ajustado(r2_cuad, n, 2)
rss_lin = np.sum((y - y_pred_lin) ** 2)
rss_cuad = np.sum((y - y_pred_cuad) ** 2)
F = (rss_lin - rss_cuad) / (rss_cuad / (n - 3))
p_f = stats.f.sf(F, 1, n - 3)

print('\n=== 3 y 4. CORRELACIÓN Y MODELOS DE REGRESIÓN ===')
print(f'n = {n} estudiantes')
print('MODELO LINEAL:')
print(f'  Ecuación: y = {coef_lin[0]:.4f}x + {coef_lin[1]:.4f}')
print(f'  R = {r_lin:.4f} | R² = {r2_lin:.4f} ({r2_lin*100:.2f}% explicado)')

print('\nMODELO CUADRÁTICO:')
print(
    f'  Ecuación: y = {coef_cuad[0]:.4f}x² + {coef_cuad[1]:.4f}x +'
    f' {coef_cuad[2]:.4f}'
)
print(
    f'  R = {r_cuad:.4f} | R² = {r2_cuad:.4f} ({r2_cuad*100:.2f}% explicado)'
)

# --- Interpretación de R y R² (Criterio 4) ---
print('\n--- INTERPRETACIÓN DE LA CORRELACIÓN ---')
if r_lin > 0:
  direccion = 'positiva (al aumentar las horas de estudio tiende a aumentar el puntaje)'
else:
  direccion = 'negativa (al aumentar las horas de estudio tiende a disminuir el puntaje)'
print(f'Dirección: correlación {direccion}.')
print(f'Fuerza   : {fuerza(r_lin)} (R = {r_lin:.3f}).')
print(
    f'R²       : {r2_lin*100:.1f}% de la variabilidad del puntaje total se asocia'
    f' linealmente con las horas de estudio; el {100*(1-r2_lin):.1f}% restante'
    ' depende de factores no incluidos (p. ej., asistencia, participación) y'
    ' del azar.'
)
print(
    'Causalidad: correlación NO implica causalidad. Que ambas variables varíen'
    ' juntas no prueba que estudiar más cause el mayor puntaje (pueden influir'
    ' terceras variables, como la motivación o la asistencia).'
)
print('Límites del análisis:')
print('  - R de Pearson solo mide asociación LINEAL; una relación curva la subestima.')
pct_techo = (y >= 100).mean() * 100
if pct_techo >= 5:
  print(
      f'  - {pct_techo:.1f}% de los estudiantes tiene el puntaje máximo (100): el'
      ' puntaje está "topado", lo que aplana la relación en las horas altas.'
  )
print(
    f'  - Efecto de los atípicos: ver tabla de la sección 1 (R² lineal'
    f' {esc_con.R2_lineal:.3f} con atípicos vs {esc_sin.R2_lineal:.3f} sin ellos).'
)

# --- Comparación y elección del modelo (Criterio 5) ---
x_ordenado = np.linspace(x.min(), x.max(), 200)
res_lin = y - y_pred_lin
res_cuad = y - y_pred_cuad
tramos = pd.qcut(x, q=8, duplicates='drop')  # 8 tramos con igual nº de estudiantes


def residuo_por_tramo(res):
  d = pd.DataFrame({'x': x, 'res': res, 'tramo': tramos})
  return d.groupby('tramo', observed=True).agg(x=('x', 'mean'), res=('res', 'mean'))


g_lin, g_cuad = residuo_por_tramo(res_lin), residuo_por_tramo(res_cuad)

print('\n--- COMPARACIÓN DE MODELOS ---')
print(
    pd.DataFrame(
        {
            'R': [r_lin, r_cuad],
            'R²': [r2_lin, r2_cuad],
            'R² ajustado': [r2aj_lin, r2aj_cuad],
        },
        index=['Lineal', 'Cuadrático'],
    ).round(4)
)
print(
    f'Mejora del cuadrático: ΔR² = {r2_cuad - r2_lin:.4f} | ΔR² ajustado ='
    f' {r2aj_cuad - r2aj_lin:.4f} | prueba F del término x²: F = {F:.2f},'
    f' {texto_p(p_f)}'
)
print(
    'Residuo medio máximo por tramo de horas (error sistemático):'
    f' lineal {g_lin["res"].abs().max():.2f} pts vs cuadrático'
    f' {g_cuad["res"].abs().max():.2f} pts'
)

elige_cuad = (p_f < 0.05) and ((r2aj_cuad - r2aj_lin) >= UMBRAL_MEJORA_R2AJ)
if elige_cuad:
  print(
      '\nMODELO ELEGIDO: CUADRÁTICO. Justificación: el término x² es'
      f' estadísticamente significativo ({texto_p(p_f)}) y la mejora del R²'
      f' ajustado ({r2aj_cuad - r2aj_lin:.3f}) supera el umbral de'
      f' {UMBRAL_MEJORA_R2AJ}, así que no se debe solo a tener un parámetro más.'
      ' Además, el cuadrático reduce el error sistemático del lineal (ver'
      ' gráfica de residuales).'
  )
  poly_elegido = poly_cuad
else:
  print(
      '\nMODELO ELEGIDO: LINEAL. Justificación: el cuadrático mejora poco'
      f' (ΔR² ajustado = {r2aj_cuad - r2aj_lin:.3f}, umbral {UMBRAL_MEJORA_R2AJ};'
      f' {texto_p(p_f)}), así que se prefiere el modelo más simple (parsimonia).'
  )
  poly_elegido = poly_lin

# --- Interpretación de coeficientes del modelo elegido ---
print('\n--- INTERPRETACIÓN DE COEFICIENTES (modelo elegido) ---')
x_min, x_max = x.min(), x.max()
if not elige_cuad:
  m, b0 = coef_lin
  print(
      f'  Pendiente m = {m:.4f}: por cada hora adicional de estudio semanal, el'
      f' puntaje total {"aumenta" if m > 0 else "disminuye"} en promedio'
      f' {abs(m):.2f} puntos.'
  )
  extra = (
      ''
      if x_min <= 0
      else f' (extrapolación: el mínimo observado es {x_min:.1f} h, no se debe'
      ' tomar literalmente)'
  )
  print(f'  Intercepto b = {b0:.4f}: puntaje esperado con 0 horas de estudio{extra}.')
else:
  a2, b2, c2 = coef_cuad
  pend_media = 2 * a2 * x.mean() + b2
  forma_cur = (
      'cóncava: rendimientos decrecientes, cada hora adicional aporta menos'
      ' puntos que la anterior'
      if a2 < 0
      else 'convexa: rendimientos crecientes, cada hora adicional aporta más'
      ' puntos que la anterior'
  )
  print(f'  a = {a2:.4f}: curvatura {forma_cur}.')
  print(
      f'  b = {b2:.4f} y c = {c2:.4f}: fijan la pendiente y el nivel en x = 0'
      ' (sin sentido práctico por separado); se interpretan con el efecto'
      ' marginal 2ax + b.'
  )
  print(
      f'  Efecto marginal en la media ({x.mean():.1f} h): {pend_media:+.2f}'
      ' puntos por hora adicional.'
  )
  x_v = -b2 / (2 * a2)
  if x_min <= x_v <= x_max:
    print(
        f'  Vértice: el modelo predice un {"máximo" if a2 < 0 else "mínimo"} de'
        f' ≈ {poly_cuad(x_v):.1f} puntos con {x_v:.1f} h/semana; a partir de ahí'
        f' la curva {"baja" if a2 < 0 else "sube"} (ojo: puede deberse al tope de'
        ' 100 puntos y no a que estudiar más perjudique).'
    )
  else:
    print(
        f'  Vértice fuera del rango observado (x = {x_v:.1f} h): dentro de los'
        f' datos la curva es {"creciente" if pend_media > 0 else "decreciente"}'
        ' sin cambiar de sentido.'
    )
if poly_elegido(x_ordenado).max() > 100:
  print(
      f'  Aviso: el modelo predice hasta {poly_elegido(x_ordenado).max():.1f}'
      ' puntos dentro del rango observado, pero el puntaje máximo posible es 100.'
  )

# Gráfico comparativo de ambos modelos sobre los datos
etq_lin = f'Lineal (R² = {r2_lin:.4f})' + ('' if elige_cuad else ' - elegido')
etq_cuad = f'Cuadrático (R² = {r2_cuad:.4f})' + (' - elegido' if elige_cuad else '')
plt.figure(figsize=(9, 6))
plt.scatter(x, y, alpha=0.5, label='Datos reales', color='gray')
plt.plot(
    x_ordenado,
    poly_lin(x_ordenado),
    color='blue',
    linewidth=2,
    label=etq_lin,
)
plt.plot(
    x_ordenado,
    poly_cuad(x_ordenado),
    color='red',
    linestyle='--',
    linewidth=2,
    label=etq_cuad,
)
plt.axhline(100, color='gray', linestyle=':', linewidth=1)
plt.title('Comparación de Ajuste: Modelo Lineal vs. Modelo Cuadrático')
plt.xlabel('Horas de estudio semanal')
plt.ylabel('Puntaje total')
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# GRÁFICA NUEVA 1: Residuales de ambos modelos (respalda la elección del modelo)
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(x, res_lin, s=16, alpha=0.22, color=AZUL, edgecolor='none')
ax.scatter(x, res_cuad, s=16, alpha=0.22, color=NARANJA, marker='s', edgecolor='none')
ax.axhline(0, color='#52514e', linewidth=1)
ax.plot(
    g_lin['x'], g_lin['res'], '-o', color=AZUL, linewidth=2.5, markersize=9,
    markeredgecolor='white', markeredgewidth=1.5, label='Lineal: residuo medio por tramo',
)
ax.plot(
    g_cuad['x'], g_cuad['res'], '-s', color=NARANJA, linewidth=2.5, markersize=9,
    markeredgecolor='white', markeredgewidth=1.5, label='Cuadrático: residuo medio por tramo',
)
ax.set_title('Residuales: ¿qué modelo deja error sistemático?')
ax.set_xlabel('Horas de estudio semanal')
ax.set_ylabel('Residuo (puntaje real - puntaje ajustado)')
ax.legend(loc='best')
sns.despine(ax=ax)
fig.text(
    0.5, 0.005,
    'Puntos tenues = residuos individuales. Líneas = promedio por tramo (8 tramos con'
    ' igual número de estudiantes).\nUna línea alejada de 0 indica error sistemático'
    ' del modelo.',
    ha='center', va='bottom', fontsize=8, color='#52514e',
)
fig.tight_layout(rect=(0, 0.06, 1, 1))
plt.show()

# ==============================================================================
# 5. DOS VARIABLES CUALITATIVAS COMBINADAS CON CUANTITATIVAS (Criterio 6 - 15%)
# ==============================================================================
print('\n=== 5. ESTADÍSTICAS POR CATEGORÍA (CUALITATIVAS + CUANTITATIVAS) ===')
for cuali in vars_cuali:
  print(f'\n--- Agrupado por: {cuali} ---')
  tabla_grupo = (
      df_limpio.groupby(cuali, observed=False)[vars_cuanti]
      .agg(['count', 'mean', 'median', 'std'])
      .round(2)
  )
  print(tabla_grupo)
  for v in vars_cuanti:
    medias = df_limpio.groupby(cuali, observed=True)[v].mean()
    print(
        f'  Interpretación ({v}): la media más alta está en "{medias.idxmax()}"'
        f' ({medias.max():.2f}) y la más baja en "{medias.idxmin()}"'
        f' ({medias.min():.2f}); diferencia = {medias.max() - medias.min():.2f}.'
    )

print('\n--- Combinando las DOS variables cualitativas (grade x nivel_asistencia) ---')
print(
    df_limpio.groupby(vars_cuali, observed=True)[vars_cuanti]
    .agg(['count', 'mean', 'median', 'std'])
    .round(2)
)
print('\nFrecuencias:')
print(pd.crosstab(df_limpio['grade'], df_limpio['nivel_asistencia'], margins=True, margins_name='Total'))
print('\nPorcentaje por calificación (cada fila suma 100%):')
print((pd.crosstab(df_limpio['grade'], df_limpio['nivel_asistencia'], normalize='index') * 100).round(1))

rangos = df_limpio.groupby('grade')['total_score'].agg(['min', 'max']).sort_values('min')
if (rangos['min'].values[1:] >= rangos['max'].values[:-1]).all():
  print(
      '\nNOTA: los rangos de total_score de cada grade NO se traslapan, es decir,'
      ' grade se deriva del puntaje. Por eso la diferencia de puntaje entre'
      ' calificaciones es esperada por construcción; el hallazgo informativo es'
      ' cómo cambian las horas de estudio y la asistencia entre calificaciones.'
  )

# Gráficos combinados (Boxplot por categoría y Barras agrupadas)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Boxplot de Puntaje Total según Calificación (grade) y Nivel de Asistencia
sns.boxplot(
    data=df_limpio,
    x='grade',
    y='total_score',
    hue='nivel_asistencia',
    order=sorted(df_limpio['grade'].unique()),
    ax=axes[0],
)
axes[0].set_title('Puntaje Total por Calificación y Nivel de Asistencia')
axes[0].set_xlabel('Calificación (Grade)')
axes[0].set_ylabel('Puntaje Total')
axes[0].legend(title='Nivel de Asistencia')

# Gráfico de barras: Promedio de Horas de Estudio por Grade y Nivel de Asistencia
sns.barplot(
    data=df_limpio,
    x='grade',
    y='weekly_self_study_hours',
    hue='nivel_asistencia',
    order=sorted(df_limpio['grade'].unique()),
    errorbar=None,
    ax=axes[1],
)
axes[1].set_title(
    'Promedio de Horas de Estudio por Calificación y Asistencia'
)
axes[1].set_xlabel('Calificación (Grade)')
axes[1].set_ylabel('Promedio de Horas de Estudio')
axes[1].legend(title='Nivel de Asistencia')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# GRÁFICA NUEVA 2: Horas vs. puntaje según nivel de asistencia (con recta por grupo)
# ------------------------------------------------------------------------------
if isinstance(df_limpio['nivel_asistencia'].dtype, pd.CategoricalDtype):
  niveles = list(df_limpio['nivel_asistencia'].cat.categories)
else:
  niveles = sorted(df_limpio['nivel_asistencia'].dropna().unique())

estilos = [(AZUL, 'o'), (NARANJA, 's'), (AQUA, '^')]  # máximo 3 niveles legibles
fig, ax = plt.subplots(figsize=(9, 6))
handles = []
pendientes = {}
for nivel, (color, marcador) in zip(niveles, estilos):
  g = df_limpio[df_limpio['nivel_asistencia'] == nivel]
  if len(g) < 3:
    continue
  gx, gy = g['weekly_self_study_hours'].to_numpy(), g['total_score'].to_numpy()
  m_g, b_g = np.polyfit(gx, gy, 1)
  r_g = np.corrcoef(gx, gy)[0, 1]
  pendientes[nivel] = (len(g), m_g, r_g)
  ax.scatter(gx, gy, s=30, alpha=0.45, color=color, marker=marcador,
             edgecolor='white', linewidth=0.4)
  xs = np.linspace(gx.min(), gx.max(), 50)
  ax.plot(xs, m_g * xs + b_g, color=color, linewidth=2.5)
  handles.append(
      Line2D([0], [0], color=color, marker=marcador, linewidth=2.5, markersize=8,
             label=f'{nivel}: n = {len(g)}, R = {r_g:.2f}')
  )
ax.axhline(100, color='gray', linestyle=':', linewidth=1)
ax.set_ylim(top=105)  # el puntaje máximo posible es 100; las rectas no se grafican mucho más allá
ax.set_title('Horas de estudio vs. puntaje según nivel de asistencia')
ax.set_xlabel('Horas de estudio semanal')
ax.set_ylabel('Puntaje total')
ax.legend(handles=handles, title='Nivel de asistencia', loc='lower right')
sns.despine(ax=ax)
plt.tight_layout()
plt.show()

print('\n--- RECTA DE AJUSTE POR NIVEL DE ASISTENCIA ---')
for nivel, (n_g, m_g, r_g) in pendientes.items():
  print(f'{nivel}: n = {n_g} | pendiente = {m_g:.3f} pts/hora | R = {r_g:.3f}')
if len(pendientes) >= 2:
  ms = [v[1] for v in pendientes.values()]
  rango_rel = (max(ms) - min(ms)) / abs(np.mean(ms))
  if rango_rel < 0.25:
    print(
        'Interpretación: las pendientes son parecidas, es decir, el efecto de una'
        ' hora extra de estudio sobre el puntaje es similar sin importar el nivel'
        ' de asistencia.'
    )
  else:
    print(
        'Interpretación: las pendientes difieren entre niveles de asistencia, es'
        ' decir, el efecto de una hora extra de estudio no es igual en todos los'
        ' grupos.'
    )