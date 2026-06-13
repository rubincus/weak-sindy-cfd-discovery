# Implementation notes

Este documento resume la organizacion interna del paquete `weak_sindy_robust`
para los experimentos weak/strong SINDy reproducibles.

## Mapa de modulos

- `grids.py`: `Grid1D`, `Dataset1D`, `Grid2D` y `Dataset2D`.
- `problems.py`: especificacion de Burgers 1D, especificacion de vorticidad Navier-Stokes 2D y coeficientes verdaderos.
- `differentiators.py`: derivadas espectrales de Fourier en 1D y 2D; inversion espectral vorticidad-velocidad para el caso incomprensible periodico.
- `solvers.py`: solver RK4 determinista para Burgers y solver pseudo-espectral RK4 para vorticidad 2D.
- `noise.py`: ruido iid, correlacionado espacial/temporal, heterocedastico, impulsivo y datos faltantes.
- `libraries.py`: biblioteca base y biblioteca grande de Burgers; biblioteca minima de vorticidad 2D.
- `test_functions.py`: familia coseno compacto.
- `strong_assembly.py`: ensamblaje fuerte con `np.gradient`, suavizado temporal y filtro espectral.
- `weak_assembly.py`: ensamblaje debil para la biblioteca base de Burgers, ensamblaje debil para la biblioteca ampliada de Burgers y ensamblaje weak-time/local-window para vorticidad 2D.
- `regressors.py`: STLSQ, ridge-STLSQ, SR3, ElasticNet y stability selection.
- `constraints.py`: restriccion explicita de difusion positiva.
- `rollouts.py`: rollout RK4 del modelo identificado, con `diffusion_policy: raw` por defecto.
- `metrics.py`: error de coeficientes, soporte, condicionamiento, energia y bootstrap.
- `experiments.py`: motor de ejecucion configurable.
- `aggregation.py`: resumen Monte Carlo con mediana, IQR e IC bootstrap.
- `plotting.py`: figuras regenerables desde CSV/NPZ cuando `matplotlib` esta instalado.
- `io.py` y `config.py`: artefactos, entorno y lectura/escritura de configuraciones.

## Compatibilidad con el script original

El caso `configs/burgers_reproduce.yaml` mantiene:

- `Nx = 128`, `Nt = 241`, `T = 1.20`, `nu = 0.075`;
- derivadas espectrales de Fourier;
- integracion RK4;
- biblioteca `[1, u, u2, ux, uux, uxx]`;
- `xi* = [0, 0, 0, 0, -1, 0.075]`;
- funciones debiles coseno compacto con `ax = 0.90`, `at = 0.22`, `nxc = 14`, `ntc = 10`;
- STLSQ con normalizacion de columnas y `max_iter = 12`.

El script heredado `mc_weak_sindy.py` se conserva, pero sus salidas ya no usan rutas absolutas Linux. Ahora escribe en `results/legacy_mc/`.

## Artefactos generados

Cada corrida de `scripts/run_experiment.py` escribe:

- `config_resolved.yaml`;
- `environment.txt`;
- `run_manifest.json`;
- `clean_solution.npz`;
- `noisy_cases/*.npz`;
- `coefficients.csv`;
- `metrics.csv`;
- `supports.csv`;
- `residuals.csv`;
- `runtimes.csv`;
- `rollouts/*.npz`;
- `summaries/mc_summary.csv`;
- `figures/*` o un `figures/README.txt` si falta `matplotlib`.

## Alcance actual

El flujo Burgers 1D esta implementado y verificado para reproduccion, Monte Carlo, baselines, ruido realista, OOD rollout, biblioteca grande y restricciones fisicas.

La biblioteca ampliada de Burgers usa integracion por partes para evitar diferenciar directamente los terminos `ux`, `uux`, `u2ux`, `uxx`, `uxxx` y `uxxxx`. Los terminos `ux2` y la identidad usada para `uuxx` requieren una derivada espacial auxiliar `ux`; por eso el experimento se reporta como un stress test de soporte bajo distractores, no como una formulacion weak totalmente libre de derivadas para todos los distractores no lineales.

La configuracion `ns2d_vorticity_minimal.yaml` tambien esta implementada. Genera una trayectoria periodica de vorticidad 2D,
`omega_t = -u omega_x - v omega_y + nu Delta omega`, con `nu = 1e-3`, recupera la velocidad mediante Biot-Savart espectral y compara `strong_stlsq_ns2d` contra `weak_stlsq_ns2d` para `sigma in {0, 0.02, 0.05}`.

El caso 2D debe interpretarse como un stress test periodico: aun no incluye paredes, terminos de borde, formulacion velocidad-presion ni espacios de prueba exactamente divergence-free.
