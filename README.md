# spectral-2

Проект для численного решения сингулярно возмущенных дробных дифференциальных уравнений.

Сейчас в репозитории есть три основных 1D-подхода:

- `FEPG-DEMM` спектральный метод в пакете [spfde](/c:/Users/danii/Desktop/Diss/spectral-2/spfde)
- `Alpha-Shishkin L1` метод на `alpha`-адаптированной сетке Шишкина в пакете [alpha_shishkin_l1](/c:/Users/danii/Desktop/Diss/spectral-2/alpha_shishkin_l1)
- `AEML-vPINN` асимптотически обогащенный вариационный PINN в пакете [aeml_vpinn](/c:/Users/danii/Desktop/Diss/spectral-2/aeml_vpinn)

Также есть набор бенчмарков в папке [benchmarks](/c:/Users/danii/Desktop/Diss/spectral-2/benchmarks).

## Структура

- [spfde](/c:/Users/danii/Desktop/Diss/spectral-2/spfde) — спектральный решатель `FEPG-DEMM`, базовый `L1`, квадратуры и вычисление функции Миттага-Леффлера
- [alpha_shishkin_l1](/c:/Users/danii/Desktop/Diss/spectral-2/alpha_shishkin_l1) — реализация `L1`-схемы на `alpha`-адаптированной сетке Шишкина
- [aeml_vpinn](/c:/Users/danii/Desktop/Diss/spectral-2/aeml_vpinn) — реализация асимптотически обогащенного вариационного `AEML-vPINN`
- [benchmarks](/c:/Users/danii/Desktop/Diss/spectral-2/benchmarks) — отдельные benchmark-сценарии
- [demo.py](/c:/Users/danii/Desktop/Diss/spectral-2/demo.py) — короткий запуск спектрального метода
- [demo_alpha_shishkin_l1.py](/c:/Users/danii/Desktop/Diss/spectral-2/demo_alpha_shishkin_l1.py) — короткий запуск метода `Alpha-Shishkin L1`
- [demo_aeml_vpinn.py](/c:/Users/danii/Desktop/Diss/spectral-2/demo_aeml_vpinn.py) — короткий запуск метода `AEML-vPINN`

## Зависимости

Нужны:

- `numpy`
- `scipy`
- `matplotlib`

Опционально:

- `pymittagleffler` для reference-проверки функции Миттага-Леффлера

## Быстрый запуск

Для Windows:

```powershell
py -3 demo.py
py -3 demo_alpha_shishkin_l1.py
py -3 demo_aeml_vpinn.py
```

## Бенчмарки

Из корня проекта:

```powershell
py -3 -m benchmarks.spectral.benchmark_spectral
py -3 -m benchmarks.alpha_shishkin_l1.benchmark_alpha_shishkin_l1
py -3 -m benchmarks.aeml_vpinn.benchmark_aeml_vpinn
py -3 -m benchmarks.two_dimensional.benchmark_2d
```

Результаты каждого сценария складываются в свою папку `results/` внутри `benchmarks/...`.

1D-бенчмарки сейчас считают две задачи:

- каноническую задачу из статьи
- дополнительную objective manufactured-задачу с точным решением `u(x)=E_alpha(-x^alpha / epsilon) + x^2`

Подробности по benchmark-структуре есть в [benchmarks/README.md](/c:/Users/danii/Desktop/Diss/spectral-2/benchmarks/README.md).
