# Fossil Fuel Dependency and Transition Priorities in China

This repository contains the core analysis code for *Assessing structural fossil fuel dependency and transition priorities in China*. The study examines fossil fuel dependency (FFD) and annual state transitions across 30 provincial-level regions in China from 2001 to 2020. It also analyzes external reliance (ERI) and supply responsibility (SRI) in 2020 and examines thermal power inputs and electricity balances in Hebei and Shandong.

## Running the Analysis with the Included Workbook

Once the dependencies are installed, the included [supplementary workbook](supplementary/Supplementary_Data.xlsx) provides everything needed to run the command below. Use Python 3.11 and run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
python -m analysis supplement --workbook supplementary/Supplementary_Data.xlsx --output outputs/supplement
```

The command reads 600 province-year FFD records from the workbook and produces annual FFD summaries, transition matrices, annual state shares, and the sensitivity results for Hainan's dominant-category persistence. Outputs are saved in `outputs/supplement/`. The workbook also contains the provincial network case results, ERI and SRI values, and energy accounts for Hebei and Shandong.

## Data Sources and Scope of Reproduction

- The historical provincial energy balances for 2001–2020 and the regional energy balance tables in physical units for 2021–2022 come from the *China Energy Statistical Yearbook*. [National Bureau of Statistics energy statistics FAQ](https://www.stats.gov.cn/hd/cjwtjd/202302/t20230207_1902276.html) · [Guidance on accessing regional energy balance tables](https://www.stats.gov.cn/hd/lyzx/zxgk/202507/t20250703_1960323.html)
- The 2023 regional energy balance tables in standard-coal equivalents are available in the *China Energy Statistical Yearbook 2024*. [China Statistics Press table of contents](https://www.zgtjcbs.com/tongjinianjian/5893) · [National Bureau of Statistics explanation of the 2024 edition](https://www.stats.gov.cn/hd/lyzx/zxgk/202602/t20260209_1962575.html)
- The 2018–2024 industrial and electricity series for Hebei and Shandong use reported values from a compiled provincial database. The release URL is still to be verified. The series used in the case studies are included in the workbook's `S5_long_series` worksheet. [National Bureau of Statistics guide to regional statistics](https://www.stats.gov.cn/zs/tjws/tjzn/202301/t20230101_1903751.html)

The full analysis starts from harmonized inputs. Reproducing it from the original yearbook tables requires identifying the relevant editions, aligning historical account categories, standardizing units, and applying the historical data corrections. The official links provide access to information on the sources. The edition-by-edition inventory and full data-cleaning workflow are outside the repository's current scope.

## Repository Files

| Path | Contents |
| --- | --- |
| `analysis/network.py` | Provincial network construction, dominant energy carriers, national subnetwork diagnostics, and three provincial network comparisons |
| `analysis/ffd.py` | FFD components, entropy-weighted TOPSIS, completion of missing records, and the Hainan sensitivity analysis |
| `analysis/transitions.py` | FFD state thresholds, consecutive-year transition events, and transition matrices |
| `analysis/roles.py` | ERI, SRI, supply contributions, provincial system roles, and alternative estimates |
| `analysis/accounts.py` | Thermal power inputs, electricity balances, and industrial and electricity series |
| `supplementary/Supplementary_Data.xlsx` | Supplementary data workbook with the original worksheet names |

## Citation

If you use this code, please contact the authors for citation details. For questions about the code, please contact the author at the address below.

[132goodhao@gmail.com](mailto:132goodhao@gmail.com)

## License

This project is for academic research purposes.
