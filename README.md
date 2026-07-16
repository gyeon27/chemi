# DFT Electronic Energy Crystal Structure Dataset

이 프로젝트는 OQMD API에서 원소별 BCC/FCC/HCP 구조 에너지를 수집하고, 실제 상온 결정구조 및 원소 물성과 병합해 분석용 CSV를 생성합니다.

## 폴더

- `data/reference/actual_structures.csv`: 실제 상온 결정구조
- `data/reference/element_properties.csv`: 원소 기본 물성
- `data/reference/phase_transitions.csv`: 상전이 정보
- `scripts/build_dataset.py`: OQMD 호출, 파싱, 병합, Delta E 계산
- `scripts/analyze_dataset.py`: 정확도, 그래프, 회귀/Feature Importance 분석
- `output/`: 생성 CSV와 그림 저장 위치

## 실행

```powershell
pip install -r requirements.txt
python scripts/build_dataset.py --elements Fe Cu Ti Ni Co Zn Zr Mo W Cr V Nb Ta Hf Sc Y Ag Au Pt Pd Al Mg Ca Li Na K Ba Sr --timeout 120 --retries 4 --delay 1.5 --output output/final_dataset.csv
python scripts/analyze_dataset.py --input output/final_dataset.csv --fig-dir output/figures
```

OQMD가 느려 특정 원소에서 timeout이 나면 같은 명령을 다시 실행하면 됩니다. 이미 성공한 원소는 `data/raw/oqmd_cache`의 캐시를 재사용하므로 처음부터 다시 API를 호출하지 않습니다.

실패한 원소만 다시 받아 캐시를 채우려면:

```powershell
python scripts/build_dataset.py --elements Ni Nb Ta Hf Sc Na K Ba Sr --timeout 180 --retries 6 --delay 10 --output output/retry_failed_elements.csv
```

캐시에 저장된 전체 원소를 모아 최종 분석용 CSV를 다시 만들려면:

```powershell
python scripts/finalize_from_cache.py
```

이 명령은 두 파일을 생성합니다.

- `output/final_dataset.csv`: 전체 병합 데이터
- `output/delta_e_transition_dataset.csv`: Delta E, 구조별 에너지, 상전이 여부 중심 분석 데이터

OQMD `formationenergy` API는 실제 total electronic energy 대신 `delta_e`를 제공하는 경우가 많습니다. `scripts/finalize_from_cache.py`는 최종 분석 CSV를 만들 때 이 `delta_e`를 구조 간 상대 에너지 비교용으로 사용합니다.

API 없이 파서만 시험하려면:

```powershell
python scripts/build_dataset.py --elements Fe Cu --mock-json data/sample_oqmd_response.json --output output/mock_final_dataset.csv
```

OQMD 서버 응답 필드가 바뀌거나 일부 구조명이 다르게 들어오는 경우가 있어, 파서는 `energy_per_atom`, `energy`, `total_energy`, `delta_e`, `name`, `prototype`, `spacegroup`, `natoms` 등 여러 후보 필드를 방어적으로 처리합니다.

주의: `formation_energy` 또는 `delta_e`는 Electronic Energy 자체가 아니므로 기본값에서는 에너지로 사용하지 않습니다. API 응답에 원자당 총에너지가 없고 예비 분석만 하고 싶을 때는 `--allow-formation-energy-fallback` 옵션을 붙입니다.

## 핵심 계산

- `DFT Stable Structure`: `E_BCC`, `E_FCC`, `E_HCP` 중 최솟값 구조
- `Delta_E`: 두 번째로 낮은 에너지와 최저 에너지의 차이
- `Delta_E_BCC_FCC`: `E_BCC - E_FCC`
- `Delta_E_BCC_HCP`: `E_BCC - E_HCP`
- `Delta_E_FCC_HCP`: `E_FCC - E_HCP`

`Delta_E`는 연구의 핵심 변수로, 값이 작을수록 구조 간 에너지 차이가 작다는 뜻입니다.
