"""
Check which 3GPP series are missing from current collection
Compare with required series for tele_qna dataset
"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_current_specs():
    """Analyze current specifications by series"""
    json_dir = Path("3GPP_JSON_DOC/processed_json_v2")

    if not json_dir.exists():
        print(f"❌ Directory not found: {json_dir}")
        return None

    # Count by series
    series_count = defaultdict(int)
    all_specs = []

    for json_file in json_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            spec_id = data.get('metadata', {}).get('specification_id', json_file.stem)
            all_specs.append(spec_id)

            # Extract series number (e.g., "TS 23.501" -> "23")
            if spec_id.startswith('ts_'):
                parts = spec_id.split('_')
                if len(parts) >= 2:
                    series_num = parts[1].split('.')[0]
                    series_count[series_num] += 1

        except Exception as e:
            print(f"⚠️  Error processing {json_file.name}: {e}")

    return series_count, all_specs


def print_series_analysis(series_count):
    """Print analysis of current series coverage"""
    print("\n" + "=" * 70)
    print("CURRENT 3GPP SERIES COVERAGE")
    print("=" * 70)

    print(f"\nTotal series covered: {len(series_count)}")
    print(f"\nBreakdown by series:")
    print("-" * 70)

    # Sort by series number
    for series in sorted(series_count.keys()):
        count = series_count[series]
        series_name = get_series_name(series)
        print(f"  TS {series}.xxx - {count:3d} specs - {series_name}")


def get_series_name(series_num):
    """Get human-readable name for each series"""
    series_names = {
        '21': 'Requirements',
        '22': 'Service aspects',
        '23': 'Technical realization',
        '24': 'Signaling protocols',
        '25': 'Radio aspects (UMTS)',
        '26': 'Codecs',
        '27': 'Data',
        '28': 'Signaling protocols (CT1)',
        '29': 'Signaling protocols (CT3)',
        '31': 'SIM/USIM',
        '32': 'OAM',
        '33': 'Security',
        '34': 'UE test specs',
        '35': 'Security algorithms',
        '36': 'LTE (E-UTRAN)',
        '37': 'Multiple radio access',
        '38': '5G NR (New Radio)',
    }
    return series_names.get(series_num, 'Unknown')


def check_missing_critical_specs(all_specs):
    """Check for critical specs needed for tele_qna"""
    print("\n" + "=" * 70)
    print("CRITICAL SPECS STATUS (for tele_qna)")
    print("=" * 70)

    critical_specs = {
        # Priority 1: TS 38.xxx (5G NR)
        'ts_38.300': 'NR Overall Description',
        'ts_38.401': 'NG-RAN Architecture',
        'ts_38.413': 'NGAP Protocol',
        'ts_38.423': 'XnAP Protocol',
        'ts_38.473': 'F1AP Protocol',
        'ts_38.211': 'Physical channels and modulation',
        'ts_38.212': 'Multiplexing and channel coding',
        'ts_38.213': 'Physical layer procedures for control',
        'ts_38.214': 'Physical layer procedures for data',
        'ts_38.215': 'Physical layer measurements',
        'ts_38.321': 'MAC protocol',
        'ts_38.322': 'RLC protocol',
        'ts_38.323': 'PDCP protocol',
        'ts_38.331': 'RRC protocol',

        # Priority 2: TS 24.xxx (NAS)
        'ts_24.501': 'NAS protocol for 5GS',
        'ts_24.502': 'Access via non-3GPP',
        'ts_24.008': 'Core network protocols',
        'ts_24.301': 'NAS protocol for EPS',

        # Priority 3: Additional TS 29.xxx
        'ts_29.501': 'Principles and Guidelines for Services',
        'ts_29.505': 'UDR Subscription Data',
        'ts_29.507': 'Access and Mobility Policy Control',
        'ts_29.508': 'Session Management Event Exposure',
        'ts_29.510': 'Network function repository',
        'ts_29.512': 'Session Management Policy Control',
        'ts_29.514': 'Policy Authorization Service',

        # Priority 4: Security & LTE
        'ts_33.501': 'Security architecture for 5GS',
        'ts_33.401': 'SAE Security architecture',
        'ts_36.300': 'E-UTRA Overall description',
        'ts_36.401': 'E-UTRAN Architecture',
        'ts_36.413': 'S1AP Protocol',

        # Priority 5: Requirements
        'ts_22.261': '5G service requirements',
        'ts_37.340': 'Multi-connectivity',
    }

    missing = []
    found = []

    for spec_id, description in critical_specs.items():
        if spec_id in all_specs:
            found.append((spec_id, description))
        else:
            missing.append((spec_id, description))

    print(f"\n✅ Found: {len(found)}/{len(critical_specs)} critical specs")
    print(f"❌ Missing: {len(missing)}/{len(critical_specs)} critical specs")

    if missing:
        print("\n" + "-" * 70)
        print("MISSING CRITICAL SPECS:")
        print("-" * 70)

        # Group by series
        by_series = defaultdict(list)
        for spec_id, desc in missing:
            series = spec_id.split('_')[1].split('.')[0]
            by_series[series].append((spec_id, desc))

        for series in sorted(by_series.keys()):
            print(f"\n  Series {series}:")
            for spec_id, desc in by_series[series]:
                print(f"    • {spec_id.upper():15s} - {desc}")

    return missing


def generate_download_script(missing_specs):
    """Generate script to download missing specs"""
    print("\n" + "=" * 70)
    print("DOWNLOAD INSTRUCTIONS")
    print("=" * 70)

    print("\nDownload missing specs from 3GPP FTP:")
    print("https://www.3gpp.org/ftp/Specs/archive/\n")

    # Group by series for easier download
    by_series = defaultdict(list)
    for spec_id, desc in missing_specs:
        series = spec_id.split('_')[1].split('.')[0]
        spec_num = spec_id.split('_')[1]
        by_series[series].append(spec_num)

    print("URLs to visit (by priority):\n")

    for series in sorted(by_series.keys()):
        print(f"Series {series}:")
        for spec_num in sorted(by_series[series]):
            # Format: 23_series/23.501/ for example
            print(f"  https://www.3gpp.org/ftp/Specs/archive/{series}_series/{spec_num}/")

    print("\nSteps:")
    print("1. Visit each URL above")
    print("2. Download latest .docx version (e.g., 23501-h40.docx)")
    print("3. Save to: document_processing/input/")
    print("4. Run: cd document_processing && python process_documents.py")
    print("5. Re-initialize KG: python orchestrator.py init-kg")


def main():
    print("\n" + "=" * 70)
    print("3GPP DOCUMENT COLLECTION ANALYSIS")
    print("=" * 70)

    # Analyze current specs
    result = analyze_current_specs()
    if result is None:
        return 1

    series_count, all_specs = result

    # Print series breakdown
    print_series_analysis(series_count)

    # Check critical specs
    missing = check_missing_critical_specs(all_specs)

    # Generate download instructions
    if missing:
        generate_download_script(missing)
    else:
        print("\n" + "=" * 70)
        print("✅ ✅ ✅ ALL CRITICAL SPECS PRESENT ✅ ✅ ✅")
        print("=" * 70)
        print("\nYou can proceed with Phase 1: Baseline Evaluation")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
