import json
import argparse
from pathlib import Path
from datetime import datetime

def aggregate_results(input_dir: str, output_path: str):
    results_dir = Path(input_dir)
    all_results = []
    
    # Читаем все результаты из artifacts
    for result_file in results_dir.rglob("*_result.json"):
        with open(result_file) as f:
            all_results.append(json.load(f))
    
    # Статистика
    total = len(all_results)
    buy_signals = sum(1 for r in all_results if r["consensus_action"] == "BUY")
    short_signals = sum(1 for r in all_results if r["consensus_action"] == "SHORT")
    wait_signals = sum(1 for r in all_results if r["consensus_action"] == "WAIT")
    avg_confidence = sum(r["avg_confidence"] for r in all_results) / total if total else 0
    
    summary = {
        "run_date": datetime.now().isoformat(),
        "total_tickers": total,
        "signals": {
            "BUY": buy_signals,
            "SHORT": short_signals,
            "WAIT": wait_signals
        },
        "avg_confidence": round(avg_confidence, 4),
        "results": all_results,
    }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Summary: {buy_signals} BUY, {short_signals} SHORT, {wait_signals} WAIT")
    print(f"✅ Saved summary to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    aggregate_results(args.input, args.output)
